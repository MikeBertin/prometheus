/* ============================================================================
 * PROMETHEUS — runq.c
 * The same transformer as run.c, with int8-quantized weights (Q8_0).
 *
 * Why quantize? matmul — 95% of inference — is MEMORY-BANDWIDTH bound: the
 * CPU spends its time streaming weight bytes, not multiplying. Store each
 * weight in 1 byte instead of 4 and you move 4x less memory (and the file
 * shrinks 4x). The catch: int8 can only represent 256 distinct values, so
 * every weight is stored as (int8 value x fp32 scale) and the trick is
 * choosing scales that lose as little precision as possible.
 *
 * Q8_0, group-wise, symmetric:
 *   - split each tensor into groups of GS consecutive values (GS=64 here)
 *   - per group: scale = max|w| / 127, then q = round(w / scale)
 *   - dequantize: w ~= q * scale
 * Small groups mean one outlier only poisons 63 neighbours, not the tensor.
 *
 * What stays fp32 and why:
 *   - norm gains + all activations between ops  (tiny; precision matters)
 *   - the KV cache                              (accumulates over the context)
 *   - accumulators inside matmul               (int32 for the int8 dots,
 *                                                fp32 across groups)
 * Weights (and the activations feeding each matmul, quantized on the fly)
 * are int8. This mirrors llama2.c's runq.c and the "W8A8"-style schemes.
 *
 * Checkpoint format v2 ("ak42"): 256-byte header {magic, version, 7 config
 * ints, shared_classifier flag, group_size}, then fp32 norm weights, then
 * each quantized tensor as [int8 q[]][fp32 s[]].
 *
 * Everything not about quantization (tokenizer, sampler, RoPE, attention,
 * generation loop) is IDENTICAL to run.c — see that file for the annotated
 * walkthrough of the transformer itself.
 * ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ----------------------------------------------------------------------------
 * 0. QUANTIZATION PRIMITIVES
 * -------------------------------------------------------------------------- */

int GS = 0; // global group size, read from the checkpoint header

typedef struct {
    int8_t* q; // quantized values
    float* s;  // one fp32 scale per group of GS values
} QuantizedTensor;

/* w ~= q * s — the lossy inverse of quantize(). */
void dequantize(QuantizedTensor* qx, float* x, int n) {
    for (int i = 0; i < n; i++) x[i] = qx->q[i] * qx->s[i / GS];
}

/* Symmetric max-abs quantization, one scale per group: the largest value in
 * each group maps to +/-127 and everything else scales linearly. Used at
 * runtime on ACTIVATIONS right before each matmul (weights were quantized
 * once, offline, by export.py). */
void quantize(QuantizedTensor* qx, float* x, int n) {
    int num_groups = n / GS;
    for (int group = 0; group < num_groups; group++) {
        float wmax = 0.0f;
        for (int i = 0; i < GS; i++) {
            float val = fabsf(x[group * GS + i]);
            if (val > wmax) wmax = val;
        }
        float scale = wmax / 127.0f;
        if (scale == 0.0f) scale = 1.0f; // all-zero group: any scale works
        qx->s[group] = scale;
        for (int i = 0; i < GS; i++)
            qx->q[group * GS + i] = (int8_t)roundf(x[group * GS + i] / scale);
    }
}

/* THE payoff of the whole file: matmul where the inner dot product runs in
 * int8 with an int32 accumulator. Per group of GS values we accumulate pure
 * integer products, then apply the two scales (weight's and activation's)
 * once. 4x less memory traffic than the fp32 version in run.c. */
void matmul(float* xout, QuantizedTensor* x, QuantizedTensor* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        int in = i * n;
        for (int j = 0; j <= n - GS; j += GS) {
            int32_t ival = 0;
            for (int k = 0; k < GS; k++)
                ival += ((int32_t)x->q[j + k]) * ((int32_t)w->q[in + j + k]);
            val += ((float)ival) * w->s[(in + j) / GS] * x->s[j / GS];
        }
        xout[i] = val;
    }
}

/* ----------------------------------------------------------------------------
 * 1. THE MODEL (config identical to run.c; weights now QuantizedTensors)
 * -------------------------------------------------------------------------- */

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Config;

typedef struct {
    // norms stay fp32 (tiny, precision-sensitive)
    float* rms_att_weight;        // (n_layers, dim)
    float* rms_ffn_weight;        // (n_layers, dim)
    float* rms_final_weight;      // (dim,)
    // everything that feeds a matmul is int8
    QuantizedTensor* q_tokens;    // (vocab_size, dim) — quantized embeddings
    float* token_embedding_table; // same, dequantized once at load for lookup
    QuantizedTensor* wq;          // (n_layers, dim, dim)
    QuantizedTensor* wk;          // (n_layers, dim, kv_dim)
    QuantizedTensor* wv;          // (n_layers, dim, kv_dim)
    QuantizedTensor* wo;          // (n_layers, dim, dim)
    QuantizedTensor* w1;          // (n_layers, hidden_dim, dim)
    QuantizedTensor* w2;          // (n_layers, dim, hidden_dim)
    QuantizedTensor* w3;          // (n_layers, hidden_dim, dim)
    QuantizedTensor* wcls;        // (vocab_size, dim) — may alias q_tokens
} TransformerWeights;

typedef struct {
    float* x;            // (dim,) residual stream
    float* xb;           // (dim,)
    float* xb2;          // (dim,)
    float* hb;           // (hidden_dim,)
    float* hb2;          // (hidden_dim,)
    QuantizedTensor xq;  // (dim,) quantized activation buffer for matmuls
    QuantizedTensor hq;  // (hidden_dim,) same, ffn-width
    float* q;            // (dim,) query
    float* att;          // (n_heads, seq_len)
    float* logits;       // (vocab_size,)
    float* key_cache;    // (n_layers, seq_len, kv_dim) — fp32, deliberately
    float* value_cache;  // (n_layers, seq_len, kv_dim)
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    void* data;
    ssize_t file_size;
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x    = calloc(p->dim, sizeof(float));
    s->xb   = calloc(p->dim, sizeof(float));
    s->xb2  = calloc(p->dim, sizeof(float));
    s->hb   = calloc(p->hidden_dim, sizeof(float));
    s->hb2  = calloc(p->hidden_dim, sizeof(float));
    s->xq.q = calloc(p->dim, sizeof(int8_t));
    s->xq.s = calloc(p->dim / GS, sizeof(float));
    s->hq.q = calloc(p->hidden_dim, sizeof(int8_t));
    s->hq.s = calloc(p->hidden_dim / GS, sizeof(float));
    s->q    = calloc(p->dim, sizeof(float));
    s->att  = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache   = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->xq.q || !s->xq.s
        || !s->hq.q || !s->hq.s || !s->q || !s->att || !s->logits
        || !s->key_cache || !s->value_cache) {
        fprintf(stderr, "malloc failed for run state\n"); exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->xq.q); free(s->xq.s); free(s->hq.q); free(s->hq.s);
    free(s->q); free(s->att); free(s->logits);
    free(s->key_cache); free(s->value_cache);
}

/* Build n QuantizedTensor views over the mmap'd blob: each tensor is laid out
 * as [size_each int8s][size_each/GS fp32 scales], back to back. */
QuantizedTensor* init_quantized_tensors(void** ptr, int n, int size_each) {
    void* p = *ptr;
    QuantizedTensor* res = malloc(n * sizeof(QuantizedTensor));
    if (!res) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); }
    for (int i = 0; i < n; i++) {
        res[i].q = (int8_t*)p;  p = (int8_t*)p + size_each;
        res[i].s = (float*)p;   p = (float*)p + size_each / GS;
    }
    *ptr = p;
    return res;
}

void memory_map_weights(TransformerWeights* w, Config* p, void* ptr, uint8_t shared_classifier) {
    int head_size = p->dim / p->n_heads;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    unsigned long long L = p->n_layers;

    // fp32 norm gains come first...
    float* fptr = (float*)ptr;
    w->rms_att_weight = fptr;   fptr += L * p->dim;
    w->rms_ffn_weight = fptr;   fptr += L * p->dim;
    w->rms_final_weight = fptr; fptr += p->dim;

    // ...then the quantized tensors
    void* qptr = (void*)fptr;
    w->q_tokens = init_quantized_tensors(&qptr, 1, p->vocab_size * p->dim);
    // dequantize the embedding table once: token lookup is a row COPY into
    // the fp32 residual stream, not a matmul, so we want it in fp32
    w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    if (!w->token_embedding_table) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); }
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&qptr, L, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&qptr, L, p->dim * kv_dim);
    w->wv = init_quantized_tensors(&qptr, L, p->dim * kv_dim);
    w->wo = init_quantized_tensors(&qptr, L, (p->n_heads * head_size) * p->dim);
    w->w1 = init_quantized_tensors(&qptr, L, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&qptr, L, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&qptr, L, p->dim * p->hidden_dim);
    w->wcls = shared_classifier ? w->q_tokens
                                : init_quantized_tensors(&qptr, 1, p->dim * p->vocab_size);
}

void read_checkpoint(char* path, Config* config, TransformerWeights* weights,
                     int* fd, void** data, ssize_t* file_size) {
    FILE* file = fopen(path, "rb");
    if (!file) { fprintf(stderr, "couldn't open %s\n", path); exit(EXIT_FAILURE); }

    uint32_t magic; int version;
    if (fread(&magic, sizeof(uint32_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (magic != 0x616b3432) { fprintf(stderr, "bad magic — not an ak42 v2 checkpoint (use export.py --q80)\n"); exit(EXIT_FAILURE); }
    if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (version != 2) { fprintf(stderr, "unsupported version %d (want 2)\n", version); exit(EXIT_FAILURE); }
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    uint8_t shared_classifier;
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    int group_size;
    if (fread(&group_size, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    GS = group_size;
    // groups must align with matrix rows, or matmul's scale indexing is off
    // by half a group on every odd row (see export.py) — refuse loudly
    if (config->dim % GS != 0 || config->hidden_dim % GS != 0) {
        fprintf(stderr, "group size %d does not divide dim %d / hidden_dim %d — re-export\n",
                GS, config->dim, config->hidden_dim);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fclose(file);

    *fd = open(path, O_RDONLY);
    if (*fd == -1) { fprintf(stderr, "open failed\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(EXIT_FAILURE); }

    void* weights_ptr = (char*)*data + 256; // the header is padded to 256 bytes
    memory_map_weights(weights, config, weights_ptr, shared_classifier);
}

void build_transformer(Transformer* t, char* checkpoint_path) {
    read_checkpoint(checkpoint_path, &t->config, &t->weights,
                    &t->fd, &t->data, &t->file_size);
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    free(t->weights.token_embedding_table);
    free(t->weights.q_tokens);
    free(t->weights.wq); free(t->weights.wk); free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1); free(t->weights.w2); free(t->weights.w3);
    if (t->weights.wcls != t->weights.q_tokens) free(t->weights.wcls);
    if (t->data != MAP_FAILED && t->data != NULL) munmap(t->data, t->file_size);
    if (t->fd != -1) close(t->fd);
    free_run_state(&t->state);
}

/* ----------------------------------------------------------------------------
 * 2. FP32 PRIMITIVES (identical to run.c)
 * -------------------------------------------------------------------------- */

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss = 1.0f / sqrtf(ss / size + 1e-5f);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

/* ----------------------------------------------------------------------------
 * 3. THE FORWARD PASS — same shape as run.c, but every matmul is preceded by
 *    an activation quantize() and runs in int8.
 * -------------------------------------------------------------------------- */
float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float* x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

    for (unsigned long long l = 0; l < (unsigned)p->n_layers; l++) {

        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // quantize the normalized activations once, reuse for q/k/v matmuls
        quantize(&s->xq, s->xb, dim);
        int loff = l * p->seq_len * kv_dim;
        float* k = s->key_cache   + loff + pos * kv_dim;
        float* v = s->value_cache + loff + pos * kv_dim;
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(k,    &s->xq, w->wk + l, dim, kv_dim);
        matmul(v,    &s->xq, w->wv + l, dim, kv_dim);

        // RoPE — identical to run.c (fp32; rotation is not a matmul)
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val), fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int vv = 0; vv < rotn; vv++) {
                float* vec = vv == 0 ? s->q : k;
                float v0 = vec[i], v1 = vec[i + 1];
                vec[i]     = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // attention — identical to run.c; scores/values stay fp32 because the
        // KV cache is fp32 (quantizing IT is the next frontier: "KV cache
        // quantization" is exactly this trade at LLM scale)
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* kt = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += q[i] * kt[i];
                att[t] = score / sqrtf(head_size);
            }
            softmax(att, pos + 1);
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* vt = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) xb[i] += a * vt[i];
            }
        }

        quantize(&s->xq, s->xb, dim);
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb2[i];

        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        quantize(&s->xq, s->xb, dim);
        matmul(s->hb,  &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

/* ----------------------------------------------------------------------------
 * 4–6. TOKENIZER, SAMPLER, GENERATION — byte-identical to run.c.
 * See run.c for the annotated versions; kept inline so runq is standalone.
 * -------------------------------------------------------------------------- */

typedef struct { char* str; int id; } TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex* sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

int compare_tokens(const void* a, const void* b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = malloc(vocab_size * sizeof(char*));
    t->vocab_scores = malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE* file = fopen(path, "rb");
    if (!file) { fprintf(stderr, "couldn't load tokenizer %s\n", path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { exit(EXIT_FAILURE); }
        if (fread(&len, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
        t->vocab[i] = malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab); free(t->vocab_scores); free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char* piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)
        piece = (char*)t->byte_pieces + byte_val * 2;
    return piece;
}

void safe_printf(char* piece) {
    if (piece == NULL || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char b = piece[0];
        if (!(isprint(b) || isspace(b))) return;
    }
    printf("%s", piece);
}

int str_lookup(char* str, TokenIndex* sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex* res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char* text, int8_t bos, int8_t eos, int* tokens, int* n_tokens) {
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char* str_buffer = malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    for (char* c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++)
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
        }
        str_len = 0;
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < (*n_tokens - 1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

typedef struct { float prob; int index; } ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0; float max_p = probabilities[0];
    for (int i = 1; i < n; i++)
        if (probabilities[i] > max_p) { max_i = i; max_p = probabilities[i]; }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) return i;
    }
    return n - 1;
}

int compare_probindex(const void* a, const void* b) {
    float pa = ((ProbIndex*)a)->prob, pb = ((ProbIndex*)b)->prob;
    return (pa < pb) ? 1 : (pa > pb) ? -1 : 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare_probindex);

    float cumulative_prob = 0.0f; int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) { last_idx = i; break; }
    }
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) return probindex[i].index;
    }
    return probindex[last_idx].index;
}

void build_sampler(Sampler* s, int vocab_size, float temperature, float topp, unsigned long long seed) {
    s->vocab_size = vocab_size;
    s->temperature = temperature;
    s->topp = topp;
    s->rng_state = seed;
    s->probindex = malloc(vocab_size * sizeof(ProbIndex));
}
void free_sampler(Sampler* s) { free(s->probindex); }

unsigned int random_u32(unsigned long long* state) {
    *state ^= *state >> 12; *state ^= *state << 25; *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long* state) { return (random_u32(state) >> 8) / 16777216.0f; }

int sample(Sampler* s, float* logits) {
    if (s->temperature == 0.0f) return sample_argmax(logits, s->vocab_size);
    for (int q = 0; q < s->vocab_size; q++) logits[q] /= s->temperature;
    softmax(logits, s->vocab_size);
    float coin = random_f32(&s->rng_state);
    if (s->topp <= 0 || s->topp >= 1) return sample_mult(logits, s->vocab_size, coin);
    return sample_topp(logits, s->vocab_size, s->topp, s->probindex, coin);
}

long time_in_ms(void) {
    struct timespec time; clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void generate(Transformer* t, Tokenizer* tok, Sampler* s, char* prompt, int steps) {
    char* empty = "";
    if (prompt == NULL) prompt = empty;

    int* prompt_tokens = malloc((strlen(prompt) + 3) * sizeof(int));
    int num_prompt_tokens = 0;
    encode(tok, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "prompt produced no tokens\n"); exit(EXIT_FAILURE);
    }

    long start = 0;
    int next, pos = 0;
    int token = prompt_tokens[0];
    while (pos < steps) {
        float* logits = forward(t, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(s, logits);
        }
        pos++;

        if (next == 1) break;

        char* piece = decode(tok, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) start = time_in_ms();
    }
    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "\nachieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }
    free(prompt_tokens);
}

/* ----------------------------------------------------------------------------
 * 7. CLI (identical to run.c)
 * -------------------------------------------------------------------------- */
#ifndef PROMETHEUS_LIB

void error_usage(void) {
    fprintf(stderr, "Usage: runq <checkpoint.q80> [options]\n");
    fprintf(stderr, "Example: runq model_q80.bin -z tokenizer.bin -t 0.9 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0 (0 = greedy)\n");
    fprintf(stderr, "  -p <float>  top-p / nucleus in [0,1], default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run, default 256\n");
    fprintf(stderr, "  -i <string> prompt string\n");
    fprintf(stderr, "  -z <string> path to tokenizer.bin, default tokenizer.bin\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
    char* checkpoint_path = NULL;
    char* tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    char* prompt = NULL;
    unsigned long long rng_seed = 0;

    if (argc >= 2) checkpoint_path = argv[1]; else error_usage();
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) error_usage();
        if (argv[i][0] != '-' || strlen(argv[i]) != 2) error_usage();
        switch (argv[i][1]) {
            case 't': temperature = atof(argv[i + 1]); break;
            case 'p': topp = atof(argv[i + 1]); break;
            case 's': rng_seed = atoi(argv[i + 1]); break;
            case 'n': steps = atoi(argv[i + 1]); break;
            case 'i': prompt = argv[i + 1]; break;
            case 'z': tokenizer_path = argv[i + 1]; break;
            default: error_usage();
        }
    }
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || topp > 1.0) topp = 0.9;

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len)
        steps = transformer.config.seq_len;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}

#endif /* PROMETHEUS_LIB */
