/* ============================================================================
 * PROMETHEUS — run.c
 * A Llama-2 style transformer, forward pass only, in pure C.
 *
 * "I gave them fire." — Prometheus, Aeschylus
 *
 * This is an *inference* engine. It loads a pre-trained checkpoint (a flat
 * file of float32 weights) and a tokenizer, then autoregressively generates
 * text one token at a time. There is no autograd, no training — every line
 * here is the math that turns a sequence of token IDs into a probability
 * distribution over the next token.
 *
 * Architecture (decoder-only transformer, Llama-2 flavor):
 *   token id -> embedding lookup
 *   for each of n_layers:
 *       RMSNorm -> multi-head self-attention (with RoPE + KV cache) -> residual
 *       RMSNorm -> SwiGLU feed-forward                              -> residual
 *   RMSNorm -> linear classifier -> logits over the vocabulary
 *
 * The checkpoint binary format is identical to Karpathy's llama2.c, so the
 * official stories15M / stories42M / stories110M checkpoints load as-is. That
 * gives us a known-good oracle: if this file generates coherent TinyStories
 * text, the forward pass is correct.
 * ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ----------------------------------------------------------------------------
 * 1. THE MODEL: config, weights, and per-step activation buffers
 * -------------------------------------------------------------------------- */

/* The hyperparameters of the network. These 7 ints are the file header. */
typedef struct {
    int dim;        // embedding / residual-stream width (e.g. 288)
    int hidden_dim; // inner width of the feed-forward network (e.g. 768)
    int n_layers;   // number of transformer blocks (e.g. 6)
    int n_heads;    // number of attention heads for queries
    int n_kv_heads; // heads for keys/values (< n_heads => grouped-query attn)
    int vocab_size; // number of distinct tokens (32000 for the Llama tokenizer)
    int seq_len;    // maximum context length the model was trained for
} Config;

/* Pointers into the memory-mapped weight blob. Nothing is copied: each field
 * points at the right offset inside the mmap'd file. */
typedef struct {
    float* token_embedding_table; // (vocab_size, dim) — row per token
    // attention RMSNorm gains
    float* rms_att_weight;        // (n_layers, dim)
    // attention projection matrices
    float* wq;                    // (n_layers, dim, n_heads * head_size)
    float* wk;                    // (n_layers, dim, n_kv_heads * head_size)
    float* wv;                    // (n_layers, dim, n_kv_heads * head_size)
    float* wo;                    // (n_layers, n_heads * head_size, dim)
    // feed-forward RMSNorm gains
    float* rms_ffn_weight;        // (n_layers, dim)
    // SwiGLU feed-forward matrices
    float* w1;                    // (n_layers, hidden_dim, dim)  gate
    float* w2;                    // (n_layers, dim, hidden_dim)  down
    float* w3;                    // (n_layers, hidden_dim, dim)  up
    // final RMSNorm + classifier
    float* rms_final_weight;      // (dim,)
    float* wcls;                  // (vocab_size, dim) — often tied to embeddings
} TransformerWeights;

/* Scratch buffers reused every forward step, plus the KV cache. The KV cache
 * is what makes autoregressive generation cheap: keys and values for past
 * positions are computed once and remembered, so step t only does O(t) work
 * for attention instead of recomputing the whole prefix. */
typedef struct {
    float* x;      // (dim,)  the residual stream — the "thought vector"
    float* xb;     // (dim,)  a normalized/temp copy of x
    float* xb2;    // (dim,)  another temp
    float* hb;     // (hidden_dim,)  ffn hidden buffer 1
    float* hb2;    // (hidden_dim,)  ffn hidden buffer 2
    float* q;      // (dim,)  query vector for the current token
    float* att;    // (n_heads, seq_len)  attention scores per head
    float* logits; // (vocab_size,)  output distribution (pre-softmax)
    // KV cache: every layer keeps keys/values for every past position.
    float* key_cache;   // (n_layers, seq_len, kv_dim)
    float* value_cache; // (n_layers, seq_len, kv_dim)
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    // bookkeeping for the memory mapping so we can clean up
    int fd;
    float* data;
    ssize_t file_size;
} Transformer;

/* Allocate the activation/cache buffers. calloc so they start zeroed. */
void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x      = calloc(p->dim, sizeof(float));
    s->xb     = calloc(p->dim, sizeof(float));
    s->xb2    = calloc(p->dim, sizeof(float));
    s->hb     = calloc(p->hidden_dim, sizeof(float));
    s->hb2    = calloc(p->hidden_dim, sizeof(float));
    s->q      = calloc(p->dim, sizeof(float));
    s->att    = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache   = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q ||
        !s->att || !s->logits || !s->key_cache || !s->value_cache) {
        fprintf(stderr, "malloc failed for run state\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->q); free(s->att); free(s->logits);
    free(s->key_cache); free(s->value_cache);
}

/* Lay the weight pointers over the raw float blob. The order here MUST match
 * the order the Python exporter wrote them. This is the legacy llama2.c
 * layout, which also stores precomputed RoPE tables we skip (we recompute
 * RoPE on the fly below — it's cheap and keeps the code self-contained). */
void memory_map_weights(TransformerWeights* w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;

    w->token_embedding_table = ptr; ptr += (unsigned long long)p->vocab_size * p->dim;
    w->rms_att_weight = ptr;        ptr += n_layers * p->dim;
    w->wq = ptr;                    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;                    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;                    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;                    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;        ptr += n_layers * p->dim;
    w->w1 = ptr;                    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;                    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;                    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;      ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip (legacy) RoPE freq_cis_real
    ptr += p->seq_len * head_size / 2; // skip (legacy) RoPE freq_cis_imag
    // If the embedding is tied to the classifier, reuse it; else it follows.
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

/* Read the checkpoint: parse the header, mmap the rest, point the weights. */
void read_checkpoint(char* path, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE* file = fopen(path, "rb");
    if (!file) { fprintf(stderr, "couldn't open %s\n", path); exit(EXIT_FAILURE); }

    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // A negative vocab_size is the flag for an *untied* classifier matrix.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);

    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fclose(file);

    // mmap the whole file so the OS pages weights in lazily; no big malloc+read.
    *fd = open(path, O_RDONLY);
    if (*fd == -1) { fprintf(stderr, "open failed\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(EXIT_FAILURE); }

    float* weights_ptr = *data + sizeof(Config) / sizeof(float); // skip header
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer* t, char* checkpoint_path) {
    read_checkpoint(checkpoint_path, &t->config, &t->weights,
                    &t->fd, &t->data, &t->file_size);
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    if (t->data != MAP_FAILED && t->data != NULL) munmap(t->data, t->file_size);
    if (t->fd != -1) close(t->fd);
    free_run_state(&t->state);
}

/* ----------------------------------------------------------------------------
 * 2. THE MATH PRIMITIVES
 * -------------------------------------------------------------------------- */

/* RMSNorm: normalize a vector by its root-mean-square, then scale per-element.
 * Llama uses this instead of LayerNorm — no mean subtraction, no bias.
 *   y_i = x_i / sqrt(mean(x^2) + eps) * weight_i
 * It keeps the residual stream at a stable scale before each sub-layer. */
void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss = ss / size + 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

/* Numerically-stable softmax, in place. Subtracting the max prevents expf
 * from overflowing. Turns raw scores into a probability distribution. */
void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

/* Matrix-vector product: out = W @ x, where W is (d, n) row-major and x is (n,).
 * This is the single hottest operation in the whole network — every projection
 * and the final classifier is a matmul. Each output row is a dot product. */
void matmul(float* out, float* x, float* w, int n, int d) {
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        float* wi = w + i * n;
        for (int j = 0; j < n; j++) val += wi[j] * x[j];
        out[i] = val;
    }
}

/* ----------------------------------------------------------------------------
 * 3. THE FORWARD PASS — one token in, a vector of logits out
 * -------------------------------------------------------------------------- */
float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float* x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // query heads sharing one kv head
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // --- embed: copy the token's learned vector into the residual stream ---
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(float));

    // --- run every transformer block ---
    for (unsigned long long l = 0; l < (unsigned)p->n_layers; l++) {

        // == attention sub-layer ==
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // Project the normalized vector to query, key, value. Keys and values
        // are written straight into this position's slot in the KV cache.
        int loff = l * p->seq_len * kv_dim; // layer offset into the cache
        float* k = s->key_cache   + loff + pos * kv_dim;
        float* v = s->value_cache + loff + pos * kv_dim;
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(k,    s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(v,    s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        // RoPE: rotary positional embeddings. Instead of adding a position
        // vector, we *rotate* each (even, odd) pair of dims by an angle that
        // grows with position and shrinks with dim index. The dot product in
        // attention then depends only on the *relative* offset between tokens.
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val), fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // rotate q (and k if within kv_dim)
            for (int vv = 0; vv < rotn; vv++) {
                float* vec = vv == 0 ? s->q : k;
                float v0 = vec[i], v1 = vec[i + 1];
                vec[i]     = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // Multi-head self-attention. Each head attends over all positions
        // 0..pos, weighting past values by softmax(query·key / sqrt(d)).
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            // scores against every cached key (this head's kv group)
            for (int t = 0; t <= pos; t++) {
                float* kt = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += q[i] * kt[i];
                att[t] = score / sqrtf(head_size);
            }
            softmax(att, pos + 1); // causal: only positions 0..pos exist
            // weighted sum of values -> this head's output, into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* vt = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) xb[i] += a * vt[i];
            }
        }

        // output projection, then residual add back into the stream
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb2[i];

        // == feed-forward sub-layer (SwiGLU) ==
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);
        // gate = w1·x, up = w3·x ; hidden = silu(gate) * up ; out = w2·hidden
        matmul(s->hb,  s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val))); // SiLU / swish: x * sigmoid(x)
            val *= s->hb2[i];                     // gated by the "up" projection
            s->hb[i] = val;
        }
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb[i]; // residual
    }

    // --- final norm + classifier head -> logits over the vocabulary ---
    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul(s->logits, x, w->wcls, dim, p->vocab_size);
    return s->logits;
}

/* ----------------------------------------------------------------------------
 * 4. THE TOKENIZER — text <-> token ids (byte-level BPE, Llama/sentencepiece)
 * -------------------------------------------------------------------------- */

typedef struct { char* str; int id; } TokenIndex;

typedef struct {
    char** vocab;        // token id -> string
    float* vocab_scores; // merge priority; higher score = merge earlier
    TokenIndex* sorted_vocab; // alphabetical, for binary-search lookup
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // single-byte fallback tokens "<0xNN>"
} Tokenizer;

int compare_tokens(const void* a, const void* b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = malloc(vocab_size * sizeof(char*));
    t->vocab_scores = malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // built lazily on first encode
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

/* Map a token id back to its printable string. Llama emits a leading space on
 * the first real token and encodes raw bytes as "<0xNN>" — handle both. */
char* decode(Tokenizer* t, int prev_token, int token) {
    char* piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++; // strip BOS-leading space
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)
        piece = (char*)t->byte_pieces + byte_val * 2;
    return piece;
}

void safe_printf(char* piece) {
    if (piece == NULL || piece[0] == '\0') return;
    if (piece[1] == '\0') { // single byte — skip unprintable control chars
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

/* Encode a string into token ids using the BPE merge algorithm:
 *   1. start from individual UTF-8 characters (byte fallback if unknown)
 *   2. repeatedly merge the adjacent pair with the best (highest) merge score
 *   3. stop when no adjacent pair exists in the vocabulary
 * BOS (1) / EOS (2) sentinels are optional. Llama also prepends a dummy space. */
void encode(Tokenizer* t, char* text, int8_t bos, int8_t eos, int* tokens, int* n_tokens) {
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) { // lazily build the search index
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // buffer big enough to merge any two adjacent tokens
    char* str_buffer = malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    // Llama tokenizer convention: a leading space is prepended to the input.
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // First pass: greedily map each UTF-8 codepoint to a token, else fall back
    // to raw bytes (token id = byte value + 3, the reserved byte-token offset).
    for (char* c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0; // start of a new codepoint
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue; // more bytes coming
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++)
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
        }
        str_len = 0;
    }

    // Second pass: keep merging the best adjacent pair until none qualify.
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
        if (best_idx == -1) break; // nothing left to merge
        tokens[best_idx] = best_id;          // replace the pair with the merge
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

/* ----------------------------------------------------------------------------
 * 5. THE SAMPLER — turn logits into the next token id
 * -------------------------------------------------------------------------- */

typedef struct { float prob; int index; } ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // scratch buffer for top-p
    float temperature;    // 0 = greedy/argmax; higher = more random
    float topp;           // nucleus sampling threshold (1 = off)
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0; float max_p = probabilities[0];
    for (int i = 1; i < n; i++)
        if (probabilities[i] > max_p) { max_i = i; max_p = probabilities[i]; }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample an index from the distribution; coin is a uniform draw in [0,1)
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) return i;
    }
    return n - 1; // rounding fallback
}

int compare_probindex(const void* a, const void* b) {
    float pa = ((ProbIndex*)a)->prob, pb = ((ProbIndex*)b)->prob;
    return (pa < pb) ? 1 : (pa > pb) ? -1 : 0; // descending
}

/* Top-p (nucleus) sampling: keep the smallest set of tokens whose cumulative
 * probability exceeds p, renormalize, and sample from just those. Cuts the
 * long tail of unlikely tokens without a hard top-k cutoff. */
int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1); // cheap pre-filter
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
    float r = coin * cumulative_prob; // sample within the nucleus
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

// xorshift RNG -> a float in [0, 1); deterministic given the seed.
unsigned int random_u32(unsigned long long* state) {
    *state ^= *state >> 12; *state ^= *state << 25; *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long* state) { return (random_u32(state) >> 8) / 16777216.0f; }

int sample(Sampler* s, float* logits) {
    if (s->temperature == 0.0f) return sample_argmax(logits, s->vocab_size);
    // temperature scaling: lower temp sharpens the distribution
    for (int q = 0; q < s->vocab_size; q++) logits[q] /= s->temperature;
    softmax(logits, s->vocab_size);
    float coin = random_f32(&s->rng_state);
    if (s->topp <= 0 || s->topp >= 1) return sample_mult(logits, s->vocab_size, coin);
    return sample_topp(logits, s->vocab_size, s->topp, s->probindex, coin);
}

/* ----------------------------------------------------------------------------
 * 6. THE GENERATION LOOP — prompt in, stream tokens out
 * -------------------------------------------------------------------------- */

long time_in_ms(void) {
    struct timespec time; clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void generate(Transformer* t, Tokenizer* tok, Sampler* s, char* prompt, int steps) {
    char* empty = "";
    if (prompt == NULL) prompt = empty;

    // tokenize the prompt (with BOS, no EOS)
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
            // still feeding the prompt: force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // past the prompt: actually sample the model's continuation
            next = sample(s, logits);
        }
        pos++;

        if (next == 1) break; // BOS marks end-of-sequence for these models

        char* piece = decode(tok, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) start = time_in_ms(); // start timing after the first step
    }
    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "\nachieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }
    free(prompt_tokens);
}

/* ----------------------------------------------------------------------------
 * 7. CLI
 * (compiled out when run.c is used as a library, e.g. the WASM web demo)
 * -------------------------------------------------------------------------- */
#ifndef PROMETHEUS_LIB

void error_usage(void) {
    fprintf(stderr, "Usage: run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -z tokenizer.bin -t 0.9 -i \"Once upon a time\"\n");
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
        steps = transformer.config.seq_len; // clamp to trained context

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
