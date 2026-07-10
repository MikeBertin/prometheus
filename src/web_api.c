/* ============================================================================
 * PROMETHEUS — web_api.c
 * A thin emscripten wrapper around run.c for the browser demo.
 *
 * The inference engine is the REAL run.c — included below, unmodified, with
 * only its CLI main() compiled out. This file adds a small stateful API the
 * page's JS drives one token at a time:
 *
 *   prom_init(model_path, tokenizer_path)  -> load checkpoint + tokenizer
 *   prom_start(prompt, temp, topp, seed)   -> encode prompt, reset position
 *   prom_next()                            -> one step: returns the decoded
 *                                             piece, or "" when finished
 *
 * JS fetches the .bin files, writes them into emscripten's in-memory FS, and
 * then run.c's mmap()/open() work exactly as they do natively.
 * ========================================================================== */

#include <emscripten.h>

/* Compile against either engine — they expose identical internals.
 * -DPROMETHEUS_Q selects the int8 runq.c (what the live site ships:
 * 2.4 MB of weights instead of 9.1, and ~2x the tok/s). */
#define PROMETHEUS_LIB
#ifdef PROMETHEUS_Q
#include "runq.c"
#else
#include "run.c"
#endif

static Transformer T;
static Tokenizer TOK;
static Sampler S;
static int loaded = 0;
static int sampler_built = 0;

/* generation state across prom_next() calls */
static int *prompt_tokens = NULL;
static int num_prompt_tokens = 0;
static int pos = 0;
static int steps = 0;
static int token = 0;
static int done = 1;

EMSCRIPTEN_KEEPALIVE
int prom_init(const char *model_path, const char *tokenizer_path) {
    if (loaded) return T.config.seq_len;
    build_transformer(&T, (char *)model_path);
    build_tokenizer(&TOK, (char *)tokenizer_path, T.config.vocab_size);
    loaded = 1;
    return T.config.seq_len;  // so the UI can show/cap the step count
}

EMSCRIPTEN_KEEPALIVE
int prom_start(const char *prompt, float temperature, float topp,
               unsigned int seed, int n_steps) {
    if (!loaded) return -1;
    if (sampler_built) free_sampler(&S);
    build_sampler(&S, T.config.vocab_size, temperature, topp,
                  (unsigned long long)seed);
    sampler_built = 1;

    free(prompt_tokens);
    prompt_tokens = malloc((strlen(prompt) + 3) * sizeof(int));
    num_prompt_tokens = 0;
    encode(&TOK, (char *)prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) { done = 1; return -1; }

    steps = (n_steps <= 0 || n_steps > T.config.seq_len) ? T.config.seq_len
                                                         : n_steps;
    pos = 0;
    token = prompt_tokens[0];
    done = 0;
    return num_prompt_tokens;
}

/* One autoregressive step. Returns the decoded piece for the token that was
 * just produced (prompt tokens are "forced", matching run.c's generate()),
 * or "" once generation has ended. The KV cache never needs resetting:
 * position p's slot is simply overwritten and attention only reads 0..pos. */
EMSCRIPTEN_KEEPALIVE
const char *prom_next(void) {
    if (done || pos >= steps) { done = 1; return ""; }

    float *logits = forward(&T, token, pos);
    int next = (pos < num_prompt_tokens - 1) ? prompt_tokens[pos + 1]
                                             : sample(&S, logits);
    pos++;

    if (next == 1) { done = 1; return ""; }  // BOS = end of sequence

    char *piece = decode(&TOK, token, next);
    token = next;
    if (piece == NULL) return "";
    if (piece[0] != '\0' && piece[1] == '\0') {
        unsigned char b = piece[0];          // same filter as safe_printf()
        if (!(isprint(b) || isspace(b))) return "";
    }
    return piece;
}

EMSCRIPTEN_KEEPALIVE
int prom_is_done(void) { return done; }
