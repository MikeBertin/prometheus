# PROMETHEUS — llama2-style transformer inference in pure C
#
#   make            -> optimized build (use this to actually run models)
#   make debug      -> -O0 -g with sanitizers, for debugging the math
#   make demo       -> build + run the stories15M model with a prompt
#   make bard       -> build + run OUR OWN trained model (Phase 2)
#   make bardq      -> the same model, int8-quantized via runq.c (Phase 4)
#   make train      -> train + export the Shakespeare model (needs .venv)
#   make stories    -> run OUR TinyStories model (Phase 5: BPE, ~7M params)
#   make storiesq   -> the TinyStories model, int8-quantized
#   make clean

CC      = cc
CFLAGS  = -O3 -ffast-math -march=native -funroll-loops -Wall -Wextra
LDFLAGS = -lm
SRC     = src/run.c
BIN     = run
QSRC    = src/runq.c
QBIN    = runq

MODEL   = models/stories15M.bin
TOKEN   = models/tokenizer.bin

.PHONY: all debug demo bard bardq stories storiesq train web clean

all: $(BIN) $(QBIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(BIN) $(LDFLAGS)

# int8-quantized engine (Q8_0 checkpoints from export.py --q80)
$(QBIN): $(QSRC)
	$(CC) $(CFLAGS) $(QSRC) -o $(QBIN) $(LDFLAGS)

# Debug build: no fast-math, full warnings, address+UB sanitizers.
debug: $(SRC)
	$(CC) -O0 -g -Wall -Wextra -fsanitize=address,undefined $(SRC) -o $(BIN) $(LDFLAGS)

demo: $(BIN)
	./$(BIN) $(MODEL) -z $(TOKEN) -t 0.9 -i "Once upon a time"

# Phase 2: our own weights, trained by src/train.py, byte-level tokenizer.
bard: $(BIN)
	./$(BIN) models/shakespeare.bin -z models/byte_tokenizer.bin -t 0.8 -i "ROMEO:"

# Phase 4: the same bard, int8-quantized (4x smaller checkpoint).
bardq: $(QBIN)
	./$(QBIN) models/shakespeare_q80.bin -z models/byte_tokenizer.bin -t 0.8 -i "ROMEO:"

# Phase 5: ~7M-param model on TinyStories with a real 4096-vocab BPE tokenizer.
stories: $(BIN)
	./$(BIN) models/tinystories.bin -z models/tinystories_tokenizer.bin -t 0.85 -i "Once upon a time"

storiesq: $(QBIN)
	./$(QBIN) models/tinystories_q80.bin -z models/tinystories_tokenizer.bin -t 0.85 -i "Once upon a time"

# Phase 6: the instruction-tuned model. Prompt uses the chat template.
chat: $(BIN)
	./$(BIN) models/tinystories_instruct.bin -z models/tinystories_tokenizer.bin -t 0.7 \
	  -i "$$(printf 'User: Write a story using the words: dragon, cake, brave.\nAssistant:')"

train:
	.venv/bin/python src/tokenizer_export.py models/byte_tokenizer.bin
	.venv/bin/python src/train.py --iters 750 --out models/shakespeare.bin --ckpt models/ckpt.pt

# WASM build for the web demo: the int8 engine (runq.c) + quantized weights.
# (Drop -DPROMETHEUS_Q and swap the .bin to serve the fp32 engine instead.)
web: src/web_api.c src/run.c src/runq.c
	emcc -O3 -msimd128 -ffast-math -DPROMETHEUS_Q src/web_api.c -o web/prometheus.js \
	  -sMODULARIZE=1 -sEXPORT_NAME=Prometheus \
	  -sEXPORTED_RUNTIME_METHODS=cwrap,FS \
	  -sALLOW_MEMORY_GROWTH=1 -sENVIRONMENT=web \
	  --no-entry
	mkdir -p web/models
	cp models/shakespeare_q80.bin models/byte_tokenizer.bin web/models/
	cp models/tinystories_q80.bin models/tinystories_tokenizer.bin web/models/
	cp models/tinystories_instruct_q80.bin models/tinystories_aligned_q80.bin \
	   models/tinystories_ppo_q80.bin models/tinystories_rloo_q80.bin web/models/

clean:
	rm -f $(BIN) $(QBIN)
	rm -rf *.dSYM
