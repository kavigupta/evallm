# Randomly Sampled Language Reasoning Problems -- talk (self-contained, LuaLaTeX).
.PHONY: all pdf prompts watch clean distclean

# Python with the evallm package available (the repo virtualenv by default).
PYTHON ?= ../env/bin/python

all: pdf

# Regenerate the verbatim prompt listings from the real prompting code whenever
# the generator changes. The script writes all generated/*.tex files at once;
# we track one of them as the make target.
GEN_SENTINEL := generated/prompt_basic.tex

$(GEN_SENTINEL): generate_prompts.py
	$(PYTHON) generate_prompts.py

prompts: $(GEN_SENTINEL)

# Compile the slides (after ensuring the generated listings exist).
pdf: $(GEN_SENTINEL)
	latexmk

# Recompile on every change.
watch: $(GEN_SENTINEL)
	latexmk -pvc

# Remove LaTeX build artifacts (keeps generated/ listings).
clean:
	latexmk -C

# Also remove the generated prompt listings.
distclean: clean
	rm -rf generated
