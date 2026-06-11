# Randomly Sampled Language Reasoning Problems -- talk (self-contained, LuaLaTeX).
.PHONY: all pdf prompts scatter table generated watch clean distclean

# Python with the evallm package available (the repo virtualenv by default).
PYTHON ?= ../env/bin/python

all: pdf

# Regenerate the verbatim prompt listings and the results scatter from the real
# code whenever a generator changes. Each script writes its generated/*.tex
# file(s) at once; we track one output of each as the make target.
PROMPTS_SENTINEL := generated/prompt_basic.tex
SCATTER := generated/scatter.tex
TABLE := generated/prompt_table.tex
GENERATED := $(PROMPTS_SENTINEL) $(SCATTER) $(TABLE)

$(PROMPTS_SENTINEL): generate_prompts.py
	$(PYTHON) generate_prompts.py

$(SCATTER): generate_scatter.py
	$(PYTHON) generate_scatter.py

$(TABLE): generate_table.py
	$(PYTHON) generate_table.py

# Emoji art: a vector PDF converted from the committed Noto SVG. The PDF is
# committed too, so this only re-runs (needing inkscape) if the SVG changes.
EMOJI := images/technologist.pdf
$(EMOJI): images/technologist.svg
	inkscape $< --export-type=pdf --export-filename=$@

# Appendix sweep figures (examples / states). Both PNGs are written at once and
# committed; this only re-runs (needing the repo env) if the generator changes.
APPENDIX := images/more-examples-per-problem.png
$(APPENDIX): generate_appendix.py
	$(PYTHON) generate_appendix.py

prompts: $(PROMPTS_SENTINEL)
scatter: $(SCATTER)
table: $(TABLE)
appendix: $(APPENDIX)
generated: $(GENERATED)

# Compile the slides (after ensuring the generated files exist).
pdf: $(GENERATED) $(EMOJI) $(APPENDIX)
	latexmk

# Recompile on every change.
watch: $(GENERATED)
	latexmk -pvc

# Remove LaTeX build artifacts (keeps generated/ listings).
clean:
	latexmk -C

# Also remove the generated prompt listings.
distclean: clean
	rm -rf generated
