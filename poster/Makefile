# Randomly Sampled Language Reasoning Problems -- conference poster (LuaLaTeX).
#
# Shares the talk's beamer theme (../presentation/beamerthemeevallm.sty, found
# via TEXINPUTS in latexmkrc) and the same result figures, regenerated here in
# a static (overlay-free) form by the talk's generators.
.PHONY: all pdf figures watch clean distclean

PYTHON ?= ../env/bin/python
GENDIR := ../presentation

# Static versions of the talk's scatter and prompt-comparison table: same data
# and layout code, but with the beamer overlays collapsed for a single page.
SCATTER := generated/scatter.tex
TABLE := generated/prompt_table.tex
FIGURES := $(SCATTER) $(TABLE)

all: pdf

$(SCATTER): $(GENDIR)/generate_scatter.py
	$(PYTHON) $(GENDIR)/generate_scatter.py --static --out $(abspath $(SCATTER))

$(TABLE): $(GENDIR)/generate_table.py
	$(PYTHON) $(GENDIR)/generate_table.py --static --out $(abspath $(TABLE))

figures: $(FIGURES)

# Compile the poster (after ensuring the static figures exist).
pdf: $(FIGURES)
	latexmk

# Recompile on every change.
watch: $(FIGURES)
	latexmk -pvc

# Remove LaTeX build artifacts (keeps generated/ figures).
clean:
	latexmk -C

# Also remove the generated figures.
distclean: clean
	rm -rf generated
