# Randomly Sampled Language Reasoning Problems -- talk (self-contained, LuaLaTeX).
.PHONY: all pdf watch clean

all: pdf

# Compile the slides.
pdf:
	latexmk

# Recompile on every change.
watch:
	latexmk -pvc

clean:
	latexmk -C
