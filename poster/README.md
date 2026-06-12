# Randomly Sampled Language Reasoning Problems — poster

A **4 ft wide × 5 ft tall** (121.92 cm × 152.4 cm) conference poster for the
paper *Randomly Sampled Language Reasoning Problems Elucidate Limitations of
In-Context Learning* (Gupta, Sanders, Solar-Lezama). Companion to the talk in
`../presentation`.

## Template

Reuses the talk's minimal black-on-cream beamer theme
(`../presentation/beamerthemeevallm.sty`, built with
[`beamerposter`](https://ctan.org/pkg/beamerposter)):

- **Text font:** Futura Medium (`Futura Md BT`; Heavy is the bold cut)
- **Code font:** Noto Mono
- **Highlight:** `#D55E00` (the colorblind-safe orange used in the paper
  figures; blue `#0072B2` / purple `#6300CC` carry the `a`/`b`/`c` alphabet)

Layout is a straightforward **two columns** under a full-width title banner,
closing on a full-width **Takeaway** footer.

## Figures

The scatter and the prompt-comparison table are regenerated from the same
cached experiment results as the talk, by the talk's generators run in a
**static** (overlay-free) mode:

```sh
../env/bin/python ../presentation/generate_scatter.py --static --out generated/scatter.tex
../env/bin/python ../presentation/generate_table.py   --static --out generated/prompt_table.tex
```

`--static` collapses the beamer overlays into a single page and, for the
scatter, labels only the baselines (the LLM cloud stays a color-coded scatter —
the individual models are named in the table). `make figures` runs both.

## Build

Requires **LuaLaTeX** (for `fontspec` / system fonts); both Futura and Noto
Mono must be installed. The bundled `latexmkrc` selects LuaLaTeX and adds
`../presentation` to `TEXINPUTS` so the shared theme is found.

```sh
make          # regenerates the static figures, then builds poster.pdf
make watch    # rebuild on save
make clean    # remove LaTeX build artifacts (keeps generated/)
make distclean
```

To resize, edit the `width`/`height` in the `beamerposter` options at the top
of `poster.tex` (and re-tune `scale` and `\colht`).
