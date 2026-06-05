# Randomly Sampled Language Reasoning Problems — talk

A self-contained beamer presentation for the paper *Randomly Sampled Language
Reasoning Problems Elucidate Limitations of In-Context Learning* (Gupta,
Sanders, Solar-Lezama). It does **not** share any files with the codebase or
the paper build; everything (theme, figures) lives in this directory.

## Template

Follows the same minimal black-on-white theme as the E-Stitch workshop deck.

- **Text font:** Futura Medium (`Futura Md BT`; Heavy is the bold cut)
- **Code font:** Noto Mono
- **Colors:** black text on a near-white background
- **Highlight:** `#D55E00` (the colorblind-safe orange used in the paper
  figures; blue `#0072B2` and purple `#6300CC` carry the `a`/`b`/`c` alphabet
  in the DFA diagrams)

The theme lives in `beamerthemeevallm.sty`. Highlight text with `\hl{...}` (or
beamer's `\alert{...}`).

## Build

Requires **LuaLaTeX** (for `fontspec` / system fonts) — `xelatex` also works if
you adjust `latexmkrc`. Both fonts must be installed on the system.

```sh
make          # builds presentation.pdf
make watch    # rebuild on save
make clean
```

Or directly: `latexmk` (the bundled `latexmkrc` selects LuaLaTeX).
