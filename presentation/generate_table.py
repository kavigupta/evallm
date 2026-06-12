#!/usr/bin/env python
r"""Generate the prompt-comparison table (a trimmed Table 2) for the talk.

Built straight from the same cached results that produce the paper's tables, so
the numbers can never drift. Run from the repo root with the venv:

    env/bin/python presentation/generate_table.py

Differences from the paper's Table 2, for legibility on a slide:
  * the two pure-ICL columns (Basic / Basic-CoT) are collapsed into one Basic
    column reporting the better of the two (the deck reports each model's best
    pure-ICL shot anyway);
  * a 6-Gram column is added as the non-world-modeling baseline to beat;
  * the two tasks sit side by side (sharing one Model column) instead of
    stacked, so the table is short and wide rather than tall.

Shadings carry the slide's point, each timed to its bullet:
  * green on the whole More-Expl column (merely knowing a grammar exists does not
    help at all), overlay beat 2;
  * red on the structure columns (DFA-CoT / Red-Green) of the non-reasoning
    models (first three rows), in both task blocks (overlay beat 3);
  * orange on o3-mini -- on the Transducer where structure helps (beat 4) and on
    Sequence Completion where it does not (beat 5);
  * blue on gpt-5 (last row) -- on Sequence Completion where exact structure does
    not benefit it (beat 6) and on the Transducer where it then solves the task
    essentially perfectly (beat 7).

Writes presentation/generated/prompt_table.tex (a bare tabular, \input in deck).
"""
import os

import numpy as np

from evallm.experiments.transducer_summary import transducer_results
from evallm.experiments.sequence_completion_summary import sequence_completion_results
from evallm.experiments.main_tables import multi_prompts, stretch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "generated", "prompt_table.tex")

# Prompt columns shown (after collapsing Basic/Basic-COT into "Basic").
PROMPTS = [r"\textsc{Basic}", r"\textsc{More-Expl}", r"\textsc{DFA-COT}", r"\textsc{Red-Green}"]
BASICS = [r"\textsc{Basic}", r"\textsc{Basic-COT}"]
GRAM = r"\textsc{$6$-Gram}"
# Short column headers (the full \textsc names are too wide stacked twice over).
HEAD = {
    r"\textsc{Basic}": r"\textsc{Basic}",
    r"\textsc{More-Expl}": r"\textsc{More-Expl}",
    r"\textsc{DFA-COT}": r"\textsc{DFA-CoT}",
    r"\textsc{Red-Green}": r"\textsc{Red-Green}",
    GRAM: GRAM,
}
COLUMNS = PROMPTS + [GRAM]  # the five value columns per task
# Per-cell shading, each timed to its bullet (overlay beat in \only<...>).
MORE_EXPL = r"\textsc{More-Expl}"
STRUCTURE = (r"\textsc{DFA-COT}", r"\textsc{Red-Green}")
NONREASONING = ("gpt-4o-mini", "gpt-4o", "claude-3.5")
# Each shading is timed to its bullet via a beamer overlay in the talk; the
# poster (--static) drops the \only wrappers and shows every shading at once.
STATIC = "--static" in __import__("sys").argv


def _shade(spec, color):
    cell = rf"\cellcolor{{{color}}}"
    return cell if STATIC else rf"\only<{spec}>{{{cell}}}"


GREEN = _shade("2-", "ecgreen!22")            # knowing a grammar exists
RED = _shade("3-", "evallmProp!18")           # structure does not help
ORANGE_T = _shade("4-", "evallmHighlight!22")  # o3-mini: helped here
ORANGE_S = _shade("5-", "evallmHighlight!22")  # o3-mini: not here
BLUE_S = _shade("6-", "ecblue!22")            # gpt-5: no benefit here
BLUE_T = _shade("7-", "ecblue!22")            # gpt-5: solves it here
SHADES = {}
for _g in ("Sequence Completion", "Transducer"):
    for _m in NONREASONING:
        for _c in STRUCTURE:
            SHADES[(_g, _m, _c)] = RED
for _c in STRUCTURE:
    SHADES[("Transducer", "o3-mini", _c)] = ORANGE_T
    SHADES[("Sequence Completion", "o3-mini", _c)] = ORANGE_S
    SHADES[("Sequence Completion", "gpt-5", _c)] = BLUE_S
    SHADES[("Transducer", "gpt-5", _c)] = BLUE_T


def cell_mean(v):
    """Mean of a result cell; -inf for an N/A (nan float) placeholder."""
    if isinstance(v, float):
        return -np.inf if np.isnan(v) else v
    return float(np.mean(v))


def fmt_acc(prefix, v):
    """Mean accuracy only (no bootstrap CI -- too busy for a slide)."""
    if isinstance(v, float) and np.isnan(v):
        return r"N/A\footnotemark[1]"
    return prefix + f"{100 * np.mean(v):.1f}"


def gram_value(raw):
    """The 6-Gram baseline result array for a task's raw results dict."""
    (key,) = [k for k in raw if k.startswith("6-")]
    (prompt,) = raw[key].keys()
    return raw[key][prompt]


def block_cells(group, name, row, gram):
    """The five value cells (Basic..6-Gram) for one model in one task block."""
    # Collapse Basic / Basic-COT into the better-scoring of the two.
    basic = max((row[b] for b in BASICS if b in row), key=cell_mean,
                default=float("nan"))
    values = {r"\textsc{Basic}": basic, GRAM: gram}
    for c in PROMPTS[1:]:
        values[c] = row.get(c, float("nan"))
    # The 6-Gram baseline competes for "best", so a model only takes the bold
    # when it actually beats the n-gram.
    best = max(COLUMNS, key=lambda c: cell_mean(values[c]))
    cells = []
    for c in COLUMNS:
        acc = fmt_acc(r"\bf " if c == best else "", values[c])
        # The whole More-Expl column is shaded green (knowing a grammar exists
        # does not help); the structure columns carry the per-model shadings.
        shade = GREEN if c == MORE_EXPL else SHADES.get((group, name, c))
        if shade:
            acc = shade + " " + acc
        cells.append(acc)
    return cells


def main(out=OUT):
    rt, rsc = transducer_results(), sequence_completion_results()
    # The two task blocks, side by side, sharing one Model column.
    blocks = [
        ("Sequence Completion", multi_prompts(rsc), gram_value(rsc)),
        ("Transducer", multi_prompts(rt), gram_value(rt)),
    ]
    models = list(blocks[0][1])

    colspec = "l|" + "|".join(["ccccc"] * len(blocks))
    L = [r"{\renewcommand{\arraystretch}{" + str(stretch) + r"}"
         + r"\begin{tabular}{" + colspec + r"}", r"\hline"]
    # group-spanning header
    span = [r" "] + [
        r"\multicolumn{5}{c" + ("|" if i < len(blocks) - 1 else "") + r"}{\bf " + g + r"}"
        for i, (g, _, _) in enumerate(blocks)
    ]
    L.append(" & ".join(span) + r" \\")
    L.append(r"\hline")
    # per-column sub-header (prompt names, repeated under each task)
    sub = [r"\bf Model"] + [r"\bf " + HEAD[c] for _ in blocks for c in COLUMNS]
    L.append(" & ".join(sub) + r" \\")
    L.append(r"\hline")

    for i, name in enumerate(models):
        cells = [name]
        for group, results, gram in blocks:
            cells += block_cells(group, name, results[name], gram)
        L.append(" & ".join(cells) + r" \\")
        if i != len(models) - 1:
            L.append(r"\hline")
    L.append(r"\hline")

    L += [r"\end{tabular}", r"}"]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--static", action="store_true",
                    help="show every shading at once (for the poster)")
    ap.add_argument("--out", default=OUT, help="output .tex path")
    args = ap.parse_args()
    main(out=args.out)
