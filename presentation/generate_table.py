#!/usr/bin/env python
r"""Generate the prompt-comparison table (a trimmed Table 2) for the talk.

Built straight from the same cached results that produce the paper's tables, so
the numbers can never drift. Run from the repo root with the venv:

    env/bin/python presentation/generate_table.py

Differences from the paper's Table 2, for legibility on a slide:
  * the two pure-ICL columns (Basic / Basic-CoT) are collapsed into one Basic
    column reporting the better of the two (the deck reports each model's best
    pure-ICL shot anyway);
  * a 6-Gram column is appended as the non-world-modeling baseline to beat.

The structure-revealed cells where the reasoning models nearly solve the
Transducer task (o3-mini / gpt-5 under DFA-CoT and Red-Green) are shaded blue
(on overlay beat 3, with the matching bullet) to carry the slide's point.

Writes presentation/generated/prompt_table.tex (a bare tabular, \input in deck).
"""
import os

import numpy as np

from evallm.experiments.transducer_summary import transducer_results
from evallm.experiments.sequence_completion_summary import sequence_completion_results
from evallm.experiments.main_tables import multi_prompts, stretch
from evallm.experiments.transducer_plotting import display_acc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "generated", "prompt_table.tex")

# Prompt columns shown (after collapsing Basic/Basic-COT into "Basic").
COLUMNS = [r"\textsc{Basic}", r"\textsc{More-Expl}", r"\textsc{DFA-COT}", r"\textsc{Red-Green}"]
BASICS = [r"\textsc{Basic}", r"\textsc{Basic-COT}"]
GRAM = r"\textsc{$6$-Gram}"
# Cells to shade: (group, model, prompt) -- reasoning models, structure revealed.
HIGHLIGHT = {
    ("Transducer", "o3-mini", r"\textsc{DFA-COT}"),
    ("Transducer", "o3-mini", r"\textsc{Red-Green}"),
    ("Transducer", "gpt-5", r"\textsc{DFA-COT}"),
    ("Transducer", "gpt-5", r"\textsc{Red-Green}"),
}
SHADE = r"\cellcolor{ecgreen!22}"


def cell_mean(v):
    """Mean of a result cell; -inf for an N/A (nan float) placeholder."""
    if isinstance(v, float):
        return -np.inf if np.isnan(v) else v
    return float(np.mean(v))


def gram_value(raw):
    """The 6-Gram baseline result array for a task's raw results dict."""
    (key,) = [k for k in raw if k.startswith("6-")]
    (prompt,) = raw[key].keys()
    return raw[key][prompt]


def main():
    rt, rsc = transducer_results(), sequence_completion_results()
    groups = [
        ("Sequence Completion", multi_prompts(rsc), gram_value(rsc)),
        ("Transducer", multi_prompts(rt), gram_value(rt)),
    ]

    L = [r"{\renewcommand{\arraystretch}{" + str(stretch) + r"}"
         + r"\begin{tabular}{l|c|ccc|c}", r"\hline"]
    header = [r"\bf Model"] + [r"\bf " + c for c in COLUMNS] + [r"\bf " + GRAM]
    L.append(" & ".join(header) + r" \\")

    for group, results, gram in groups:
        L.append(r"\hline")
        L.append(r"\multicolumn{6}{l}{ \bf " + group + r"} \\")
        L.append(r"\hline")
        models = list(results)
        for i, name in enumerate(models):
            row = results[name]
            # Collapse Basic / Basic-COT into the better-scoring of the two.
            basic = max((row[b] for b in BASICS if b in row), key=cell_mean,
                        default=float("nan"))
            values = {r"\textsc{Basic}": basic, GRAM: gram}
            for c in COLUMNS[1:]:
                values[c] = row.get(c, float("nan"))
            # The 6-Gram baseline competes for "best" too, so a model only takes
            # the bold when it actually beats the n-gram.
            best = max(COLUMNS + [GRAM], key=lambda c: cell_mean(values[c]))
            cells = [name]
            for c in COLUMNS + [GRAM]:
                acc = display_acc({best: r"\bf "}, values[c], c)
                if (group, name, c) in HIGHLIGHT:
                    acc = SHADE + " " + acc
                cells.append(acc)
            L.append(" & ".join(cells) + r" \\")
            if i != len(models) - 1:
                L.append(r"\hline")
        L.append(r"\hline")

    L += [r"\end{tabular}", r"}"]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
