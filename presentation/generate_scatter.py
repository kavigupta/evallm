#!/usr/bin/env python
r"""Generate the Transducer-vs-Sequence-Completion scatter (TikZ) for the talk.

Reuses the same cached experiment results and grouping logic that build the
paper's tables / matplotlib scatter (evallm.experiments.main_tables), so the
points always match the reported numbers. Run from the repo root with the venv:

    env/bin/python presentation/generate_scatter.py

Writes presentation/generated/scatter.tex (a standalone tikzpicture, included
in the deck via \input). Labels are placed by a small greedy de-collision pass
since the real points cluster tightly.
"""
import math
import os

import numpy as np

from evallm.experiments.transducer_summary import transducer_results
from evallm.experiments.sequence_completion_summary import sequence_completion_results
from evallm.experiments.main_tables import (
    best_prompt_among_basics,
    replace_null_random,
    models_to_category,
    grouped_models,
    random_null,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "generated", "scatter.tex")

# Category -> deck color (defined in the preamble).
CATEGORY_COLOR = {
    "Baselines": "ecblue",
    "Open Weight Completion": "ecpurple",
    "Open Weight Code": "ecorange",
    "Proprietary": "evallmProp",
}

# Beamer overlay spec for each category's labels (the talk reveals them in turn:
# baselines stay, then the region, then open-weight, then code (replacing it),
# then proprietary).
CATEGORY_OVERLAY = {
    "Baselines": "2-",
    "Open Weight Completion": "4",
    "Open Weight Code": "5",
    "Proprietary": "6-",
}
REGION_OVERLAY = "3-"

# Hand-tuned label offset directions (dirx, diry, rel_dist) copied verbatim from
# the paper's matplotlib scatter (main_tables.plot_transducer_vs_sequence_-
# completion), so the static poster reproduces that figure's labels and layout.
# Offsets are in data units; rel_dist scales the offset distance.
PAPER_DIRECTIONS = {
    r"\textsc{BruteForce}": (-1, -1, 0.1),
    r"6-\textsc{Gram}": (1, -1, 0.1),
    r"5-\textsc{Gram}": (-1, 1, 0.1),
    r"4-\textsc{Gram}": (1, -1, 0.1),
    r"3-\textsc{Gram}": (1, 1, 0.1),
    r"2-\textsc{Gram}": (1, 1, 0.1),
    "mistral-nemo-minitron-8B": (-0.0001, 1, 1),
    "qwen-2.5-coder-7B": (0.0001, -1, 1),
    "qwen-2.5-coder-instruct-7B": (0.0001, 1, 1),
    "gpt-3.5-instruct": (-1, 1, 0.1),
    "gpt-4o-mini": (1, -1, 0.1),
    "gpt-4o": (1, -1, 0.1),
    "gemma-7b": (-1, -1, 0.1),
    "falcon-7b": (-1, -1, 0.1),
    "starcoder2-15b": (-1, 1, 1),
    "deepseek-coder-33b-instruct": (1, -0.5, 1),
    "claude-3.5": (1, -0.5, 1),
    random_null: (1, 1, 0.1),
    "gpt-5": (1, -1, 1),
}
PAPER_OFF_DIST = 2.5  # data-unit offset scale (matplotlib off_dist)

# canvas size (cm) and approximate \tiny glyph metrics (cm)
# Keep the x/y data scale equal so equal value differences render at the same
# physical distance on both axes (for example, 80-90 on x and y match).
W, H = 12.0, 7.2
CHAR_W, LINE_H = 0.105, 0.20

# 8 placement directions as integer signs (sx, sy)
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
GAP = 0.13  # minimum clear distance between a label's near edge and its point


def label_for(model):
    if model == random_null:
        return "Random/Null"
    return model.replace(r"\textsc{", "").replace("}", "")


def anchor_str(sx, sy):
    va = {1: "south", -1: "north", 0: ""}[sy]
    ha = {1: "west", -1: "east", 0: ""}[sx]
    return (va + " " + ha).strip() or "center"


def box_for(ax, ay, w, h, sx, sy):
    """Bounding box of a node anchored per (sx, sy) at (ax, ay)."""
    x0, x1 = (ax, ax + w) if sx > 0 else (ax - w, ax) if sx < 0 else (ax - w / 2, ax + w / 2)
    y0, y1 = (ay, ay + h) if sy > 0 else (ay - h, ay) if sy < 0 else (ay - h / 2, ay + h / 2)
    return (x0, y0, x1, y1)


def overlap(a, b):
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0.0


def place_labels(labels, pts, allow, obstacles=(), extras=(0.0, 0.12, 0.28, 0.5)):
    """Greedy anchored placement: a label is offset from its point by at least GAP
    (so it always lies outside the marker), in the direction minimizing overlap
    with other dots, already-placed labels, and the canvas margins.

    ``obstacles`` are fixed boxes (legend, annotations) that labels must also
    dodge; ``extras`` is the ladder of extra offsets a label may take to escape a
    crowded region (a leader line is drawn when it ends up far from its point)."""
    ax0, ay0, ax1, ay1 = allow
    dot_xy = list(pts.values())

    def crowding(m):
        px, py = pts[m]
        return sum(1 for qx, qy in dot_xy if (qx - px) ** 2 + (qy - py) ** 2 < 1.0)

    placed = list(obstacles)
    out = {}
    for m in sorted(labels, key=crowding, reverse=True):
        px, py = pts[m]
        w = CHAR_W * len(label_for(m)) + 0.06
        h = LINE_H
        best, best_score = None, 1e18
        for extra in extras:
            for sx, sy in DIRS:
                ax = px + sx * (GAP + extra)
                ay = py + sy * (GAP + extra)
                box = box_for(ax, ay, w, h, sx, sy)
                s = 0.0
                if box[0] < ax0:
                    s += 25 + (ax0 - box[0]) * 40
                if box[2] > ax1:
                    s += 25 + (box[2] - ax1) * 40
                if box[1] < ay0:
                    s += 25 + (ay0 - box[1]) * 40
                if box[3] > ay1:
                    s += 25 + (box[3] - ay1) * 40
                for qx, qy in dot_xy:
                    if box[0] <= qx <= box[2] and box[1] <= qy <= box[3]:
                        s += 6
                for b in placed:
                    a = overlap(box, b)
                    if a > 0:
                        s += 10 + a * 60
                s += extra * 1.4
                if sx != 0 and sy != 0:
                    s += 0.15  # mild preference for cardinal placement
                if s < best_score:
                    best_score, best = s, (ax, ay, sx, sy, box)
        out[m] = best[:4]
        placed.append(best[4])
    return out


def place_labels_paper(labels, x, y, cx, cy, obstacles=()):
    """Reproduce the paper figure's label placement: each label is pushed from
    its point along a hand-tuned direction (PAPER_DIRECTIONS), the offset taken
    in data units and mapped to cm. Where the slight aspect-ratio difference
    from the paper's matplotlib figure makes two labels collide, the later one
    is slid further out along its own direction until clear, so the paper's
    layout intent is preserved. Returns the same (ax, ay, sx, sy) tuples as
    place_labels so the emission/bounding-box code is shared."""
    # Place the isolated baseline frontier first, then the denser LLM cloud, so
    # the cloud labels yield to the (fixed-looking) frontier ones.
    order = sorted(labels, key=lambda m: x[m] + y[m], reverse=True)
    placed = list(obstacles)
    out = {}
    for m in order:
        dirx, diry, rel = PAPER_DIRECTIONS.get(m, (1, 1, 1))
        base = PAPER_OFF_DIST / math.hypot(dirx, diry)
        w, h = CHAR_W * len(label_for(m)) + 0.06, LINE_H
        sx, sy = (1 if dirx >= 0 else -1), (1 if diry >= 0 else -1)
        chosen = None
        for mult in (rel, rel + 0.4, rel + 0.9, rel + 1.6, rel + 2.5):
            r = mult * base
            ax, ay = cx(x[m] + dirx * r), cy(y[m] + diry * r)
            box = box_for(ax, ay, w, h, sx, sy)
            if not any(overlap(box, b) > 0 for b in placed):
                chosen = (ax, ay, box)
                break
        if chosen is None:
            chosen = (ax, ay, box)
        out[m] = (chosen[0], chosen[1], sx, sy)
        placed.append(chosen[2])
    return out


def main(out=OUT, static=False):
    rt = transducer_results()
    rsc = sequence_completion_results()
    summary_t = replace_null_random(best_prompt_among_basics(rt))
    summary_sc = replace_null_random(best_prompt_among_basics(rsc))

    ordered = [
        m
        for m in models_to_category
        if m in summary_t
        and m in summary_sc
        and not isinstance(summary_sc[m], float)
        and not isinstance(summary_t[m], float)
    ]
    x = {m: 100 * float(np.mean(summary_sc[m])) for m in ordered}
    y = {m: 100 * float(np.mean(summary_t[m])) for m in ordered}

    # Label all baselines + proprietary, plus the *best* model in each category
    # on each metric. The talk reveals these in turn, so the worst open-source
    # ones would just add clutter; the static poster matches the paper figure
    # instead, which also labels each category's *worst* per metric.
    labelled = {m for m in ordered if models_to_category[m] in ("Baselines", "Proprietary")}
    for members in grouped_models.values():
        ms = [m for m in members if m in ordered]
        for metric in (y, x):
            vals = [metric[m] for m in ms]
            labelled.add(ms[int(np.argmax(vals))])
            if static:
                labelled.add(ms[int(np.argmin(vals))])
    to_label = [m for m in ordered if m in labelled]

    non_baseline = [m for m in ordered if models_to_category[m] != "Baselines"]
    x_model_max = max(x[m] for m in non_baseline)
    y_model_max = max(y[m] for m in non_baseline)

    x_lo = 5 * math.floor(min(x.values()) / 5)
    y_lo = 5 * math.floor(min(y.values()) / 5)
    x_hi = y_hi = 100

    def cx(v):
        return (v - x_lo) / (x_hi - x_lo) * W

    def cy(v):
        return (v - y_lo) / (y_hi - y_lo) * H

    pts = {m: (cx(x[m]), cy(y[m])) for m in ordered}
    if static:
        # Reproduce the paper figure's hand-tuned label layout, keeping labels
        # clear of the top-left legend and the top-right "better than all" note.
        legend_box = (cx(x_lo) + 0.30, cy(y_hi) - 0.15 - 3 * 0.34 - 0.13,
                      cx(x_lo) + 0.30 + 0.13 + 0.135 * len("open-weight code") + 0.1,
                      cy(y_hi) - 0.15 + 0.13)
        anno_box = (cx(x_model_max) + 0.05, cy(y_hi) - 0.50,
                    cx(x_model_max) + 1.7, cy(y_hi) + 0.02)
        label_pos = place_labels_paper(to_label, x, y, cx, cy,
                                       obstacles=(legend_box, anno_box))
    else:
        # labels may spill into the right/top margins, but not far off-canvas
        label_pos = place_labels(to_label, pts, allow=(-0.7, -0.45, W + 1.7, H + 0.55))
        # Tuck Random/Null into the bottom-left corner (just below its marker).
        if random_null in pts:
            rpx, rpy = pts[random_null]
            label_pos[random_null] = (rpx, rpy - GAP, 0, -1)

    # Fixed bounding box (covers every label) so the picture never resizes as
    # labels appear/disappear between overlays -- otherwise it jitters.
    bxs, bys = [-1.2, W + 0.3, W / 2], [-0.85, H + 0.25, H / 2]
    for m in to_label:
        ax, ay, sx, sy = label_pos[m]
        bw, bh = CHAR_W * len(label_for(m)) + 0.06, LINE_H
        qx0, qy0, qx1, qy1 = box_for(ax, ay, bw, bh, sx, sy)
        bxs += [qx0, qx1]
        bys += [qy0, qy1]
    # The static poster scales the picture to a tight column width with no slide
    # margin to spill into, so pad the right edge: the \tiny glyph-width estimate
    # undershoots, and the far-right "BruteForce" label would otherwise clip.
    pad_r = 0.6 if static else 0.05
    bb = (min(bxs) - 0.05, min(bys) - 0.05, max(bxs) + pad_r, max(bys) + 0.05)

    L = [r"% Generated by generate_scatter.py -- do not edit by hand.",
         r"\begin{tikzpicture}[font=\scriptsize]",
         rf"  \useasboundingbox ({bb[0]:.2f},{bb[1]:.2f}) rectangle ({bb[2]:.2f},{bb[3]:.2f});"]
    # Wrap a block of TikZ lines in a beamer overlay spec, except in --static
    # mode (the poster), where everything is shown at once.
    def overlay(spec, lines):
        if static:
            return list(lines)
        return [rf"  \only<{spec}>{{"] + lines + [r"  }"]

    # "better than all LLMs on both" region + annotation (top-left of the region)
    L += overlay(REGION_OVERLAY, [
        rf"    \fill[ecblue!14] ({cx(x_model_max):.3f},{cy(y_model_max):.3f}) "
        rf"rectangle ({cx(x_hi):.3f},{cy(y_hi):.3f});",
        rf"    \node[ecblue, anchor=north west, align=left, font=\tiny] "
        rf"at ({cx(x_model_max)+0.08:.3f},{cy(y_hi)-0.06:.3f}) "
        r"{better than all\\LLMs on both};",
    ])
    # axes (short arrows; labels run alongside each axis, centered)
    L.append(rf"  \draw[->, black!60] (-0.2,0) -- ({W + 0.25:.2f},0);")
    L.append(rf"  \draw[->, black!60] (0,-0.2) -- (0,{H + 0.2:.2f});")
    L.append(
        rf"  \node[font=\scriptsize, text=black!80, anchor=north] "
        rf"at ({W / 2:.2f},-0.55) {{Seq.\ Completion}};"
    )
    L.append(
        rf"  \node[font=\scriptsize, text=black!80, rotate=90] "
        rf"at (-0.95,{H / 2:.2f}) {{Transducer}};"
    )
    for v in range(int(x_lo), x_hi + 1, 10):
        L.append(
            rf"  \draw[black!50] ({cx(v):.3f},0) -- ({cx(v):.3f},-0.08) node[below]{{{v}}};"
        )
    for v in range(int(math.ceil(y_lo / 10.0) * 10), y_hi + 1, 10):
        L.append(
            rf"  \draw[black!50] (0,{cy(v):.3f}) -- (-0.08,{cy(v):.3f}) node[left]{{{v}}};"
        )
    # legend, placed inside the (empty) top-left of the plot
    legend = [
        ("Baselines", "baselines"),
        ("Open Weight Completion", "open-weight"),
        ("Open Weight Code", "open-weight code"),
        ("Proprietary", "proprietary"),
    ]
    lx0, ly0 = cx(x_lo) + 0.35, cy(y_hi) - 0.15
    for i, (cat, name) in enumerate(legend):
        yy = ly0 - i * 0.34
        L.append(rf"  \fill[{CATEGORY_COLOR[cat]}] ({lx0:.3f},{yy:.3f}) circle (1.8pt);")
        L.append(
            rf"  \node[anchor=west, text=black, font=\scriptsize] "
            rf"at ({lx0 + 0.13:.3f},{yy:.3f}) {{{name}}};"
        )
    # points
    for m in ordered:
        color = CATEGORY_COLOR[models_to_category[m]]
        px, py = pts[m]
        L.append(rf"  \fill[{color}] ({px:.3f},{py:.3f}) circle (1.8pt);")
    # labels: anchored a clear GAP outside the marker, with a faint leader line
    # to the nearest box edge only when the label is pushed far from its point.
    # Each category's labels are wrapped in its beamer overlay spec.
    for m in to_label:
        cat = models_to_category[m]
        color = CATEGORY_COLOR[cat]
        px, py = pts[m]
        ax, ay, sx, sy = label_pos[m]
        w, h = CHAR_W * len(label_for(m)) + 0.06, LINE_H
        x0, y0, x1, y1 = box_for(ax, ay, w, h, sx, sy)
        nx, ny = min(max(px, x0), x1), min(max(py, y0), y1)
        lab = []
        if math.hypot(px - nx, py - ny) > 0.32:
            lab.append(
                rf"    \draw[{color}!50, line width=0.3pt] "
                rf"({px:.3f},{py:.3f}) -- ({nx:.3f},{ny:.3f});"
            )
        lab.append(
            rf"    \node[{color}, anchor={anchor_str(sx, sy)}, inner sep=0.5pt, font=\tiny] "
            rf"at ({ax:.3f},{ay:.3f}) {{{label_for(m)}}};"
        )
        L += overlay(CATEGORY_OVERLAY[cat], lab)
    L.append(r"\end{tikzpicture}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(
        f"wrote {out}: {len(ordered)} points, {len(to_label)} labelled; "
        f"x_lo={x_lo} y_lo={y_lo} frontier=({x_model_max:.1f},{y_model_max:.1f})"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--static", action="store_true",
                    help="emit a single overlay-free picture (for the poster)")
    ap.add_argument("--out", default=OUT, help="output .tex path")
    args = ap.parse_args()
    main(out=args.out, static=args.static)
