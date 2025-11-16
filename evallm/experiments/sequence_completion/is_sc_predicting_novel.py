import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import tqdm.auto as tqdm
from permacache import permacache

from evallm.experiments.sequence_completion.sequence_completion_experiments import (
    get_examples,
    run_model,
)
from evallm.experiments.sequence_completion.sequence_completion_prompt import (
    extract_sequence_from_response,
)

from ...cachedir import cache_dir
from ..example import blue, green, orange, red
from ..models_display import model_by_display_key
from ..sequence_completion_summary import current_setting, prompts_by_key


def is_a_suffix(candidate, positive_examples):
    for example in positive_examples:
        if example.endswith(candidate):
            return True
    return False


def is_a_substring(candidate, positive_examples):
    for example in positive_examples:
        if candidate in example:
            return True
    return False


def classify_response(response, positive_examples, prefix):
    positive_examples = ["".join(example) for example in positive_examples]
    prefix = "".join(prefix)
    if is_a_suffix(prefix[-1] + response, positive_examples):
        return "ngram-suffix"
    if is_a_suffix(response, positive_examples):
        return "suffix"
    if is_a_substring(response, positive_examples):
        return "substring"
    return "novel"


ordered_classifications = ["ngram-suffix", "suffix", "substring", "novel"]


def novelty_experiment(num_seeds, model_display_key, prompt_key="Basic"):
    prompt = prompts_by_key[prompt_key](current_setting)
    model_key = model_by_display_key[model_display_key]
    is_correct, classifications = predict_novelty_for_examples(
        num_seeds, current_setting, model_key, prompt
    )
    return [x for xs in is_correct for x in xs], [
        x for xs in classifications for x in xs
    ]


@permacache(
    os.path.join(cache_dir, "predict_novelty_for_examples_sequence_completion"),
    key_function=dict(prompt=lambda prompt: prompt.hash_prompt()),
    shelf_type="individual-file",
)
def predict_novelty_for_examples(num_seeds, setting, model_key, prompt):
    all_is_correct = []
    all_classifications = []
    for seed in tqdm.trange(num_seeds):
        is_correct, classifications = predict_novelty_for_example(
            seed, setting, model_key, prompt
        )
        all_is_correct.append(is_correct)
        all_classifications.append(classifications)
    return all_is_correct, all_classifications


def predict_novelty_for_example(seed, setting, model_key, prompt):
    dfa, io = get_examples(seed, setting)
    responses = run_model(model_key, prompt, dfa, io)

    is_correct = [
        prompt.score_response(dfa, *io_eg, response)
        for io_eg, response in zip(io, responses)
    ]
    completions = [
        extract_sequence_from_response(dfa, response) for response in responses
    ]
    classifications = [
        classify_response(completion, *io_eg)
        for completion, io_eg in zip(completions, io)
    ]
    return is_correct, classifications


def _draw_stacked_bar(
    ax, xpos_left, side_width, counts_side, classes, colors, annotate=True
):
    """Draw a single top-down stacked bar at xpos_left with given visual width.

    counts_side: mapping class -> count (may contain zeros)
    classes, colors: ordered lists of same length
    Returns total count for the side.
    """
    total = sum(counts_side.values())
    heights = [counts_side.get(c, 0) / total for c in classes]
    top = 1.0
    x_center = xpos_left + side_width / 2.0
    for h, col in zip(heights, colors):
        if h > 0:
            seg_bottom = top - h
            ax.bar(
                xpos_left,
                h,
                width=side_width,
                bottom=seg_bottom,
                color=col,
                linewidth=0.6,
                align="edge",
            )
            if annotate and h >= 0.03:
                ax.text(
                    x_center,
                    seg_bottom + h / 2,
                    f"{h:.1%}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
            top = seg_bottom
    return total


def plot_novelty_results(is_correct, classifications, ax, control_classifs=None):

    counts_correct = Counter(
        classification
        for corr, classification in zip(is_correct, classifications)
        if corr
    )
    counts_incorrect = Counter(
        classification
        for corr, classification in zip(is_correct, classifications)
        if not corr
    )

    total_correct = sum(counts_correct.values())
    total_incorrect = sum(counts_incorrect.values())

    # small gap proportional to the larger width
    gap = max(0.5, max(total_correct, total_incorrect) * 0.05)
    # left-edge positions so bars sit side-by-side
    x_correct_left = 0.0

    colors = [blue, green, orange, red]

    # build a simple sequence of sides (label, counts, width) and accumulate x positions
    sides = [
        ("Correct", counts_correct, total_correct),
        ("Incorrect", counts_incorrect, total_incorrect),
    ]

    if control_classifs is not None:
        counts_control = Counter(control_classifs)
        # control column visual width is a fixed fraction of the two main columns
        width_control = 0.25 * (total_correct + total_incorrect)
        if width_control <= 0:
            width_control = 1.0
        sides.append(("Control", counts_control, width_control))

    # iterate sides left-to-right, accumulating x position
    x = x_correct_left
    xticks = []
    xticklabels = []
    total_main = total_correct + total_incorrect
    for label, counts_side, side_width in sides:
        xpos_left = x
        # draw column
        _draw_stacked_bar(
            ax,
            xpos_left,
            side_width,
            counts_side,
            ordered_classifications,
            colors,
            annotate=True,
        )
        # xtick label: for the two main columns show percent (of main total), for control show 'Random'
        x_center = xpos_left + side_width / 2.0
        if label in ("Correct", "Incorrect"):
            pct = sum(counts_side.values()) / total_main
            xticks.append(x_center)
            xticklabels.append(f"{label} ({pct:.1%})".replace("%", r"\%"))
        else:
            xticks.append(x_center)
            xticklabels.append("Random")

        # advance x by column width + gap
        x += side_width + gap

    # set axis limits and ticks
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(x_correct_left - gap, x - gap)
    ax.set_yticks([])
    ax.set_title("Novelty classifications by correctness")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend with class totals
    legend_entries = []
    for c in ordered_classifications:
        total_c = counts_correct.get(c, 0) + counts_incorrect.get(c, 0)
        if control_classifs is not None:
            total_c += counts_control.get(c, 0)
        legend_entries.append(f"{c} ({total_c})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in colors]
    ax.legend(
        handles,
        legend_entries,
        title="Classes",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    return ax


@permacache(
    "evallm/experiments/sequence_completion/is_sc_predicting_novelty/control_classifications",
)
def control_classifications(num_examples):
    classifications = []
    rng = np.random.default_rng(0)
    for seed in tqdm.trange(num_examples):
        dfa, io = get_examples(seed, current_setting)
        for io_eg in io:
            completion = "".join(
                rng.choice(
                    sorted(dfa.input_symbols), size=len(io_eg[0][0]) - len(io_eg[1])
                ).tolist()
            )
            classifications.append(classify_response(completion, *io_eg))

    return classifications
