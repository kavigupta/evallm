#!/usr/bin/env python
r"""Generate the transducer prompt-example listings used in the talk.

Rather than transcribing the prompts by hand (and risking drift from the real
code), this regenerates them straight from the prompting classes used in the
experiments. Run from the repo root with the project venv:

    env/bin/python presentation/generate_prompts.py

It writes, into presentation/generated/:
  * prompt_<key>.tex  -- the verbatim user prompt the model receives
                         (included in the deck via \lstinputlisting)
  * out_<key>.tex     -- the (correct) model output for that example
"""
import os

import numpy as np

from evallm.experiments.transducer_summary import prompt_by_key_and_settings
from evallm.experiments.transducer_experiment import current_dfa_sample_spec
from evallm.sample_dfa.sample_dfa import sample_dfa

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "generated")
NUM_SYMBOLS = 6  # short enough to fit on a slide


def pick_example():
    """First seed whose transducer trace mixes 0s and 1s (so it isn't trivial)."""
    spec = current_dfa_sample_spec(num_states=3)
    basic = prompt_by_key_and_settings(
        num_sequence_symbols=NUM_SYMBOLS, num_states=3
    )["Basic"]["chat"]
    for seed in range(10000):
        rng = np.random.RandomState(seed)
        dfa = sample_dfa(spec, rng)
        inp, out = basic.input_output(dfa, rng)
        out = [int(x) for x in out]
        if set(out) == {0, 1}:
            return seed, inp, out
    raise RuntimeError("no suitable example found")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seed, inp, out = pick_example()
    prompts = prompt_by_key_and_settings(
        num_sequence_symbols=NUM_SYMBOLS, num_states=3
    )

    answer = out[-1]
    color = {0: "red", 1: "green"}[answer]
    model_out = {
        "Basic": str(answer),
        "Basic-COT": f"(reasoning...) <answer>{answer}</answer>",
        "More-Expl": str(answer),
        "DFA-COT": f"(reasoning...) <answer>{answer}</answer>",
        "Red-Green": f"<answer>{color}</answer>",
    }

    for key, by_mode in prompts.items():
        user = by_mode["chat"].display_prompt(inp, out, is_chat=True)["user"]
        slug = key.lower()
        with open(os.path.join(OUT_DIR, f"prompt_{slug}.tex"), "w") as f:
            f.write(user.rstrip("\n") + "\n")
        with open(os.path.join(OUT_DIR, f"out_{slug}.tex"), "w") as f:
            f.write(model_out[key])

    print(
        f"seed={seed}  inp={list(map(str, inp))}  out={out}  "
        f"answer={answer} ({color})"
    )
    print(f"wrote {2 * len(prompts)} files to {OUT_DIR}")


if __name__ == "__main__":
    main()
