import numpy as np
from evallm.lib import (
    sequence_completion_dfa_prompts_answers,
    sequence_completion_evaluate_responses,
    transducer_evaluate_responses,
    transducer_metas_prompts_answers,
)


def transducer_example(seed):
    metas, prompts, answers = transducer_metas_prompts_answers(
        seed=seed,
        model_is_chat=True,
        num_repeats_per_dfa=30,
        prompt_key="Basic",
    )

    print("prompts look like", prompts[0])

    responses = ["0"] * len(prompts)

    result = transducer_evaluate_responses(
        responses=responses,
        answers=answers,
        model_is_chat=False,
        prompt_key="Basic",
    )
    return result


def sequence_completion_example(seed):

    dfa, prompts, data = sequence_completion_dfa_prompts_answers(
        model_is_chat=True, seed=seed, prompt_key="Basic"
    )

    print("prompts look like", prompts[0])

    responses = ["aaaaa"] * len(prompts)

    return sequence_completion_evaluate_responses(
        dfa=dfa,
        responses=responses,
        sequences_prefixes=data,
        prompt_key="Basic",
    )


if __name__ == "__main__":
    seed = 42
    print("*" * 80)
    print("Transducer example")
    print("*" * 80)
    print()
    res = transducer_example(seed)
    print()
    print("Transducer example result:", res)
    print()
    print()
    print("*" * 80)
    print("Sequence completion example")
    print("*" * 80)
    print()
    res = sequence_completion_example(seed)
    print()
    print("Sequence completion example result:", res)
