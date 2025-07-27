from evallm.experiments.models_display import model_by_display_key
from evallm.experiments.sequence_completion.sequence_completion_experiments import (
    compute_brute_force_scores,
    compute_common_suffix_heuristic,
    compute_model_scores,
    compute_random_baseline_scores,
    compute_true_ngrams,
)
from evallm.experiments.sequence_completion.sequence_completion_prompt import (
    MoreExplanationPrompt2,
    MoreExplanationPromptCOT,
    RedGreenPrompt,
    SequencePromptDirectAlien2,
    SequencePromptDirectAlien2WithCommas,
)
from evallm.experiments.transducer_experiment import current_dfa_sample_spec

spec = current_dfa_sample_spec(3)

num_sequence_symbols = 10
current_setting = dict(
    dfa_spec=spec,
    num_sequences=30,
    num_sequence_symbols=num_sequence_symbols,
    num_sequence_symbols_prompt=num_sequence_symbols // 2,
    num_instances=30,
)

prompts_by_key = {
    "Basic": SequencePromptDirectAlien2.for_setting,
    "Basic-Commas": SequencePromptDirectAlien2WithCommas.for_setting,
    "More-Expl": MoreExplanationPrompt2.for_setting,
    "COT": MoreExplanationPromptCOT.for_setting,
    "Red-Green": RedGreenPrompt.for_setting,
}


# pylint: disable=dangerous-default-value
def for_model(
    model, count, *prompts, na_mode, wrapper=lambda x: x, setting=current_setting
):
    setting = setting.copy()
    return {
        (model, prompt): compute_model_scores(
            count,
            setting,
            model_by_display_key[model],
            wrapper(prompts_by_key[prompt]),
            na_mode=na_mode,
        )
        for prompt in prompts
    }


def results_for_models(na_mode="ignore"):
    results = {
        # open source completion
        **for_model("llama3-8B", 1000, "Basic", "Basic-Commas", na_mode=na_mode),
        **for_model("llama3-70B", 1000, "Basic", "Basic-Commas", na_mode=na_mode),
        **for_model(
            "llama3.1-8B-Instruct", 1000, "Basic", "Basic-Commas", na_mode=na_mode
        ),
        **for_model("llama3.1-8B", 1000, "Basic", "Basic-Commas", na_mode=na_mode),
        **for_model("llama3.1-70B", 1000, "Basic", "Basic-Commas", na_mode=na_mode),
        **for_model(
            "mistral-nemo-minitron-8B", 1000, "Basic", "Basic-Commas", na_mode=na_mode
        ),
        **for_model(
            "mistral-nemo-base-12B", 1000, "Basic", "Basic-Commas", na_mode=na_mode
        ),
        **for_model(
            "mistral-nemo-instruct-12B", 1000, "Basic", "Basic-Commas", na_mode=na_mode
        ),
        **for_model("gemma-7b", 1000, "Basic", "Basic-Commas", na_mode=na_mode),
        **for_model("falcon-7b", 1000, "Basic", "Basic-Commas", na_mode=na_mode),
        # open source code
        **for_model("starcoder2-15b", 1000, "Basic", "Basic-Commas", na_mode=na_mode),
        **for_model("codestral-22B", 1000, "Basic", "Basic-Commas", na_mode=na_mode),
        **for_model(
            "deepseek-coder-33b-instruct",
            1000,
            "Basic",
            "Basic-Commas",
            na_mode=na_mode,
        ),
        **for_model(
            "qwen-2.5-coder-instruct-7B", 1000, "Basic", "Basic-Commas", na_mode=na_mode
        ),
        **for_model("qwen-2.5-7B", 1000, "Basic", "Basic-Commas", na_mode=na_mode),
        **for_model("qwen-2.5-32B", 1000, "Basic", "Basic-Commas", na_mode=na_mode),
        **for_model(
            "qwen-2.5-coder-instruct-32B",
            1000,
            "Basic",
            "Basic-Commas",
            na_mode=na_mode,
        ),
        **for_model(
            "qwen-2.5-coder-7B", 1000, "Basic", "Basic-Commas", na_mode=na_mode
        ),
        # closed source
        **for_model("gpt-3.5-instruct", 100, "Basic", "Basic-Commas", na_mode=na_mode),
        **for_model("gpt-3.5-chat", 100, "Basic", "Basic-Commas", na_mode=na_mode),
        **for_model(
            "gpt-4o-mini",
            100,
            "Basic",
            "Basic-Commas",
            "More-Expl",
            "COT",
            "Red-Green",
            na_mode=na_mode,
        ),
        **for_model(
            "gpt-4o",
            30,
            "Basic",
            "Basic-Commas",
            "More-Expl",
            "COT",
            "Red-Green",
            na_mode=na_mode,
        ),
        **for_model(
            "claude-3.5",
            30,
            "Basic",
            "Basic-Commas",
            "More-Expl",
            "COT",
            "Red-Green",
            na_mode=na_mode,
        ),
        **for_model(
            "o3-mini",
            30,
            "Basic",
            "Basic-Commas",
            "More-Expl",
            "COT",
            "Red-Green",
            na_mode=na_mode,
        ),
    }
    return results


# pylint: disable=dangerous-default-value
def results_for_baseline(
    setting=current_setting,
    include_non_ngram=True,
    min_ngram=2,
    max_ngram=6,
    amount_baselines=1000,
):
    setting = setting.copy()
    results = {}

    if include_non_ngram:

        results[r"\textsc{Random}$_S$", "Basic"] = compute_random_baseline_scores(
            amount_baselines, setting=setting
        )

        results[r"\textsc{Common-Suffix}$_S$", "Basic"] = (
            compute_common_suffix_heuristic(amount_baselines, setting=setting)
        )

        results[r"\textsc{BruteForce}$_S$", "Basic"] = compute_brute_force_scores(
            100, setting
        )

    for ngram in range(min_ngram, max_ngram + 1):
        results[rf"{ngram}-\textsc{{Gram}}$_S$", "Basic"] = compute_true_ngrams(
            ngram, amount_baselines, setting
        )

    return results


def display_prompt(p):
    return rf"\textsc{{{p}}}$_S$"


def sequence_completion_results():
    results = {**results_for_models(), **results_for_baseline()}
    results_nested = {m: {} for m, _ in results}
    for m, p in results:
        results_nested[m][display_prompt(p)] = results[m, p]
    return results_nested


def sequence_completion_null_results():
    result = {}
    for (model, prompt), v in results_for_models("count-na").items():
        result[model] = result.get(model, {})
        result[model][display_prompt(prompt)] = v
    return result
