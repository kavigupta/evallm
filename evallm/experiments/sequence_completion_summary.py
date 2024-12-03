from evallm.experiments.models_display import model_by_display_key
from evallm.experiments.sequence_completion.sequence_completion_experiments import (
    compute_brute_force_scores,
    compute_model_scores,
    compute_ngram_scores,
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


def for_model(model, count, *prompts):
    return {
        (model, prompt): compute_model_scores(
            count,
            current_setting,
            model_by_display_key[model],
            prompts_by_key[prompt],
            na_mode="ignore",
        )
        for prompt in prompts
    }


def results_for_models():
    results = {
        # open source completion
        **for_model("llama3-8B", 1000, "Basic", "Basic-Commas"),
        **for_model("llama3-70B", 1000, "Basic", "Basic-Commas"),
        **for_model("llama3.1-8B-Instruct", 1000, "Basic", "Basic-Commas"),
        **for_model("mistral-nemo-minitron-8B", 1000, "Basic", "Basic-Commas"),
        **for_model("mistral-nemo-base-12B", 1000, "Basic", "Basic-Commas"),
        **for_model("mistral-nemo-instruct-12B", 1000, "Basic", "Basic-Commas"),
        **for_model("gemma-7b", 1000, "Basic", "Basic-Commas"),
        **for_model("falcon-7b", 1000, "Basic", "Basic-Commas"),
        # open source code
        **for_model("starcoder2-15b", 1000, "Basic", "Basic-Commas"),
        **for_model("codestral-22B", 1000, "Basic", "Basic-Commas"),
        **for_model("deepseek-coder-33b-instruct", 1000, "Basic", "Basic-Commas"),
        **for_model("qwen-2.5-coder-instruct-7B", 1000, "Basic", "Basic-Commas"),
        **for_model("qwen-2.5-coder-instruct-32B", 1000, "Basic", "Basic-Commas"),
        **for_model("qwen-2.5-coder-7B", 1000, "Basic", "Basic-Commas"),
        # closed source
        **for_model("gpt-3.5-instruct", 100, "Basic", "Basic-Commas"),
        **for_model("gpt-3.5-chat", 100, "Basic", "Basic-Commas"),
        **for_model(
            "gpt-4o-mini", 100, "Basic", "Basic-Commas", "More-Expl", "COT", "Red-Green"
        ),
        **for_model(
            "gpt-4o", 30, "Basic", "Basic-Commas", "More-Expl", "COT", "Red-Green"
        ),
        **for_model(
            "claude-3.5", 30, "Basic", "Basic-Commas", "More-Expl", "COT", "Red-Green"
        ),
    }
    return results


def results_for_baseline():
    amount_baselines = 1000

    results = {}

    results[r"\textsc{Random}$_S$", "Basic"] = compute_random_baseline_scores(
        amount_baselines, setting=current_setting
    )

    results[r"\textsc{Common-Suffix}$_S$", "Basic"] = compute_ngram_scores(
        amount_baselines, setting=current_setting
    )

    results[r"$\textsc{BruteForce}_S$", "Basic"] = compute_brute_force_scores(
        100, current_setting
    )

    for ngram in (2, 3, 4, 5, 6):
        results[rf"{ngram}-$\textsc{{Gram}}_S$", "Basic"] = compute_true_ngrams(
            ngram, amount_baselines, current_setting
        )

    return results
