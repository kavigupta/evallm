from collections import defaultdict

from evallm.experiments.models_display import model_by_display_key
from evallm.experiments.transducer_experiment import (
    current_dfa_sample_spec,
    run_brute_force_transducer,
    run_transducer_experiment,
    run_transducer_experiment_just_stats,
)
from evallm.llm.llm import model_specs
from evallm.prompting.transducer_prompt import (
    BasicInstructionTransducerPrompter,
    BasicSequencePrompt,
    BasicSequencePromptNoChat,
    BasicSequencePromptSlightlyMoreExplanation,
    RedGreenRoomPrompt1,
    SequencePromptWithExplanationChainOfThought,
)

num_states = 3
num_symbols = 3
num_sequence_symbols = 30
num_repeats_per_dfa = 30
sample_dfa_spec = current_dfa_sample_spec(num_states=num_states)
setting_kwargs = dict(
    num_sequence_symbols=num_sequence_symbols,
    sample_dfa_spec=sample_dfa_spec,
    num_states=num_states,
)

prompt_by_key = {
    "Basic": {
        "non-chat": BasicSequencePromptNoChat.for_setting(setting_kwargs),
        "chat": BasicSequencePrompt.for_setting(setting_kwargs),
    },
    "More-Expl": {
        "chat": BasicSequencePromptSlightlyMoreExplanation.for_setting(setting_kwargs)
    },
    "COT": {
        "chat": SequencePromptWithExplanationChainOfThought.for_setting(setting_kwargs)
    },
    "Red-Green": {"chat": RedGreenRoomPrompt1.for_setting(setting_kwargs)},
}


def for_model_and_prompt(model, num_dfas, *prompts):
    model_key = model_by_display_key[model]
    if model_specs[model_key].is_chat:
        prompt_kind = "chat"
    else:
        prompt_kind = "non-chat"
    return {
        (model, prompt): run_transducer_experiment(
            model_key,
            sample_dfa_spec,
            prompt_by_key[prompt][prompt_kind],
            num_repeats_per_dfa=num_repeats_per_dfa,
            num_dfas=num_dfas,
        )
        for prompt in prompts
    }


def compute_results():
    deterministic_baseline_outcomes = run_transducer_experiment_just_stats(
        "none",
        sample_dfa_spec,
        BasicInstructionTransducerPrompter(num_sequence_symbols, strip=True),
        num_repeats_per_dfa=num_repeats_per_dfa,
        num_dfas=1000,
    )
    model_outcomes = {
        **for_model_and_prompt("llama3-8B", 1000, "Basic"),
        **for_model_and_prompt("llama3-70B", 1000, "Basic"),
        **for_model_and_prompt("llama3.1-8B-Instruct", 1000, "Basic"),
        **for_model_and_prompt("starcoder2-15b", 100, "Basic"),
        **for_model_and_prompt("codestral-22B", 1000, "Basic"),
        **for_model_and_prompt("deepseek-coder-33b-instruct", 1000, "Basic"),
        **for_model_and_prompt("qwen-2.5-coder-7B", 1000, "Basic"),
        **for_model_and_prompt("qwen-2.5-coder-instruct-7B", 1000, "Basic"),
        **for_model_and_prompt("qwen-2.5-coder-instruct-32B", 1000, "Basic"),
        **for_model_and_prompt("mistral-nemo-minitron-8B", 1000, "Basic"),
        **for_model_and_prompt("mistral-nemo-base-12B", 1000, "Basic"),
        **for_model_and_prompt("mistral-nemo-instruct-12B", 1000, "Basic"),
        **for_model_and_prompt("gemma-7b", 1000, "Basic"),
        **for_model_and_prompt("falcon-7b", 1000, "Basic"),
        **for_model_and_prompt("gpt-3.5-instruct", 100, "Basic"),
        **for_model_and_prompt("gpt-3.5-chat", 100, "Basic"),
        **for_model_and_prompt(
            "gpt-4o-mini",
            100,
            "Basic",
            "More-Expl",
            "COT",
            "Red-Green",
        ),
        **for_model_and_prompt("gpt-4o", 30, "Basic"),
        **for_model_and_prompt(
            "claude-3.5",
            30,
            "Basic",
            "More-Expl",
            "COT",
            "Red-Green",
        ),
    }

    no_prompt = "Basic"

    accuracies = defaultdict(dict)
    accuracies[r"\textsc{Null}$_T$"][no_prompt] = [
        r.null_success_rate for r in deterministic_baseline_outcomes
    ]
    for ngram in range(2, 2 + 5):
        accuracies[rf"{ngram}-\textsc{{Gram}}$_T$"][no_prompt] = [
            r.kgram_success_rates_each[ngram - 2]
            for r in deterministic_baseline_outcomes
        ]
    accuracies[r"\textsc{BruteForce}$_T$"][no_prompt] = run_brute_force_transducer(
        sample_dfa_spec,
        num_states,
        num_symbols,
        num_sequence_symbols,
        num_repeats_per_dfa,
        num_dfas=1000,
    )
    for model, prompt in model_outcomes:
        accuracies[model][prompt] = [
            r.success_rate_binary_ignore_na for r in model_outcomes[model, prompt]
        ]
    return accuracies