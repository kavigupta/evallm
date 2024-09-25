import evallm
from evallm.prompting.transducer_prompt import (
    ChainOfThoughtPromptRealExampleNoExplanation,
)

evallm.experiments.transducer_experiment.chatgpt_transducer_experiments(
    "gpt-4o-mini-2024-07-18",
    lambda num_sequence_symbols, sample_dfa_spec: ChainOfThoughtPromptRealExampleNoExplanation(
        num_sequence_symbols, sample_dfa_spec, 1
    ),
)
