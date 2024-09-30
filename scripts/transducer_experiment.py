import evallm
from evallm.prompting.transducer_prompt import BasicSequencePrompt

evallm.experiments.transducer_experiment.chatgpt_transducer_experiments(
    "gpt-4o-mini-2024-07-18",
    lambda num_sequence_symbols, sample_dfa_spec: BasicSequencePrompt(
        num_sequence_symbols, version=3
    ),
)
