import evallm
from evallm.prompting.transducer_prompt import BasicSequencePrompt

evallm.experiments.transducer_experiment.chatgpt_transducer_experiments(
    "gpt-4o-2024-05-13",
    cot_prompt=lambda num_sequence_symbols, sample_dfa_spec: BasicSequencePrompt(
        num_sequence_symbols, version=3
    ),
    num_states_options=(3,),
)
