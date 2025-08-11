

import numpy as np

from evallm.experiments.transducer_summary import sample_dfa_spec, prompt_by_key_default
from evallm.experiments.transducer_experiment import run_transducer_experiment

def small_experiment(model_name, num_dfas):
    return run_transducer_experiment(
        model=model_name,
        num_repeats_per_dfa=30,
        sample_dfa_spec=sample_dfa_spec,
        prompter=prompt_by_key_default["Basic"]["chat"],
        num_dfas=num_dfas,
    )

# res_5 = small_experiment("gpt-5-2025-08-07", 2)

def compute_frac(x):
    numer = (np.array(x.success_rate_each) == 1).sum()
    denom = (np.array(x.success_rate_each) != 0.5).sum()
    return numer, denom

def render_frac(x):
    numer, denom = compute_frac(x)
    return f"{numer:.0f}/{denom:.0f}"


for i in range(10):
	res_5 = small_experiment("gpt-5-2025-08-07", i + 1)
	print([f"{x.kgram_success_rates_each[6 - 2] * 30:.0f}/30" for x in res_5])
	print([render_frac(x) for x in res_5])
