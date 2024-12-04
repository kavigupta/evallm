from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib_venn import venn3

blue = "#009bff"
green = "#26d94a"
red = "#ff0062"

from evallm.experiments.exhaustive_transducer_experiment import (
    TransducerExperimentResultPacked,
)


def transducer_example_csv(
    results: TransducerExperimentResultPacked, samples: Dict[str, List[int]]
):
    completions = [x.message.content for x in results.completions]
    return pd.concat(
        [
            pd.DataFrame(
                [
                    dict(
                        id=f"{k}-{i}",
                        transducer_trace="".join(
                            x
                            for xy in zip(
                                np.array(list("abc"))[results.inputs_packed[idx]],
                                (str(int(x)) for x in results.outputs_packed[idx]),
                            )
                            for x in xy
                        ),
                        completion=completions[idx],
                    )
                    for i, idx in enumerate(samples[k])
                ]
            )
            for k in samples
        ]
    )


def plot_errors(transducer_masks: Dict[str, np.ndarray]):
    venn3(
        subsets=[set(np.where(x == 0)[0]) for x in transducer_masks.values()],
        set_labels=[f"{k}: {v.mean():.1%}" for k, v in transducer_masks.items()],
        set_colors=(red, green, blue),
    )
