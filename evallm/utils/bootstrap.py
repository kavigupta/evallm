import numpy as np


def boostrap_mean(values, *, n_samples=1_000, pct=95):
    """
    Compute the boostrap confidence interval of the mean of a list of ndarrays.
        Does so independently for each element of the vector.

    :param values: ndarray of shape (n_values, *vector_dims)
    :param n_samples: number of bootstrap samples to draw
    :param pct: confidence interval percentage

    :return: ndarray of shape (2, *vector_dims) containing the lower and
    upper bounds of the confidence interval
    """
    lo_pct, hi_pct = (100 - pct) / 2, 100 - (100 - pct) / 2
    n_values = len(values)
    indices = np.random.RandomState(0).choice(n_values, (n_values, n_samples))
    samples = values[indices]
    means = samples.mean(axis=0)
    return np.percentile(means, [lo_pct, hi_pct], axis=0)
