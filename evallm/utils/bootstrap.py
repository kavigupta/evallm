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


def permutation_test(values_1, values_2, *, n_samples=1_000):
    """
    Perform a permutation test to compare the means of two sets of values.

    :param values_1: ndarray of shape (n_values_1, *vector_dims)
    :param values_2: ndarray of shape (n_values_2, *vector_dims)
    :param n_samples: number of permutations to test

    :return: p-value
    """
    n_values_1, n_values_2 = len(values_1), len(values_2)
    values = np.concatenate([values_1, values_2], axis=0)
    n_values = n_values_1 + n_values_2
    observed_diff = values_1.mean(axis=0) - values_2.mean(axis=0)
    n_values_1 = len(values_1)
    rng = np.random.RandomState(0)
    indices = np.array(
        [rng.choice(n_values, n_values, replace=False) for _ in range(n_samples)]
    ).T
    values_permuted = values[indices]
    means_1, means_2 = values_permuted[:n_values_1].mean(0), values_permuted[
        n_values_1:
    ].mean(0)
    return (observed_diff <= means_1 - means_2).mean()
