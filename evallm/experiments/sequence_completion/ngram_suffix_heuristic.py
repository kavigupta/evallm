from collections import Counter


def ngram_heuristic(sequences, prefix):
    """
    Compute a completion for the given prefix using an n-gram heuristic.

    For this heuristic, the prefix is ignored and we instead find the suffix
    of the sequences S = argmax_S (# of sequences that have S as a suffix) * len(S).

    S must have a length <= len(sequence[0]) - len(prefix)
    """
    suffix_counts = Counter()
    for sequence in sequences:
        for i in range(1, len(sequence) - len(prefix)):
            suffix_counts[tuple(sequence[-i:])] += 1

    return list(max(suffix_counts, key=lambda x: suffix_counts[x] * len(x)))
