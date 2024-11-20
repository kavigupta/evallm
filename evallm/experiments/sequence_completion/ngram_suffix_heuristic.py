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


def suffix_after(sequences, prefix):
    for sequence in sequences:
        if prefix in sequence[:-1]:
            idx = sequence[:-1].rindex(prefix)
            return sequence[idx + len(prefix) :]


def ngram_heuristic_with_prefix(sequences, prefix):
    for trim in range(len(prefix)):
        s = suffix_after(
            ["".join(sequence[trim:]) for sequence in sequences], "".join(prefix[trim:])
        )
        if s is not None:
            return list(s)
    return ngram_heuristic(sequences, prefix)
