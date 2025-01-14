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


def ngram_completions(sequences, prefix, max_completion_length):
    results = []
    for sequence in sequences:
        for start in range(len(sequence) - len(prefix)):
            end = start + len(prefix)
            if len(sequence) - end > max_completion_length:
                continue
            if sequence[start:end] == prefix:
                results.append(sequence[end:])
    return results


def multiple_ngrams(sequences, prefix):
    sequences = ["".join(sequence) for sequence in sequences]
    prefix = "".join(prefix)
    max_completion_length = len(sequences[0]) - len(prefix)
    suggestions_each = []
    for prefix_length in range(1, 1 + len(prefix)):
        prefix_to_use = prefix[-prefix_length:]
        completions = ngram_completions(sequences, prefix_to_use, max_completion_length)
        if not completions:
            suggestions_each += [suggestions_each[-1]]
            continue
        counts = Counter(completions)
        best = max(counts, key=counts.get)
        suggestions_each.append(list(best))
    return suggestions_each
