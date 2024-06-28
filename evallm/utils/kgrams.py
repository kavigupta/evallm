def longest_terminal_repeated_kgrams(s):
    """
    Get the longest repeated k-gram that satisfies the property s.endswith(kgram)
    """
    k = 1
    indices = [i + 1 for i in range(len(s) - 1) if s[i] == s[-1]]
    results = []
    while True:
        results.append((k, s[-k:], [i - k for i in indices]))
        k += 1
        new_indices = [i for i in indices if i - k >= 0 and s[i - k] == s[-k]]
        if len(new_indices) < 1:
            return results
        indices = new_indices


def predict_from_sequence_based_on_kgram(seq):
    results = []
    for k, _, idxs in longest_terminal_repeated_kgrams(seq):
        preds = [seq[i + k] for i in idxs]
        num_ones = sum(1 for x in preds if x == 1)
        if num_ones * 2 > len(preds):
            results.append(1)
        elif num_ones * 2 < len(preds):
            results.append(0)
        else:
            # break ties by whatever the most recent match was
            results.append(preds[-1])
    return results


def predict_based_on_kgram(inp, out):
    prefix = [x for i, o in zip(inp, out) for x in [i, o]][:-1]

    return predict_from_sequence_based_on_kgram(prefix)
