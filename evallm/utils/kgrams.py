def longest_terminal_repeated_kgram(s):
    """
    Get the longest repeated k-gram that satisfies the property s.endswith(kgram)
    """
    k = 1
    indices = [i + 1 for i in range(len(s) - 1) if s[i] == s[-1]]
    while True:
        k += 1
        new_indices = [i for i in indices if i - k >= 0 and s[i - k] == s[-k]]
        if len(new_indices) < 1:
            k -= 1
            return k, s[-k:], [i - k for i in indices]
        indices = new_indices


def predict_based_on_kgram(inp, out):
    prefix = [x for i, o in zip(inp, out) for x in [i, o]][:-1]
    k, _, idxs = longest_terminal_repeated_kgram(prefix)
    preds = [prefix[i + k] for i in idxs]
    num_ones = sum(1 for x in preds if x == 1)
    if num_ones * 2 > len(preds):
        return 1
    if num_ones * 2 < len(preds):
        return 0
    # break ties by whatever the most recent match was
    return preds[-1]
