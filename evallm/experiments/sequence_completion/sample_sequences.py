from evallm.sample_dfa.sample_dfa import sample_dfa


def sample_sequence_satisfying(
    dfa, rng, *, num_sequence_symbols, try_limit, separate_filter=lambda x: True
):
    """
    Sample a sequence that the DFA accepts, using rejection sampling.
    """
    sorted_syms = sorted(dfa.input_symbols)

    for _ in range(try_limit):
        sequence = rng.choice(len(sorted_syms), size=num_sequence_symbols)
        sequence_symbols = [sorted_syms[sym] for sym in sequence]
        if dfa.accepts_input(sequence_symbols) and separate_filter(sequence_symbols):
            return sequence_symbols
    raise UnsamplableDFAError


def sequence_completion_task(
    dfa,
    rng,
    *,
    num_sequences,
    num_sequence_symbols,
    num_sequence_symbols_prompt,
    try_limit
):
    """
    Sample sequences that the DFA accepts, using rejection sampling, and then one
    sequence to complete. The sequence to complete is also guaranteed to have a
    valid continuation. The sequence to complete cannot be a prefix of any of the
    sequences sampled.

    We try at least try_limit times to sample each sequence, if any fail
    we raise an error and move on to another DFA.
    """
    sequences = [
        sample_sequence_satisfying(
            dfa,
            rng,
            num_sequence_symbols=num_sequence_symbols,
            try_limit=try_limit,
        )
        for _ in range(num_sequences)
    ]
    while True:
        sequence_to_complete = sample_sequence_satisfying(
            dfa,
            rng,
            num_sequence_symbols=num_sequence_symbols,
            try_limit=try_limit,
            separate_filter=lambda x: not any(
                x == sequence[:num_sequence_symbols_prompt] for sequence in sequences
            ),
        )
        sequence_to_complete = sequence_to_complete[:num_sequence_symbols_prompt]
        return sequences, sequence_to_complete


def sample_sequence_completion_problem(
    rng,
    *,
    dfa_spec,
    num_sequences,
    num_sequence_symbols,
    num_sequence_symbols_prompt,
    try_limit,
    num_instances
):
    """
    We sample a DFA, verify that we can sample at least one sequence completion task
    from it, then sample num_instances sequence completion tasks from it.
    """
    while True:
        dfa = sample_dfa(dfa_spec, rng)
        if len(dfa.final_states) == 0 or len(dfa.final_states) == len(
            dfa.input_symbols
        ):
            continue

        try:
            sequence_completion_task(
                dfa,
                rng,
                num_sequences=num_sequences,
                num_sequence_symbols=num_sequence_symbols,
                num_sequence_symbols_prompt=num_sequence_symbols_prompt,
                try_limit=try_limit,
            )
        except UnsamplableDFAError:
            continue
        break

    result = sample_task_instances_given_dfa(
        rng,
        num_sequences,
        num_sequence_symbols,
        num_sequence_symbols_prompt,
        try_limit,
        num_instances,
        dfa,
    )
    return dfa, result


def sample_task_instances_given_dfa(
    rng,
    num_sequences,
    num_sequence_symbols,
    num_sequence_symbols_prompt,
    try_limit,
    num_instances,
    dfa,
):
    result = []
    while len(result) < num_instances:
        try:
            result.append(
                sequence_completion_task(
                    dfa,
                    rng,
                    num_sequences=num_sequences,
                    num_sequence_symbols=num_sequence_symbols,
                    num_sequence_symbols_prompt=num_sequence_symbols_prompt,
                    try_limit=try_limit,
                )
            )
        except UnsamplableDFAError:
            continue
    return result


class UnsamplableDFAError(Exception):
    pass
