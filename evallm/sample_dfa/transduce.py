from automata.fa.dfa import DFA


def transduce(dfa: DFA, input_string: list[str]) -> list[bool]:
    """
    Transduce the given input string with the given DFA.
    """
    return [
        x in dfa.final_states
        for x in dfa.read_input_stepwise(input_string, ignore_rejection=True)
    ][1:]
