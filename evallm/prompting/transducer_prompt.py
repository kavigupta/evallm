from abc import abstractmethod

import numpy as np

from evallm.sample_dfa.transduce import transduce

from .prompter import Prompter


class TransducerPrompter(Prompter):

    def __init__(self, num_symbols):
        self.num_symbols = num_symbols

    def prompt_and_answer(self, dfa, rng):
        inp = rng.choice(sorted(dfa.input_symbols), size=self.num_symbols)
        out = transduce(dfa, inp)

        return (inp, out), self.display_prompt(inp, out), out[-1]

    def trivial(self, metas, answers):
        return len(set(answers)) == 1

    @abstractmethod
    def display_prompt(self, inp, out):
        pass


class CleanTransducerPrompter(TransducerPrompter):
    def display_prompt(self, inp, out):
        return "".join(
            [
                tok
                for x, y in zip(inp, out)
                for tok in (f"in: {x}, ", f"out: {int(y)}\n")
            ][:-1]
        )

    def prompt_kwargs(self):
        return {
            "max_tokens": 4,
            "logprobs": 2,
        }

    def score_completion(self, output, choice):
        logprobs = choice.logprobs.top_logprobs[-1]
        probs = {k: np.exp(v) for k, v in logprobs.items() if k in ("0", "1")}
        denom = sum(probs.values())
        probs = {k: v / denom for k, v in probs.items()}
        confusion = np.zeros((2, 2))
        for j in range(2):
            confusion[int(output), j] = probs.get(str(j), 0)
        return confusion


class BasicInstructionTransducerPrompter(CleanTransducerPrompter):

    def display_prompt(self, inp, out):
        pattern = super().display_prompt(inp, out)
        return (
            "Find the final output, which alternates between"
            f" inputs and outputs. The pattern is created via a DFA:\n{pattern}"
        )
