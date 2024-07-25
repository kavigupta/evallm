from abc import abstractmethod

import numpy as np

from evallm.sample_dfa.transduce import transduce

from .prompter import Prompter


class TransducerPrompter(Prompter):

    def __init__(self, num_symbols):
        self.num_symbols = num_symbols

    def prompt_and_answer(self, dfa, rng, is_chat):
        inp = rng.choice(sorted(dfa.input_symbols), size=self.num_symbols)
        out = transduce(dfa, inp)

        return (inp, out), self.display_prompt(inp, out, is_chat), out[-1]

    def trivial(self, metas, answers):
        return len(set(answers)) == 1

    @abstractmethod
    def display_prompt(self, inp, out, is_chat):
        pass


def convert_to_prompt(inp, out):
    return "".join(
        [tok for x, y in zip(inp, out) for tok in (f"in: {x}, ", f"out: {int(y)}\n")][
            :-1
        ]
    )


class CleanTransducerPrompter(TransducerPrompter):
    def display_prompt(self, inp, out, is_chat):
        assert (
            not is_chat
        ), "chat systems won't work with this because we don't have logits"
        return convert_to_prompt(inp, out)

    def prompt_kwargs(self):
        return {
            "logprobs": 2,
            "max_tokens": 4,
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

    def display(self):
        return f"BasicInstructionTransducerPrompter({self.num_symbols})"

    def display_prompt(self, inp, out):
        pattern = super().display_prompt(inp, out)
        return (
            "A DFA was used to create these outputs given a random sequence of inputs. "
            f"Your job is to fill in the last output:\n{pattern}"
        )


class ChainOfThoughtPrompt(TransducerPrompter):

    def __init__(self, num_symbols, version=1):
        assert version == 1, "version mismatch"
        super().__init__(num_symbols)
        self.version = version

    def display(self):
        return f"ChainOfThoughtPrompt({self.num_symbols}, {self.version})"

    def display_prompt(self, inp, out, is_chat):
        assert is_chat, "for now, we only support chat systems for this prompter"
        return dict(
            system="You are a question answering system. For each question, think step by step and place your answer between tags like <answer>0</answer> or <answer>1</answer>\n"
            "QUESTION:\nWhat is 20*2?\n"
            + "ANSWER:\nTo solve this problem, we need to multiply 20 by 0. We can accomplish this via noticing that anything multiplied by 0 is 0. <answer>0</answer>\n",
            user="QUESTION:\nA DFA was used to create these outputs given a random sequence of inputs. "
            + f"Your job is to fill in the last output:\n"
            + convert_to_prompt(inp, out)
            + "out: _\n"
            + "\nANSWER:\n",
        )

    def prompt_kwargs(self):
        return {
            "max_tokens": 5000,
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
