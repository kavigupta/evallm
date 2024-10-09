import re
from abc import abstractmethod

import numpy as np

from evallm.sample_dfa.sample_dfa import sample_dfa
from evallm.sample_dfa.transduce import transduce

from .prompter import Prompter

ANSWER_PATTERN = re.compile(r"<answer>([01])</answer>")


class TransducerPrompter(Prompter):

    def __init__(self, num_symbols):
        self.num_symbols = num_symbols

    @classmethod
    @abstractmethod
    def for_setting(cls, setting_kwargs):
        pass

    def prompt_and_answer(self, dfa, rng, is_chat):
        inp, out = self.input_output(dfa, rng)

        return (inp, out), self.display_prompt(inp, out, is_chat), out[-1]

    def input_output(self, dfa, rng):
        inp = rng.choice(sorted(dfa.input_symbols), size=self.num_symbols)
        out = transduce(dfa, inp)
        return inp, out

    def trivial(self, metas, answers):
        return len(set(answers)) == 1

    @abstractmethod
    def display_prompt(self, inp, out, is_chat):
        pass


def serialize_transducer_prompt(inp, out):
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
        return serialize_transducer_prompt(inp, out)

    def prompt_kwargs(self):
        return {
            "logprobs": 2,
            "max_tokens": 4,
        }

    def score_completion(self, output, choice):
        if choice is None:
            return np.zeros((2, 2))
        logprobs = choice.logprobs.top_logprobs[-1]
        probs = {k: np.exp(v) for k, v in logprobs.items() if k in ("0", "1")}
        denom = sum(probs.values())
        probs = {k: v / denom for k, v in probs.items()}
        confusion = np.zeros((2, 2))
        for j in range(2):
            confusion[int(output), j] = probs.get(str(j), 0)
        return confusion


class BasicInstructionTransducerPrompter(CleanTransducerPrompter):
    version = 3

    def __init__(self, num_symbols, *, strip=False):
        super().__init__(num_symbols)
        self.strip = strip

    @classmethod
    def for_setting(cls, setting_kwargs):
        return cls(
            setting_kwargs["num_sequence_symbols"],
            strip=setting_kwargs.get("strip", False),
        )

    def display(self):
        return (
            f"BasicInstructionTransducerPrompter({self.num_symbols}"
            + (f", strip={self.strip}" if self.strip else "")
            + f", version={self.version})"
        )

    def display_prompt(self, inp, out, is_chat):
        assert (
            not is_chat
        ), "for now, we only support non-chat systems for this prompter"
        pattern = super().display_prompt(inp, out, is_chat)
        result = (
            "A DFA was used to create these outputs given a random sequence of inputs. "
            f"Your job is to fill in the last output:\n{pattern}"
        )
        if self.strip:
            result = result.strip()
        return result


class ChainOfThoughtPrompt(TransducerPrompter):

    version = 3

    def display(self):
        return f"ChainOfThoughtPrompt({self.num_symbols}, {self.version})"

    def display_prompt(self, inp, out, is_chat):
        assert is_chat, "for now, we only support chat systems for this prompter"
        return dict(
            system="You are a question answering system. For each question, think step by step and"
            + " place your answer between tags like <answer>0</answer> or <answer>1</answer>."
            + " MAKE SURE TO PUT YOUR ANSWER IN ANSWER TAGS or you will get NO CREDIT.\n"
            + "QUESTION:\nWhat is 20*2?\n"
            + "ANSWER:\nTo solve this problem, we need to multiply 20 by 0. We can accomplish this"
            + " via noticing that anything multiplied by 0 is 0. <answer>0</answer>\n",
            user="QUESTION:\nA DFA was used to create these outputs given a random"
            + " sequence of inputs. "
            + "Your job is to fill in the last output:\n"
            + serialize_transducer_prompt(inp, out)
            + "out: _\n"
            + "\nANSWER:\n",
        )

    def prompt_kwargs(self):
        return {
            "max_tokens": 5000,
        }

    def get_numeric_answer(self, message_content):
        answer_tag_result = [
            x.group(1) for x in ANSWER_PATTERN.finditer(message_content)
        ]
        if answer_tag_result:
            return int(answer_tag_result[-1])
        last_index_1 = message_content.rfind("1")
        last_index_0 = message_content.rfind("0")
        if last_index_0 == -1 and last_index_1 == -1:
            # neither occurs in the message
            return None
        if last_index_1 == -1:
            # only 0 occurs in the message
            return 0
        if last_index_0 == -1:
            # only 1 occurs in the message
            return 1
        # both occur in the message, return the one that occurs last
        return 1 if last_index_1 > last_index_0 else 0

    def score_completion(self, output, choice):
        numeric = self.get_numeric_answer(choice.message.content)
        return numeric_answer_to_confusion(output, numeric)


def numeric_answer_to_confusion(output, numeric):
    output = int(output)
    confusion = np.zeros((2, 2))
    if numeric is not None:
        assert numeric in (0, 1)
        confusion[output, numeric] = 1
    else:
        confusion[output, :] = 0.5
    return confusion


class ChainOfThoughtPromptRealExampleNoExplanation(ChainOfThoughtPrompt):

    version = 0

    def __init__(self, num_symbols, sample_dfa_spec, num_samples):
        super().__init__(num_symbols)
        self.sample_dfa_spec = sample_dfa_spec
        self.num_samples = num_samples
        rng = np.random.RandomState(2**32 - 2)
        self.samples = [sample_dfa(sample_dfa_spec, rng) for _ in range(num_samples)]
        self.io = [self.input_output(dfa, rng) for dfa in self.samples]
        self.user_prompts = [
            ChainOfThoughtPrompt.display_prompt(self, inp, out, is_chat=True)["user"]
            + f"<answer>{int(out[-1])}</answer>\n\n"
            for (inp, out) in self.io
        ]

    def display(self):
        return (
            f"ChainOfThoughtPromptRealExamples({self.num_symbols},"
            f" {self.sample_dfa_spec}, {self.num_samples}, {self.version})"
        )

    def display_prompt(self, inp, out, is_chat):
        prompt = ChainOfThoughtPrompt.display_prompt(self, inp, out, is_chat)
        prompt["user"] = "".join(self.user_prompts) + prompt["user"]
        return prompt


class BasicSequencePrompt(TransducerPrompter):

    version = 3

    @classmethod
    def for_setting(cls, setting_kwargs):
        return BasicSequencePrompt(setting_kwargs["num_sequence_symbols"])

    def display(self):
        return f"BasicSequencePrompt({self.num_symbols}, {self.version})"

    def display_prompt(self, inp, out, is_chat):
        assert is_chat, "for now, we only support chat systems for this prompter"
        return dict(
            system="",
            user="You are a sequence completion model."
            + " Output the next element of the sequence, and nothing else.\n\n"
            + self.packed_sequence(inp, out),
        )

    def packed_sequence(self, inp, out):
        seq_els = [x + ", " for (a, b) in zip(inp, out) for x in [a, str(int(b))]]
        seq_els.pop()
        return "".join(seq_els)

    def prompt_kwargs(self):
        return {
            "max_tokens": 5,
            "temperature": 0.0,
        }

    def get_numeric_answer(self, message_content):
        # neihther 0 nor 1 occurs in the message, return None
        if "0" not in message_content and "1" not in message_content:
            return None
        first_0 = message_content.find("0")
        first_1 = message_content.find("1")
        if first_0 == -1:
            return 1
        if first_1 == -1:
            return 0
        # both occur in the message, return the one that occurs first
        return 0 if first_0 < first_1 else 1

    def score_completion(self, output, choice):
        numeric = self.get_numeric_answer(choice.message.content)
        return numeric_answer_to_confusion(output, numeric)


class SequencePromptWithExplanation(BasicSequencePrompt):

    version = 0

    def __init__(self, num_symbols, num_states):
        super().__init__(num_symbols)
        self.num_states = num_states

    @classmethod
    def for_setting(cls, setting_kwargs):
        return cls(
            setting_kwargs["num_sequence_symbols"],
            setting_kwargs["num_states"],
        )

    def display(self):
        return f"SequencePromptWithExplanation({self.num_symbols}, {self.num_states}, {self.version})"

    def display_prompt(self, inp, out, is_chat):
        assert is_chat, "for now, we only support chat systems for this prompter"
        return dict(
            system="",
            user="A DFA is a finite-state machine that accepts or rejects a given string of symbols,"
            + " by running through a n-state sequence uniquely determined by the string."
            + "\n\n"
            + f"I have a {self.num_states}-state DFA model that outputs either 0 or 1 after each element"
            + ' I input. 1 indicates that the input string thus far results in a "valid" state, and 0'
            + " indicates that it does not. I collect the inputs and outputs into an input sequence and"
            + " an output sequence. Infer the underlying DFA model to predict the next integer in the"
            + f" output sequence. {self.terminal_instruction()}"
            + "\n\n"
            + "Input sequence: "
            + "".join([f"{x}, " for x in inp])[0:-2]
            + "\n"
            + "Output sequence: "
            + "".join([f"{int(x)}, " for x in out[:-1]]),
        )

    def terminal_instruction(self):
        return "Only output the next output and nothing else."


class SequencePromptWithExplanationChainOfThought(SequencePromptWithExplanation):

    version = 1

    def __init__(self, num_symbols, num_states):
        super().__init__(num_symbols, num_states)

    def display(self):
        return f"SequencePromptWithExplanationChainOfThought({self.num_symbols}, {self.num_states}, {self.version})"

    def terminal_instruction(self):
        return (
            "Reason step by step, and then output the next output"
            " integer using <output> tags, like <output>0</output>."
        )

    def get_numeric_answer(self, message_content):
        return ChainOfThoughtPrompt.get_numeric_answer(self, message_content)

    def prompt_kwargs(self):
        return {
            "max_tokens": 5000,
            "temperature": 0.0,
        }
