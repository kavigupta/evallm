import re
from abc import abstractmethod

import numpy as np

from evallm.sample_dfa.sample_dfa import sample_dfa
from evallm.sample_dfa.transduce import transduce

from .prompter import Prompter

ANSWER_PATTERN = re.compile(r"<answer>([01])</answer>")
ANSWER_PATTERN_RED_GREEN = re.compile(r"<answer>(red|green)</answer>", re.IGNORECASE)


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


class BasicSequencePromptNoChat(BasicSequencePrompt):
    version = 0

    @classmethod
    def for_setting(cls, setting_kwargs):
        return BasicSequencePromptNoChat(setting_kwargs["num_sequence_symbols"])

    def display(self):
        return f"BasicSequencePromptNoChat({self.num_symbols}, {self.version})"

    def display_prompt(self, inp, out, is_chat):
        assert (
            not is_chat
        ), "for now, we only support non-chat systems for this prompter"
        result = super().display_prompt(inp, out, is_chat=True)
        assert not result["system"]
        return result["user"]

    def score_completion(self, output, choice):
        numeric = self.get_numeric_answer(choice.text)
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


class RedGreenRoomPrompt1(TransducerPrompter):

    version = 1

    def __init__(self, num_symbols, num_alphabet_symbols, num_states):
        super().__init__(num_symbols)
        self.num_alphabet_symbols = num_alphabet_symbols
        self.num_states = num_states

    @classmethod
    def for_setting(cls, setting_kwargs):
        return RedGreenRoomPrompt1(
            setting_kwargs["num_sequence_symbols"],
            setting_kwargs["sample_dfa_spec"]["n_symbols"],
            setting_kwargs["num_states"],
        )

    def display(self):
        return f"RedGreenRoomPrompt1({self.num_symbols}, {self.num_states}, {self.version})"

    def room_transcript(self, inp, out):
        inp = [x.upper() for x in inp]
        out = [{0: "red", 1: "green"}[int(x)] for x in out]
        lines = []
        lines += [
            f'You walk through a portal labeled "{inp[0]}" and end up in a {out[0]} room.'
        ]
        for i in range(1, len(inp)):
            prefix = (
                f'Then, you walk through a portal labeled "{inp[i]}" and end up in a '
            )
            if i == len(inp) - 1:
                lines += [prefix + "..."]
            else:
                lines += [prefix + f"{out[i]} room."]
        return "\n".join(lines)

    def display_prompt(self, inp, out, is_chat):
        assert (
            self.num_alphabet_symbols == 3
        ), "not implemented for num_alphabet_symbols != 3"
        assert is_chat, "for now, we only support chat systems for this prompter"
        return dict(
            system="",
            user="```"
            + "\n"
            + f"You are in a house of rooms and portals. There are {self.num_states} rooms in the house,"
            + " and each room has 3 unique portals labeled A, B, and C."
            + " Each portal teleports you to one room of the house (and sometimes the"
            + " destination is the room the portal is in). Every portal in a given room"
            + " always behaves the same way."
            + "\n\n"
            + "In this house, each of the rooms look exactly the same, except some of the rooms"
            + " have red walls and some have green walls. However, there are *three* rooms in total,"
            + " so you cannot determine which room you are in by color alone, and two rooms of the same"
            + " color may have portals that behave differently.  As you move through the house, at each"
            + " time step you write down what portal you take and the color of the room you arrive (or stay) in."
            + " Based on your notes, predict what color room you will end up in after the last step."
            + "\n\n"
            + "Tag your final answer like <answer>color</answer>."
            + "\n\n"
            + self.room_transcript(inp, out)
            + "\n"
            + "```",
        )

    def prompt_kwargs(self):
        # for backwards compatibility. 1000 tokens for 30 symbols
        return {"max_tokens": (1000 * self.num_symbols) // 30, "temperature": 0.0}

    def get_numeric_answer(self, message_content):
        answer_tag_result = [
            x.group(1) for x in ANSWER_PATTERN_RED_GREEN.finditer(message_content)
        ]
        if answer_tag_result:
            color = answer_tag_result[0].lower()
            return {"red": 0, "green": 1}[color]
        return None

    def score_completion(self, output, choice):
        numeric = self.get_numeric_answer(choice.message.content)
        return numeric_answer_to_confusion(output, numeric)
