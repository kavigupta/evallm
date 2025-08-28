import re
from abc import ABC, abstractmethod


class SequenceCompletionPrompt(ABC):

    @classmethod
    @abstractmethod
    def for_setting(cls, setting_kwargs):
        pass

    @abstractmethod
    def hash_prompt(self):
        pass

    @abstractmethod
    def display_prompt(self, dfa, sequences, prefix, is_chat):
        pass

    @abstractmethod
    def score_response(self, dfa, sequences, prefix, response):
        pass

    @abstractmethod
    def model_kwargs(self):
        pass


class SequencePromptDirect(SequenceCompletionPrompt):
    """
    Sequence prompt that directly displays the sequences to complete.
    """

    def display_prompt(self, dfa, sequences, prefix, is_chat):
        prompt = self.preamble()
        prompt += "\n"
        prompt += "\n".join(
            self.format_sequence(dfa, sequence) for sequence in sequences
        )
        prompt += self.instructions_before_prefix()
        prompt += "\n"
        prompt += self.format_sequence(dfa, prefix) + self.terminate_prefix()
        if is_chat:
            prompt = {"system": "", "user": prompt}
        return prompt

    @abstractmethod
    def preamble(self):
        pass

    def instructions_before_prefix(self):
        return "\n\nComplete the following string:"

    def terminate_prefix(self):
        return ""

    @abstractmethod
    def format_sequence(self, dfa, sequence):
        pass

    def score_response(self, dfa, sequences, prefix, response):
        prediction = ""
        for tok in response:
            if tok in dfa.input_symbols:
                prediction += tok
            elif tok == " ":
                continue
            else:
                break
        if not prediction:
            return 0.5
        return dfa.accepts_input([*prefix, *prediction])


class SequencePromptDirectAlien(SequencePromptDirect):

    def __init__(self, max_out_characters):
        self.max_out_characters = max_out_characters

    @classmethod
    def for_setting(cls, setting_kwargs):
        return cls(
            max_out_characters=setting_kwargs["num_sequence_symbols"]
            - setting_kwargs["num_sequence_symbols_prompt"]
        )

    def hash_prompt(self):
        return f"SequencePromptDirectAlien({self.max_out_characters})"

    def format_sequence(self, dfa, sequence):
        return "".join(sequence)

    def preamble(self):
        return (
            "The following strings come from an alien language that follows a simple grammar."
            + " Infer the alien grammar using the example strings. Then, add a suffix to the final string"
            + f" using between 1 and {self.max_out_characters} characters such that the full string"
            + " follows the grammar. Output the necessary suffix, and nothing else."
            + "\n\n"
            + "Given these grammatically correct example strings:"
        )

    def model_kwargs(self):
        return {"max_tokens": self.max_out_characters * 5, "temperature": 0.0}


class SequencePromptDirectAlien2(SequencePromptDirectAlien):
    def hash_prompt(self):
        return f"SequencePromptDirectAlien2({self.max_out_characters})"

    def preamble(self):
        return (
            "The following strings come from an alien language that follows a simple grammar."
            + " Infer the alien grammar using the example strings. Then, add a suffix to the final string"
            + f" using between 1 and {self.max_out_characters} characters such that the full string"
            + f" follows the grammar. {self.end_of_preamble()}"
            + "\n"
        )

    def end_of_preamble(self):
        return "Output only the necessary suffix to complete the final string, and nothing else."

    def instructions_before_prefix(self):
        return ""


class SequencePromptDirectAlien2WithSpaces(SequencePromptDirectAlien2):

    def hash_prompt(self):
        return f"SequencePromptDirectAlien2WithSpaces({self.max_out_characters})"

    def format_sequence(self, dfa, sequence):
        return " ".join(sequence)


class SequencePromptDirectAlien2WithCommas(SequencePromptDirectAlien2):
    version = 2

    def hash_prompt(self):
        return f"SequencePromptDirectAlien2WithCommas({self.max_out_characters}, {self.version})"

    def format_sequence(self, dfa, sequence):
        return ", ".join(sequence)

    def terminate_prefix(self):
        return ","


class MoreExplanationPrompt(SequencePromptDirectAlien):

    def __init__(self, max_out_characters, num_states):
        super().__init__(max_out_characters)
        self.num_states = num_states

    @classmethod
    def for_setting(cls, setting_kwargs):
        return cls(
            max_out_characters=setting_kwargs["num_sequence_symbols"]
            - setting_kwargs["num_sequence_symbols_prompt"],
            num_states=setting_kwargs["dfa_spec"]["n_states"],
        )

    def hash_prompt(self):
        return f"MoreExplanationPrompt({self.max_out_characters}, {self.num_states})"

    def preamble(self):
        return (
            f"I have a {self.num_states}-state DFA model that outputs either 0 or 1 after each element I input."
            + ' 1 indicates that the input string thus far results in a "valid" state,'
            + " and 0 indicates that it does not. I collect a set of valid strings using this DFA,"
            + " listed below. Infer the underlying DFA model using these strings and complete the final string,"
            + " using up to n characters, such that it is also a valid string."
            + " "
            + self.end_of_preamble()
            + "\n\n"
            + "Given these valid strings:"
        )

    def end_of_preamble(self):
        return "Output the necessary suffix for this final string, and nothing else."


class MoreExplanationPrompt2(SequencePromptDirectAlien):

    def __init__(self, max_out_characters, num_states):
        super().__init__(max_out_characters)
        self.num_states = num_states

    @classmethod
    def for_setting(cls, setting_kwargs):
        return cls(
            max_out_characters=setting_kwargs["num_sequence_symbols"]
            - setting_kwargs["num_sequence_symbols_prompt"],
            num_states=setting_kwargs["dfa_spec"]["n_states"],
        )

    def hash_prompt(self):
        return f"MoreExplanationPrompt2({self.max_out_characters}, {self.num_states})"

    def preamble(self):
        return (
            f"I have a {self.num_states}-state DFA model that outputs either 0 or 1 after each element I input."
            + ' 1 indicates that the input string thus far results in a "valid" state,'
            + " and 0 indicates that it does not. I collect a set of valid strings using this DFA,"
            + " listed below. Infer the underlying DFA model using these strings and complete the final string,"
            + " using up to n characters, such that it is also a valid string."
            + " "
            + self.end_of_preamble()
        )

    def end_of_preamble(self):
        return "Output only the necessary suffix to complete the final string, and nothing else."

    def instructions_before_prefix(self):
        return ""


ANSWER_PATTERN = re.compile(r"<answer>([^<]+)</answer>")


class MoreExplanationPromptCOT(MoreExplanationPrompt):

    version = 3

    @classmethod
    def for_setting(cls, setting_kwargs):
        return cls(
            max_out_characters=setting_kwargs["num_sequence_symbols"]
            - setting_kwargs["num_sequence_symbols_prompt"],
            num_states=setting_kwargs["dfa_spec"]["n_states"],
        )

    def hash_prompt(self):
        return f"MoreExplanationPromptCOT({self.max_out_characters}, {self.num_states}, {self.version})"

    def end_of_preamble(self):
        return (
            "Reason step by step, and then output the next necessary suffix for this final string,"
            + f" <answer> tags, like <answer>{self.format_sequence(None, ['a', 'b'])}</answer>."
        )

    def score_response(self, dfa, sequences, prefix, response):
        match = ANSWER_PATTERN.search(response)
        if match is None:
            return 0.5
        prediction = match.group(1)
        return super().score_response(dfa, sequences, prefix, prediction)

    def model_kwargs(self):
        return {"max_tokens": 4090, "temperature": 0.0}


class SequencePromptDirectAlien2COT(SequencePromptDirectAlien2):
    version = 1

    @classmethod
    def for_setting(cls, setting_kwargs):
        return cls(
            max_out_characters=setting_kwargs["num_sequence_symbols"]
            - setting_kwargs["num_sequence_symbols_prompt"],
        )

    def hash_prompt(self):
        return f"SequencePromptDirectAlien2COT({self.max_out_characters}, {self.version})"

    def end_of_preamble(self):
        return MoreExplanationPromptCOT.end_of_preamble(self)

    def score_response(self, dfa, sequences, prefix, response):
        return MoreExplanationPromptCOT.score_response(
            self, dfa, sequences, prefix, response
        )

    def model_kwargs(self):
        return MoreExplanationPromptCOT.model_kwargs(self)


class RedGreenPrompt(MoreExplanationPromptCOT):

    version = 1

    @classmethod
    def for_setting(cls, setting_kwargs):
        return cls(
            max_out_characters=setting_kwargs["num_sequence_symbols"]
            - setting_kwargs["num_sequence_symbols_prompt"],
            num_states=setting_kwargs["dfa_spec"]["n_states"],
        )

    def hash_prompt(self):
        return f"RedGreenPrompt({self.max_out_characters}, {self.version})"

    def preamble(self):
        return (
            f"You are outside a house of rooms and portals. There are {self.num_states} rooms in the house, and each room has 3"
            + " unique portals labeled a, b, and c. Each portal teleports you to one room of the house"
            + " (and sometimes the destination is the room the portal is in). Every portal in a given room always behaves the same way."
            + "\n\n"
            + "In this house, each of the rooms look exactly the same, except some of the rooms have red walls and some"
            f" have green walls. However, there are *{self.num_states}* rooms in total, so you cannot determine which room"
            " you are in by color alone, and two rooms of the same color may have portals that behave differently."
            " You've been into this house many times before. Each time, as you move through the house, you write down"
            " what series of portals you take and the color of the room you end up in. You have a collection of paths you've"
            " taken where you've ended up in a room with green walls, listed below. Given the final incomplete path at the bottom,"
            f" write a series of up to {self.max_out_characters} remaining steps that will cause you to end up in a room with green walls again."
            + "\n\n"
            + f"Tag your final answer like <answer>{self.format_sequence(None, ['a', 'b'])}</answer>."
            + "\n\n"
            + "Given these paths that end in a room with green walls:"
        )

    def end_of_preamble(self):
        raise NotImplementedError

    def instructions_before_prefix(self):
        return "\n\nComplete the following path:"


class WithTemperatureSequenceCompletionPrompt(SequenceCompletionPrompt):
    """
    A sequence completion prompt that allows for temperature settings.
    """

    def __init__(self, underlying_prompt, temperature=0.0):
        self.underlying_prompt = underlying_prompt
        self.temperature = temperature

    @classmethod
    def for_setting(cls, setting_kwargs):
        raise NotImplementedError(
            "This prompt does not support for_setting. Use the underlying prompt's for_setting."
        )

    def hash_prompt(self):
        return f"WithTemperatureSequenceCompletionPrompt({self.underlying_prompt.hash_prompt()}, {self.temperature})"

    def display_prompt(self, dfa, sequences, prefix, is_chat):
        return self.underlying_prompt.display_prompt(dfa, sequences, prefix, is_chat)

    def score_response(self, dfa, sequences, prefix, response):
        return self.underlying_prompt.score_response(dfa, sequences, prefix, response)

    def model_kwargs(self):
        kwargs = self.underlying_prompt.model_kwargs()
        kwargs = kwargs.copy()
        kwargs["temperature"] = self.temperature
        return kwargs
