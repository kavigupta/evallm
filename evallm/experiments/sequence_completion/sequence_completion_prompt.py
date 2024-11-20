from abc import ABC, abstractmethod

import numpy as np


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
        prompt += "\n\n"
        prompt += "Complete the following string:"
        prompt += "\n"
        prompt += self.format_sequence(dfa, prefix)
        if is_chat:
            prompt = {"system": "", "user": prompt}
        return prompt

    @abstractmethod
    def preamble(self):
        pass

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


class SequencePromptDirectAlienWithSpaces(SequencePromptDirectAlien):

    def hash_prompt(self):
        return f"SequencePromptDirectAlienWithSpaces({self.max_out_characters})"

    def format_sequence(self, dfa, sequence):
        return " ".join(sequence)
