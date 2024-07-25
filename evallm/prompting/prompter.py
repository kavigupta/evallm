from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from evallm.llm.llm import model_specs, run_prompt


class Prompter(ABC):

    @abstractmethod
    def prompt_and_answer(self, dfa, rng, is_chat) -> Tuple[object, str, object]:
        """
        Return a triple of meta information, a prompt, and an answer.
        """

    @abstractmethod
    def prompt_kwargs(self) -> dict:
        pass

    @abstractmethod
    def trivial(self, metas, answers) -> bool:
        """
        Return True if the given metas and answers are trivial and should be discarded.
        """

    @abstractmethod
    def display(self) -> str:
        """
        Return a unique string representation of the prompter. This should be
        a valid Python expression that can be used to reconstruct the prompter.
        """

    @abstractmethod
    def score_completion(self, output, choice) -> np.ndarray:
        """
        Produce a confusion matrix of the completion's likelihood of each output.

        :param output: the correct output
        :param choice: the completion choice

        :return: a confusion matrix of the likelihood of each output.
            confusion[i, j] is the likelihood of the completion outputting j when the
            correct output is i. This should have all rows except for the correct output
            zeroed out.
        """

    def run_experiment(self, dfa, rng, model, num_samples):
        metas, prompts, answers = zip(
            *[
                self.prompt_and_answer(dfa, rng, is_chat=model_specs[model].is_chat)
                for _ in range(num_samples)
            ]
        )
        if self.trivial(metas, answers):
            raise TrivialProblemError
        print(prompts[0])
        completions = run_prompt(
            model=model,
            prompt=prompts,
            kwargs=self.prompt_kwargs(),
        )
        print(completions.choices[0])
        scores = [
            self.score_completion(answer, choice)
            for answer, choice in zip(answers, completions.choices)
        ]
        return metas, prompts, scores

    def __repr__(self):
        return self.display()


class TrivialProblemError(Exception):
    pass
