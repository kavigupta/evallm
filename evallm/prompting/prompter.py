from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from evallm.llm.llm import run_prompt


class Prompter(ABC):

    @abstractmethod
    def prompt_and_answer(self, dfa, rng) -> Tuple[object, str, object]:
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
            *[self.prompt_and_answer(dfa, rng) for _ in range(num_samples)]
        )
        if self.trivial(metas, answers):
            raise TrivialProblemError
        completions = run_prompt(
            model=model,
            prompt=prompts,
            **self.prompt_kwargs(),
        )
        scores = [
            self.score_completion(answer, choice)
            for answer, choice in zip(answers, completions.choices)
        ]
        return metas, prompts, scores


class TrivialProblemError(Exception):
    pass
