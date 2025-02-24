from dataclasses import dataclass
from typing import TypeVar

import torch

from delphi.latents import (
    ActivatingExample,
    Example,
    LatentRecord,
    NonActivatingExample,
)
from delphi.scorers.classifier.classifier import Classifier, ScorerResult
from delphi.scorers.classifier.sample import ClassifierOutput

from ..default.prompt_builder import build_prompt as default_prompt
from ..explainer import Explainer
from .prompt_builder import build_prompt

# we use this type variable to ensure that the examples are either
# ActivatingExample or NonActivatingExample
Examples = TypeVar("Examples", bound=Example)


@dataclass
class IterativeExplainer(Explainer):
    activations: bool = True
    """Whether to show activations to the explainer."""

    def _to_string_examples(
        self, examples: list[Examples], show_activations: bool
    ) -> str:
        highlighted_examples = []

        for i, example in enumerate(examples):
            str_toks = example.str_tokens
            activations = example.activations.tolist()
            highlighted_examples.append(self._highlight(str_toks, activations))

            if show_activations:
                assert (
                    example.normalized_activations is not None
                ), "Normalized activations are required for activations in explainer"
                normalized_activations = example.normalized_activations.tolist()
                highlighted_examples.append(
                    self._join_activations(
                        str_toks, activations, normalized_activations
                    )
                )

        highlighted_examples = "\n".join(highlighted_examples)

        return highlighted_examples

    def _build_prompt(self, record: LatentRecord) -> list[dict]:
        examples = record.train
        if record.explanation == "":
            # If there is no explanation, we use the default prompt
            highlighted_examples = self._to_string_examples(examples, self.activations)
            return default_prompt(highlighted_examples, self.activations)
        else:
            # If there is explanation, use the explanation, the normal examples
            # and the extra examples
            number_extra_examples = len(record.extra_examples)
            # we show at most 10 extra examples
            number_extra_examples = min(number_extra_examples, 10)
            normal_examples = self._to_string_examples(
                examples[:-number_extra_examples], self.activations
            )
            extra_examples = self._to_string_examples(
                record.extra_examples[:number_extra_examples], False
            )
            return build_prompt(record.explanation, normal_examples, extra_examples)


@dataclass
class HillClimbing:
    scorer: Classifier
    """Scorer to use for explanation generation."""

    explainer: IterativeExplainer
    """Explainer to use for explanation generation."""

    n_loops: int = 5
    """Number of loops to run the explanation generation."""

    def _compute_score(self, results: list[ClassifierOutput]) -> float:
        score = 0
        for i, sample in enumerate(results):
            if sample.correct:
                score += 1

        print(f"Score: {score}")

        return score / len(results)

    def _get_wrong_examples(self, results: list[ClassifierOutput]) -> list[Example]:
        wrong_examples = []
        for i, sample in enumerate(results):
            if not sample.correct:
                # Create a extra example
                if sample.activating:
                    new_example = ActivatingExample(
                        tokens=torch.tensor(0),  # this does not matter
                        activations=torch.tensor(sample.activations),
                        str_tokens=sample.str_tokens,
                        normalized_activations=torch.tensor(sample.activations),
                    )
                else:
                    new_example = NonActivatingExample(
                        tokens=torch.tensor(0),  # this does not matter
                        activations=torch.tensor(sample.activations),
                        str_tokens=sample.str_tokens,
                    )
                wrong_examples.append(new_example)
        return wrong_examples

    async def __call__(self, record: LatentRecord) -> list[ScorerResult] | None:

        first_generation_examples = record.train
        held_out_set_size = len(record.test) // 3
        held_out_activating_examples = record.test[:held_out_set_size]
        held_out_non_activating_examples = record.not_active[:held_out_set_size]
        train_test_activating_examples = record.test[held_out_set_size:]
        train_test_non_activating_examples = record.not_active[held_out_set_size:]

        first_explanation = await self.explainer(record)
        if first_explanation.explanation == "Explanation could not be parsed.":
            # TODO: Do I want this?
            return None

        test_record = LatentRecord(
            latent=record.latent,
            train=first_generation_examples,
            not_active=held_out_non_activating_examples,
            test=held_out_activating_examples,
            explanation=first_explanation.explanation,
        )

        results = await self.scorer(test_record)
        print("----- Holdout score ------")
        _ = self._compute_score(results.score)

        number_examples_per_loop = len(train_test_activating_examples) // self.n_loops
        scores = [results]
        for i in range(self.n_loops):
            start_idx = i * number_examples_per_loop
            end_idx = (i + 1) * number_examples_per_loop
            selected_activating = train_test_activating_examples[start_idx:end_idx]
            selected_non_active = train_test_non_activating_examples[start_idx:end_idx]
            new_record = LatentRecord(
                latent=record.latent,
                train=first_generation_examples,
                not_active=selected_non_active,
                test=selected_activating,
                explanation=first_explanation.explanation,
            )
            results = await self.scorer(new_record)
            print("----- Train score ------")
            _ = self._compute_score(results.score)
            # get the wrong examples
            wrong_examples = self._get_wrong_examples(results.score)
            # update the record
            record.extra_examples.extend(wrong_examples)
            # update the explanation
            new_explanation = await self.explainer(record)
            if new_explanation.explanation == "Explanation could not be parsed.":
                pass  # we do not update the explanation
            else:
                first_explanation = new_explanation

            # compute the score in the held out set
            test_record.explanation = new_explanation.explanation
            results = await self.scorer(test_record)
            print("----- Holdout score ------")
            _ = self._compute_score(results.score)
            scores.append(results)

            return scores
