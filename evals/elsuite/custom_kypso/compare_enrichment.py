import json
import random
from typing import Any, Dict, List, Mapping, Union, cast

import numpy as np

import evals
from evals.api import CompletionFn
from evals.record import RecorderBase


def json_match(sampled_json: Any, correct_json: Any) -> bool:
    # Check boolean values
    boolean_keys = [
        key for key, value in sampled_json.items() if isinstance(value, bool)
    ]
    boolean_values_equal = all(
        sampled_json[key] == correct_json[key] for key in boolean_keys
    )

    # Check list lengths
    list_keys = [key for key, value in sampled_json.items() if isinstance(value, list)]
    list_length_equal = all(
        len(sampled_json[key]) == len(correct_json[key]) for key in list_keys
    )

    return boolean_values_equal and list_length_equal


class JsonMatch(evals.Eval):

    """Compares a JSON completion with one or more ideal answers,
    also coded in JSON. The decoded JSON objects are compared
    elementwise and must match exactly."""

    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        *args: Any,
        max_tokens: int = 512,  # Increase this for longer JSON completions
        **kwargs: Any,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "JsonMatch only supports one completion fn"
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl

    def eval_sample(self, sample: Any, rng: random.Random):
        del rng

        assert isinstance(sample, dict), "sample must be a dict"
        assert "input" in sample, "sample must have an 'input' key"
        assert "ideal" in sample, "sample must have an 'ideal' key"

        prompt = cast(str, sample["input"])

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,  # Q: why are these hardcoded?
            max_tokens=self.max_tokens,
        )
        sampled = result.get_completions()[0]

        sampled_json: Any
        try:
            sampled_json = json.loads(sampled)
        except ValueError:
            # If the sampled string is not valid JSON, it will never match
            sampled_json = None

        # Allow the following to raise ValueError; the correct answers
        # should always be valid JSON
        correct_json = json.loads(sample["ideal"])

        is_match = json_match(sampled_json, correct_json)

        evals.record.record_match(
            is_match,
            expected=sample["ideal"],
            # picked=[sampled for i in range(len(correct_answers)) if matches[i]],
        )
        evals.record.record_metrics(
            accuracy=float(is_match),
        )

    def run(self, recorder: RecorderBase) -> Dict[str, float]:
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)

        return {
            "accuracy": np.mean(recorder.get_scores("accuracy")),
        }
