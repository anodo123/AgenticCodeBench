"""Evaluator for Challenge 007: missing information.

It verifies that the model asks for the missing height, no registered tool is
executed, and the learner returns a truthful one-turn ``needs_input`` result.
"""

from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import pytest

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.tools import TOOL_REGISTRY
from agentic_code_bench.solutions import challenge_007_missing_information as solution

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "challenges"
    / "007_missing_information.json"
)


def test_challenge_007_missing_information(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    challenge = load_challenge(CHALLENGE_PATH)
    expected = challenge["expected"]
    executions: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    original_model_call = solution.call_openai_model

    def record_tool(
        name: str, function: Callable[..., object]
    ) -> Callable[..., object]:
        @wraps(function)
        def recorded(**arguments: Any) -> object:
            executions.append({"tool": name, "arguments": arguments})
            return function(**arguments)

        return recorded

    def record_model_request(**kwargs: Any) -> Any:
        response = original_model_call(**kwargs)
        requests.append({**kwargs, "response": response})
        return response

    for name, function in TOOL_REGISTRY.items():
        monkeypatch.setitem(TOOL_REGISTRY, name, record_tool(name, function))
    monkeypatch.setattr(solution, "call_openai_model", record_model_request)

    actual = solution.solve()

    assert set(actual) == {
        "status",
        "message",
        "call_count",
        "model_turn_count",
    }
    assert actual["status"] == expected["status"] == "needs_input"
    assert isinstance(actual["message"], str)
    assert actual["message"].strip()
    assert expected["missing_field"] in actual["message"].lower()
    assert type(actual["call_count"]) is int
    assert actual["call_count"] == len(executions) == expected["call_count"] == 0
    assert type(actual["model_turn_count"]) is int
    assert actual["model_turn_count"] == len(requests) == 1
    assert actual["model_turn_count"] == expected["model_turn_count"]

    request = requests[0]
    response = request["response"]
    assert request["messages"] == challenge["messages"]
    assert request["tools"] == challenge["tools"]
    assert request.get("previous_response_id") is None
    assert not any(item.type == "function_call" for item in response.output)
    assert any(item.type == "message" for item in response.output)
    assert response.output_text.strip() == actual["message"].strip()
