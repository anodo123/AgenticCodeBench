"""Evaluator for Challenge 006: tool failure handling.

It verifies the actual exception, the failure record and model handoff, response
continuity, retry prevention, bounded execution, and final-message stopping.
"""

import json
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import pytest

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.tools import TOOL_REGISTRY
from agentic_code_bench.solutions import challenge_006_tool_failure_handling as solution

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "challenges"
    / "006_tool_failure_handling.json"
)


def test_challenge_006_tool_failure_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    challenge = load_challenge(CHALLENGE_PATH)
    expected = challenge["expected"]
    attempts: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    original_model_call = solution.call_openai_model

    def record_tool_attempt(
        name: str, function: Callable[..., object]
    ) -> Callable[..., object]:
        @wraps(function)
        def recorded(**arguments: Any) -> object:
            try:
                result = function(**arguments)
            except Exception as error:
                attempts.append(
                    {
                        "tool": name,
                        "arguments": arguments,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                raise
            attempts.append(
                {
                    "tool": name,
                    "arguments": arguments,
                    "status": "success",
                    "result": result,
                }
            )
            return result

        return recorded

    def record_model_request(**kwargs: Any) -> Any:
        response = original_model_call(**kwargs)
        requests.append({**kwargs, "response": response})
        return response

    for name, function in TOOL_REGISTRY.items():
        monkeypatch.setitem(
            TOOL_REGISTRY, name, record_tool_attempt(name, function)
        )
    monkeypatch.setattr(solution, "call_openai_model", record_model_request)

    actual = solution.solve()

    assert set(actual) == {
        "steps",
        "result",
        "status",
        "call_count",
        "model_turn_count",
    }
    assert actual["steps"] == attempts == expected["steps"]
    assert actual["result"] is expected["result"] is None
    assert actual["status"] == expected["status"] == "failed"
    assert type(actual["call_count"]) is int
    assert actual["call_count"] == len(attempts) == expected["call_count"]
    assert actual["call_count"] <= solution.MAX_TOOL_ATTEMPTS
    assert type(actual["model_turn_count"]) is int
    assert actual["model_turn_count"] == len(requests)
    assert len(requests) == expected["model_turn_count"]
    assert len(requests) <= solution.MAX_MODEL_TURNS

    first_request, second_request = requests
    assert first_request["messages"] == challenge["messages"]
    assert first_request.get("previous_response_id") is None

    first_calls = [
        item
        for item in first_request["response"].output
        if item.type == "function_call"
    ]
    assert len(first_calls) == 1
    assert second_request["previous_response_id"] == first_request["response"].id
    assert len(second_request["messages"]) == 1

    tool_output = second_request["messages"][0]
    assert tool_output["type"] == "function_call_output"
    assert tool_output["call_id"] == first_calls[0].call_id
    assert isinstance(tool_output["output"], str)
    assert json.loads(tool_output["output"]) == {
        "status": attempts[0]["status"],
        "error_type": attempts[0]["error_type"],
        "error_message": attempts[0]["error_message"],
    }

    final_response = second_request["response"]
    assert not any(
        item.type == "function_call" for item in final_response.output
    )
    assert any(item.type == "message" for item in final_response.output)
