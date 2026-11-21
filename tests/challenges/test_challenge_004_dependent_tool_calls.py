import json
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import pytest

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.tools import TOOL_REGISTRY
from agentic_code_bench.solutions import challenge_004_dependent_tool_calls as solution

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "challenges"
    / "004_dependent_tool_calls.json"
)


def test_challenge_004_dependent_tool_calls(
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
            result = function(**arguments)
            executions.append(
                {"tool": name, "arguments": arguments, "result": result}
            )
            return result

        return recorded

    def record_model_request(**kwargs: Any) -> Any:
        response = original_model_call(**kwargs)
        requests.append({**kwargs, "response": response})
        return response

    for name, function in TOOL_REGISTRY.items():
        monkeypatch.setitem(TOOL_REGISTRY, name, record_tool(name, function))
    monkeypatch.setattr(solution, "call_openai_model", record_model_request)

    actual = solution.solve()

    assert set(actual) == {"steps", "call_count"}
    assert type(actual["call_count"]) is int
    assert actual["call_count"] == len(actual["steps"]) == len(executions) == 2
    assert actual["steps"] == executions == expected["steps"]

    assert len(requests) == 2
    first_request, second_request = requests
    assert first_request["messages"] == challenge["messages"]
    assert first_request.get("previous_response_id") is None
    assert second_request["previous_response_id"] == first_request["response"].id

    first_function_calls = [
        item
        for item in first_request["response"].output
        if item.type == "function_call"
    ]
    assert len(first_function_calls) == 1
    assert len(second_request["messages"]) == 1
    tool_output = second_request["messages"][0]
    assert tool_output["type"] == "function_call_output"
    assert tool_output["call_id"] == first_function_calls[0].call_id
    assert isinstance(tool_output["output"], str)
    assert json.loads(tool_output["output"]) == executions[0]["result"]

    for actual_step, expected_step in zip(
        actual["steps"], expected["steps"], strict=True
    ):
        assert type(actual_step["result"]) is type(expected_step["result"])
