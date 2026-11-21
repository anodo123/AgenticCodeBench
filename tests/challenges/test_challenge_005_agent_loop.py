"""Evaluator for Challenge 005: agent loop and stopping.

The evaluator checks learner-owned orchestration: real registered executions,
dependent outputs returned to the model, response-chain continuity, the final
non-tool stopping turn, and the bounded number of model turns.
"""

import json
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import pytest

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.tools import TOOL_REGISTRY
from agentic_code_bench.solutions import challenge_005_agent_loop as solution

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[2] / "challenges" / "005_agent_loop.json"
)


def test_challenge_005_agent_loop(monkeypatch: pytest.MonkeyPatch) -> None:
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

    assert set(actual) == {
        "steps",
        "result",
        "call_count",
        "model_turn_count",
    }
    assert actual["steps"] == executions == expected["steps"]
    assert type(actual["result"]) is int
    assert actual["result"] == executions[-1]["result"] == expected["result"]
    assert type(actual["call_count"]) is int
    assert actual["call_count"] == len(executions) == expected["call_count"]
    assert type(actual["model_turn_count"]) is int
    assert actual["model_turn_count"] == len(requests)
    assert len(requests) == expected["model_turn_count"]
    assert len(requests) <= solution.MAX_MODEL_TURNS

    assert requests[0]["messages"] == challenge["messages"]
    assert requests[0].get("previous_response_id") is None

    for index, execution in enumerate(executions):
        response = requests[index]["response"]
        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        assert len(function_calls) == 1

        next_request = requests[index + 1]
        assert next_request["previous_response_id"] == response.id
        assert len(next_request["messages"]) == 1
        tool_output = next_request["messages"][0]
        assert tool_output["type"] == "function_call_output"
        assert tool_output["call_id"] == function_calls[0].call_id
        assert isinstance(tool_output["output"], str)
        assert json.loads(tool_output["output"]) == execution["result"]

    final_response = requests[-1]["response"]
    final_function_calls = [
        item for item in final_response.output if item.type == "function_call"
    ]
    assert final_function_calls == []
    assert any(item.type == "message" for item in final_response.output)

    for actual_step, expected_step in zip(
        actual["steps"], expected["steps"], strict=True
    ):
        assert type(actual_step["result"]) is type(expected_step["result"])
