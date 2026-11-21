"""Offline evaluator for Challenge 008.

Question: With width 5, ask for height and waste percentage, then calculate
area, waste allowance, and total material through three dependent tools.
Verify the interactive loop and its failure paths without paid API calls.
"""

import json
from copy import deepcopy
from functools import wraps
from types import SimpleNamespace
from typing import Any

import pytest

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.tools import TOOL_REGISTRY
from agentic_code_bench.solutions import challenge_008_resume_after_clarification as solution


def message_response(status: str, message: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="", status="completed",
        output=[SimpleNamespace(type="reasoning"), SimpleNamespace(type="message")],
        output_text=json.dumps({"status": status, "message": message}),
    )


def tool_response(name: str, arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(
        id="", status="completed", output_text="",
        output=[SimpleNamespace(
            type="function_call", name=name,
            arguments=json.dumps(arguments), call_id=f"call_{name}",
        )],
    )


def install_model(
    monkeypatch: pytest.MonkeyPatch, responses: list[SimpleNamespace],
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for index, response in enumerate(responses):
        response.id = f"resp_{index + 1}"

    def fake_model(**kwargs: Any) -> SimpleNamespace:
        assert len(requests) < len(responses), "Unexpected extra model request"
        response = responses[len(requests)]
        requests.append({**deepcopy(kwargs), "response": response})
        return response

    monkeypatch.setattr(solution, "call_openai_model", fake_model)
    return requests


@pytest.mark.parametrize("height,waste", [(3, 10), (50, 1)])
def test_challenge_008_resume_after_clarification(
    monkeypatch: pytest.MonkeyPatch, height: int, waste: int,
) -> None:
    challenge = load_challenge(solution.CHALLENGE_PATH)
    area = 5 * height
    allowance = area * waste / 100
    total = area + allowance
    final_message = f"Total material required is {total} square units."
    # Include a tool execution between questions: the loop must not assume a
    # fixed sequence of question turns followed by tool turns.
    responses = [
        message_response("needs_input", "What is the height?"),
        tool_response("calculate_rectangle_area", {"width": 5, "height": height}),
        message_response("needs_input", "What waste percentage should be added?"),
        tool_response("calculate_percentage", {"amount": area, "percentage": waste}),
        tool_response("add_numbers", {"left": area, "right": allowance}),
        message_response("completed", final_message),
    ]
    requests = install_model(monkeypatch, responses)
    executions: list[dict[str, Any]] = []

    def record_tool(name: str, function: Any) -> Any:
        @wraps(function)
        def recorded(**arguments: Any) -> Any:
            result = function(**arguments)
            executions.append({"tool": name, "arguments": arguments, "result": result})
            return result
        return recorded

    for name, function in list(TOOL_REGISTRY.items()):
        monkeypatch.setitem(TOOL_REGISTRY, name, record_tool(name, function))

    replies = iter([str(height), str(waste)])
    user_turns: list[int] = []

    def read_user(prompt: str) -> str:
        assert prompt.strip()
        user_turns.append(len(requests))
        assert len(executions) == len(user_turns) - 1
        return next(replies)

    # The learner must not use the evaluator's answer key or example replies.
    learner_challenge = deepcopy(challenge)
    learner_challenge.pop("expected")
    learner_challenge.pop("example_user_replies")
    monkeypatch.setattr(solution, "load_challenge", lambda _: learner_challenge)
    actual = solution.solve(debug=False, read_user=read_user)

    assert actual["status"] == "completed"
    assert actual["message"] == final_message
    assert actual["steps"] == executions
    assert [step["tool"] for step in executions] == challenge["expected"]["ordered_tools"]
    assert [step["result"] for step in executions] == [area, allowance, total]
    assert [type(step["result"]) for step in executions] == [int, float, float]
    assert actual["result"] == total
    assert type(actual["call_count"]) is int
    assert actual["call_count"] == len(executions) == 3
    assert type(actual["model_turn_count"]) is int
    assert actual["model_turn_count"] == len(requests) == 6
    assert user_turns == [1, 3]
    assert actual["clarifications"] == [
        {"question": "What is the height?", "reply": str(height)},
        {"question": "What waste percentage should be added?", "reply": str(waste)},
    ]
    assert actual["previous_response_id"] == responses[-1].id
    assert requests[0]["messages"] == challenge["messages"]
    assert requests[0]["previous_response_id"] is None
    for index, request in enumerate(requests):
        assert request["tools"] == challenge["tools"]
        if index:
            assert request["previous_response_id"] == responses[index - 1].id
    assert requests[1]["messages"] == [{"role": "user", "content": str(height)}]
    assert requests[3]["messages"] == [{"role": "user", "content": str(waste)}]
    for response_index, result in [(1, area), (3, allowance), (4, total)]:
        output = requests[response_index + 1]["messages"]
        assert len(output) == 1
        assert output[0]["type"] == "function_call_output"
        assert output[0]["call_id"] == responses[response_index].output[0].call_id
        assert isinstance(output[0]["output"], str)
        assert json.loads(output[0]["output"]) == result


@pytest.mark.parametrize("text", [
    "not JSON", "[]", '{"status":"unknown","message":"Hello"}',
    '{"status":"needs_input","message":" "}',
    '{"status":"needs_input","message":42}',
])
def test_malformed_decision(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    response = message_response("needs_input", "Height?")
    response.output_text = text
    install_model(monkeypatch, [response])
    with pytest.raises(RuntimeError, match="Malformed response"):
        solution.solve(debug=False, read_user=lambda _: pytest.fail("Must not ask user"))


def test_premature_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    install_model(monkeypatch, [message_response("completed", "Done")])
    with pytest.raises(RuntimeError, match="Premature completion"):
        solution.solve(debug=False)


@pytest.mark.parametrize("interruption", [EOFError, KeyboardInterrupt])
def test_interrupted_input(monkeypatch: pytest.MonkeyPatch, interruption: Any) -> None:
    requests = install_model(monkeypatch, [message_response("needs_input", "Height?")])

    def read_user(_: str) -> str:
        raise interruption()

    actual = solution.solve(debug=False, read_user=read_user)
    assert actual["status"] == "needs_input"
    assert actual["message"] == "Height?"
    assert actual["steps"] == actual["clarifications"] == []
    assert actual["call_count"] == 0
    assert actual["model_turn_count"] == len(requests) == 1
    assert actual["previous_response_id"] == "resp_1"
    assert actual["result"] is None


def test_model_budget_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(solution, "MAX_MODEL_TURNS", 1)
    requests = install_model(monkeypatch, [message_response("needs_input", "Height?")])
    with pytest.raises(RuntimeError, match="budget exhausted"):
        solution.solve(debug=False, read_user=lambda _: pytest.fail("Budget exhausted"))
    assert len(requests) == 1


@pytest.mark.parametrize("case", ["multiple", "unknown", "arguments", "budget"])
def test_invalid_tool_request(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    response = tool_response("calculate_rectangle_area", {"width": 5, "height": 3})
    if case == "multiple":
        response.output.append(response.output[0])
    elif case == "unknown":
        response.output[0].name = "not_exposed"
    elif case == "arguments":
        response.output[0].arguments = "[]"
    else:
        monkeypatch.setattr(solution, "MAX_TOOL_CALLS", 0)
    install_model(monkeypatch, [response])
    monkeypatch.setitem(TOOL_REGISTRY, "calculate_rectangle_area",
                        lambda **_: pytest.fail("Invalid call must not execute"))
    with pytest.raises(RuntimeError):
        solution.solve(debug=False)


def test_tool_failure_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = install_model(monkeypatch, [
        tool_response("calculate_rectangle_area", {"width": 5, "height": 3}),
    ])
    attempts: list[dict[str, Any]] = []

    def failing_tool(**arguments: Any) -> int:
        attempts.append(arguments)
        raise ValueError("Tool failed")

    monkeypatch.setitem(TOOL_REGISTRY, "calculate_rectangle_area", failing_tool)
    with pytest.raises(ValueError, match="Tool failed"):
        solution.solve(debug=False)
    assert len(requests) == len(attempts) == 1