"""Offline evaluator for Challenge 009.

Question: Recover from an order-service timeout, then query the carrier using
its real tracking ID; stop truthfully when an error is permanent or exhausted.
Learner orchestration tests intentionally fail until solve() is implemented.
"""

import json
from copy import deepcopy
from functools import wraps
from types import SimpleNamespace
from typing import Any

import pytest

from agentic_code_bench.core.order_tools import (
    OrderNotFoundError,
    create_order_tool_registry,
)
from agentic_code_bench.solutions import challenge_009_tool_failure_recovery as solution


def test_simulated_tools_reset_and_recover() -> None:
    for _ in range(2):
        registry = create_order_tool_registry()
        with pytest.raises(OrderNotFoundError):
            registry["get_order_status"](order_id="UNKNOWN")
        with pytest.raises(TimeoutError):
            registry["get_order_status"](order_id="ORD-1042")
        order = registry["get_order_status"](order_id="ORD-1042")
        carrier = registry["get_carrier_status"](tracking_id=order["tracking_id"])
        assert carrier["delivery_status"] == "out_for_delivery"


@pytest.mark.parametrize("scenario", ["recover", "permanent", "exhausted"])
def test_challenge_009_tool_failure_recovery(
    monkeypatch: pytest.MonkeyPatch, scenario: str,
) -> None:
    requests: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    registry = create_order_tool_registry()
    if scenario != "recover":
        def fail_order(**_: Any) -> Any:
            if scenario == "permanent":
                raise OrderNotFoundError("Order was not found")
            raise TimeoutError("Order service timed out temporarily")
        registry["get_order_status"] = fail_order

    def record(name: str, function: Any) -> Any:
        @wraps(function)
        def wrapped(**arguments: Any) -> Any:
            evidence = {"tool": name, "arguments": arguments}
            try:
                result = function(**arguments)
            except Exception as error:
                evidence.update(status="error", error_type=type(error).__name__,
                                error_message=str(error), retryable=isinstance(error, TimeoutError))
                executions.append(evidence)
                raise
            evidence.update(status="success", result=result)
            executions.append(evidence)
            return result
        return wrapped

    registry = {name: record(name, function) for name, function in registry.items()}
    monkeypatch.setattr(solution, "create_order_tool_registry", lambda: registry)

    def fake_model(**kwargs: Any) -> SimpleNamespace:
        turn = len(requests)
        if turn:
            assert kwargs["previous_response_id"] == requests[-1]["response"].id
            output = kwargs["messages"]
            assert len(output) == 1
            assert output[0]["type"] == "function_call_output"
            assert output[0]["call_id"] == f"call_{turn}"
            payload = json.loads(output[0]["output"])
            evidence = executions[-1]
            assert payload["status"] == evidence["status"]
            if evidence["status"] == "success":
                assert payload["result"] == evidence["result"]
            else:
                assert payload["error_type"] == evidence["error_type"]
                assert payload["error_message"] == evidence["error_message"]
                assert payload["retryable"] is (scenario != "permanent")
                assert payload["retry_allowed"] is (scenario != "permanent" and turn == 1)
        else:
            assert kwargs.get("previous_response_id") is None
            assert "ORD-1042" in kwargs["messages"][0]["content"]
        assert kwargs["tools"]
        final_turn = {"recover": 3, "permanent": 1, "exhausted": 2}[scenario]
        assert turn <= final_turn, "Unexpected extra request/retry"
        if turn == final_turn:
            status = "completed" if scenario == "recover" else "failed"
            text = "Delivery: out_for_delivery" if status == "completed" else executions[-1]["error_message"]
            response = SimpleNamespace(
                id=f"resp_{turn + 1}", status="completed",
                output=[SimpleNamespace(type="reasoning"), SimpleNamespace(type="message")],
                output_text=json.dumps({"status": status, "message": text}),
            )
        else:
            name = "get_carrier_status" if turn == 2 else "get_order_status"
            arguments = {"tracking_id": "TRK-1042"} if turn == 2 else {"order_id": "ORD-1042"}
            response = SimpleNamespace(
                id=f"resp_{turn + 1}", status="completed", output_text="",
                output=[SimpleNamespace(type="reasoning"), SimpleNamespace(
                    type="function_call", name=name, arguments=json.dumps(arguments),
                    call_id=f"call_{turn + 1}",
                )],
            )
        requests.append({**deepcopy(kwargs), "response": response})
        return response

    monkeypatch.setattr(solution, "call_openai_model", fake_model)
    original_load = solution.load_challenge

    def without_answer_key(path: Any) -> Any:
        challenge = original_load(path)
        challenge.pop("expected")
        return challenge

    monkeypatch.setattr(solution, "load_challenge", without_answer_key)
    actual = solution.solve(debug=False)
    assert actual["steps"] == executions
    assert actual["call_count"] == len(executions)
    assert actual["model_turn_count"] == len(requests)
    assert type(actual["call_count"]) is type(actual["model_turn_count"]) is int
    assert actual["message"] == json.loads(requests[-1]["response"].output_text)["message"]
    assert actual["status"] == ("completed" if scenario == "recover" else "failed")
    if scenario == "recover":
        assert actual["result"] == executions[-1]["result"]
        assert actual["result"]["delivery_status"] == "out_for_delivery"
        assert [step["status"] for step in executions] == ["error", "success", "success"]
    else:
        assert actual["result"] is None
        assert all(step["tool"] == "get_order_status" for step in executions)