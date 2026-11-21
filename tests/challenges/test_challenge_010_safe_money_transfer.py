"""Offline evaluator for Challenge 010's approval and reconciliation workflow."""

import json
from copy import deepcopy
from functools import wraps
from types import SimpleNamespace
from typing import Any

import pytest

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.wallet_tools import create_wallet_tool_registry
from agentic_code_bench.solutions import challenge_010_safe_money_transfer as solution

TEST_IDEMPOTENCY_KEY = "transfer-test-run-010"


def message_response(status: str, message: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="",
        status="completed",
        output=[
            SimpleNamespace(type="reasoning"),
            SimpleNamespace(type="message"),
        ],
        output_text=json.dumps({"status": status, "message": message}),
    )


def tool_response(name: str, arguments: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        id="",
        status="completed",
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                name=name,
                arguments=json.dumps(arguments),
                call_id=f"call_{name}",
            )
        ],
    )


def install_model(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[SimpleNamespace],
    requests: list[dict[str, Any]],
) -> None:
    for index, response in enumerate(responses, start=1):
        response.id = f"resp_{index}"

    def fake_model(**kwargs: Any) -> SimpleNamespace:
        assert len(requests) < len(responses), "Unexpected extra model request"
        response = responses[len(requests)]
        requests.append({**deepcopy(kwargs), "response": response})
        return response

    monkeypatch.setattr(solution, "call_openai_model", fake_model)


def record_registry(
    registry: dict[str, Any],
    executions: list[dict[str, Any]],
) -> dict[str, Any]:
    recorded: dict[str, Any] = {}
    for name, function in registry.items():
        def make_wrapper(tool_name: str, tool_function: Any) -> Any:
            @wraps(tool_function)
            def wrapper(**arguments: Any) -> Any:
                evidence = {"tool": tool_name, "arguments": arguments}
                try:
                    result = tool_function(**arguments)
                except Exception as error:
                    evidence.update(
                        status="error",
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                    executions.append(evidence)
                    raise
                evidence.update(status="success", result=result)
                executions.append(evidence)
                return result

            return wrapper

        recorded[name] = make_wrapper(name, function)
    return recorded


def test_wallet_registry_commits_once_and_reconciles() -> None:
    registry = create_wallet_tool_registry()
    balance = registry["get_wallet_balance"](wallet_id="WALLET-A")
    preview = registry["create_transfer_preview"](
        source_wallet_id="WALLET-A",
        destination_wallet_id="WALLET-B",
        transfer_amount_paise=250_000,
    )
    assert balance == {"wallet_id": "WALLET-A", "balance_paise": 500_000}
    assert preview["transfer_amount_paise"] == 250_000
    assert preview["service_fee_paise"] == 2_500
    assert preview["source_total_debit_paise"] == 252_500
    assert preview["destination_credit_paise"] == 250_000
    assert type(preview["transfer_amount_paise"]) is int
    assert type(preview["service_fee_paise"]) is int
    assert type(preview["source_total_debit_paise"]) is int
    assert type(preview["destination_credit_paise"]) is int
    with pytest.raises(TimeoutError):
        registry["execute_transfer"](
            preview_id=preview["preview_id"],
            idempotency_key=TEST_IDEMPOTENCY_KEY,
        )
    transfer = registry["lookup_transfer_by_idempotency_key"](
        idempotency_key=TEST_IDEMPOTENCY_KEY
    )
    duplicate = registry["execute_transfer"](
        preview_id=preview["preview_id"],
        idempotency_key=TEST_IDEMPOTENCY_KEY,
    )
    assert duplicate == transfer
    assert transfer["status"] == "completed"
    assert transfer["source_balance_paise"] == 247_500
    assert transfer["destination_balance_paise"] == 350_000


def test_challenge_010_approved_transfer(monkeypatch: pytest.MonkeyPatch) -> None:
    challenge = load_challenge(solution.CHALLENGE_PATH)
    preview_id = "PREVIEW-1042"
    confirmation = (
        "Transfer 2500 rupees with a 25 rupee fee? "
        "The total debit is 2525 rupees."
    )
    responses = [
        tool_response("get_wallet_balance", {"wallet_id": "WALLET-A"}),
        tool_response(
            "create_transfer_preview",
            {
                "source_wallet_id": "WALLET-A",
                "destination_wallet_id": "WALLET-B",
                "transfer_amount_paise": 250_000,
            },
        ),
        message_response("needs_confirmation", confirmation),
        tool_response(
            "execute_transfer",
            {
                "preview_id": preview_id,
                "idempotency_key": TEST_IDEMPOTENCY_KEY,
            },
        ),
        tool_response(
            "lookup_transfer_by_idempotency_key",
            {"idempotency_key": TEST_IDEMPOTENCY_KEY},
        ),
        message_response("completed", "Transfer completed and verified."),
    ]
    requests: list[dict[str, Any]] = []
    install_model(monkeypatch, responses, requests)
    executions: list[dict[str, Any]] = []
    registry = record_registry(create_wallet_tool_registry(), executions)
    monkeypatch.setattr(solution, "create_wallet_tool_registry", lambda: registry)
    monkeypatch.setattr(
        solution,
        "create_idempotency_key",
        lambda: TEST_IDEMPOTENCY_KEY,
    )
    learner_challenge = deepcopy(challenge)
    learner_challenge.pop("expected")
    monkeypatch.setattr(solution, "load_challenge", lambda _: learner_challenge)
    user_prompts: list[str] = []

    def approve(prompt: str) -> str:
        user_prompts.append(prompt)
        assert len(executions) == 2
        return "Yes, proceed."

    actual = solution.solve(debug=False, read_user=approve)
    expected = challenge["expected"]
    assert actual["status"] == expected["status"]
    assert actual["approved"] is True
    assert actual["message"] == "Transfer completed and verified."
    assert actual["steps"] == executions
    assert actual["result"] == executions[-1]["result"]
    assert actual["result"]["status"] == expected["transfer_status"]
    assert actual["call_count"] == len(executions) == expected["call_count"]
    assert actual["side_effect_count"] == expected["side_effect_count"] == 1
    assert actual["model_turn_count"] == len(requests)
    assert actual["idempotency_key"] == TEST_IDEMPOTENCY_KEY
    assert [step["tool"] for step in executions] == expected["ordered_tools"]
    assert [step["status"] for step in executions] == [
        "success",
        "success",
        "error",
        "success",
    ]
    assert executions[2]["error_type"] == "TimeoutError"
    assert len(user_prompts) == 1
    assert requests[0]["messages"] == challenge["messages"]
    assert requests[0]["previous_response_id"] is None
    initial_messages = [request["messages"] for request in requests[:3]]
    assert TEST_IDEMPOTENCY_KEY not in json.dumps(initial_messages)
    assert TEST_IDEMPOTENCY_KEY in json.dumps(requests[3]["messages"])
    for index, request in enumerate(requests[1:], start=1):
        assert request["previous_response_id"] == responses[index - 1].id
    for response_index, execution_index in [(0, 0), (1, 1), (4, 3)]:
        output = requests[response_index + 1]["messages"]
        assert len(output) == 1
        assert output[0]["type"] == "function_call_output"
        assert output[0]["call_id"] == responses[response_index].output[0].call_id
        assert json.loads(output[0]["output"]) == executions[execution_index]["result"]
    timeout_output = requests[4]["messages"]
    assert len(timeout_output) == 1
    assert timeout_output[0]["type"] == "function_call_output"
    assert timeout_output[0]["call_id"] == "call_execute_transfer"
    timeout_payload = json.loads(timeout_output[0]["output"])
    assert timeout_payload["status"] == "unknown"
    assert timeout_payload["retry_allowed"] is False
    assert timeout_payload["reconciliation_required"] is True


def test_challenge_010_declined_transfer(monkeypatch: pytest.MonkeyPatch) -> None:
    challenge = load_challenge(solution.CHALLENGE_PATH)
    responses = [
        tool_response("get_wallet_balance", {"wallet_id": "WALLET-A"}),
        tool_response(
            "create_transfer_preview",
            {
                "source_wallet_id": "WALLET-A",
                "destination_wallet_id": "WALLET-B",
                "transfer_amount_paise": 250_000,
            },
        ),
        message_response("needs_confirmation", "Approve the transfer?"),
        message_response("cancelled", "Transfer cancelled. No money was moved."),
    ]
    requests: list[dict[str, Any]] = []
    install_model(monkeypatch, responses, requests)
    executions: list[dict[str, Any]] = []
    registry = record_registry(create_wallet_tool_registry(), executions)
    monkeypatch.setattr(solution, "create_wallet_tool_registry", lambda: registry)
    monkeypatch.setattr(
        solution,
        "create_idempotency_key",
        lambda: pytest.fail("A declined transfer needs no idempotency key"),
    )
    learner_challenge = deepcopy(challenge)
    learner_challenge.pop("expected")
    monkeypatch.setattr(solution, "load_challenge", lambda _: learner_challenge)
    actual = solution.solve(
        debug=False,
        read_user=lambda _: "No, cancel it.",
    )
    assert actual["status"] == "cancelled"
    assert actual["approved"] is False
    assert actual["result"] is None
    assert actual["call_count"] == len(executions) == 2
    assert actual["side_effect_count"] == 0
    assert actual["model_turn_count"] == len(requests) == 4
    assert actual["idempotency_key"] is None
    assert [step["tool"] for step in executions] == [
        "get_wallet_balance",
        "create_transfer_preview",
    ]
    assert all(step["status"] == "success" for step in executions)
    for response_index, execution_index in [(0, 0), (1, 1)]:
        output = requests[response_index + 1]["messages"]
        assert len(output) == 1
        assert output[0]["type"] == "function_call_output"
        assert output[0]["call_id"] == responses[response_index].output[0].call_id
        assert json.loads(output[0]["output"]) == executions[execution_index]["result"]
