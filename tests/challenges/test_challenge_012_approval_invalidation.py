"""Offline evaluator for Challenge 012's approval invalidation flow."""

import json
from copy import deepcopy
from functools import wraps
from types import SimpleNamespace
from typing import Any

import pytest

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.ticket_tools import (
    EventNotFoundError,
    InvalidQuantityError,
    InvalidTicketActionError,
    PurchaseNotFoundError,
    QuoteExpiredError,
    QuoteNotFoundError,
    create_ticket_tool_registry,
)
from agentic_code_bench.solutions import (
    challenge_012_approval_invalidation as solution,
)

PURCHASE_KEYS = {
    "QUOTE-001": "purchase-quote-001-test-012",
    "QUOTE-002": "purchase-quote-002-test-012",
}


def message_response(status: str, message: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="",
        status="completed",
        output=[SimpleNamespace(type="reasoning"), SimpleNamespace(type="message")],
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


def approved_responses() -> list[SimpleNamespace]:
    return [
        tool_response("get_ticket_quote", {"event_id": "EVT-901"}),
        tool_response(
            "calculate_ticket_total",
            {"unit_price_paise": 500_000, "quantity": 3},
        ),
        message_response(
            "needs_confirmation",
            "Approve 3 EVT-901 tickets under QUOTE-001 for 15900 rupees?",
        ),
        tool_response(
            "purchase_tickets",
            {
                "quote_id": "QUOTE-001",
                "quantity": 3,
                "idempotency_key": PURCHASE_KEYS["QUOTE-001"],
            },
        ),
        tool_response("get_ticket_quote", {"event_id": "EVT-901"}),
        tool_response(
            "calculate_ticket_total",
            {"unit_price_paise": 550_000, "quantity": 3},
        ),
        message_response(
            "needs_confirmation",
            "Approve revised QUOTE-002 for 3 tickets and 17400 rupees?",
        ),
        tool_response(
            "purchase_tickets",
            {
                "quote_id": "QUOTE-002",
                "quantity": 3,
                "idempotency_key": PURCHASE_KEYS["QUOTE-002"],
            },
        ),
        tool_response(
            "lookup_purchase_by_idempotency_key",
            {"idempotency_key": PURCHASE_KEYS["QUOTE-002"]},
        ),
        message_response(
            "completed",
            "Verified 3 tickets under the newly approved quote.",
        ),
    ]


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


def advance_to_revised_quote(registry: dict[str, Any]) -> None:
    with pytest.raises(QuoteExpiredError):
        registry["purchase_tickets"](
            quote_id="QUOTE-001",
            quantity=3,
            idempotency_key=PURCHASE_KEYS["QUOTE-001"],
        )


def test_quote_is_stable_until_the_provider_changes_it() -> None:
    registry = create_ticket_tool_registry()
    first = registry["get_ticket_quote"](event_id="EVT-901")
    repeated = registry["get_ticket_quote"](event_id="EVT-901")
    assert first == repeated == {
        "event_id": "EVT-901",
        "quote_id": "QUOTE-001",
        "unit_price_paise": 500_000,
    }
    advance_to_revised_quote(registry)
    assert registry["get_ticket_quote"](event_id="EVT-901") == {
        "event_id": "EVT-901",
        "quote_id": "QUOTE-002",
        "unit_price_paise": 550_000,
    }


@pytest.mark.parametrize("event_id", ["", "UNKNOWN"])
def test_quote_rejects_an_unknown_event(event_id: str) -> None:
    registry = create_ticket_tool_registry()
    with pytest.raises(EventNotFoundError):
        registry["get_ticket_quote"](event_id=event_id)


def test_ticket_total_calculates_fees() -> None:
    registry = create_ticket_tool_registry()
    assert registry["calculate_ticket_total"](
        unit_price_paise=550_000,
        quantity=3,
    ) == {
        "unit_price_paise": 550_000,
        "quantity": 3,
        "subtotal_paise": 1_650_000,
        "fee_paise": 90_000,
        "total_price_paise": 1_740_000,
    }


@pytest.mark.parametrize("quantity", [0, -1, 2.5, "3", True])
def test_ticket_total_rejects_invalid_quantity(quantity: Any) -> None:
    registry = create_ticket_tool_registry()
    with pytest.raises(InvalidQuantityError):
        registry["calculate_ticket_total"](
            unit_price_paise=550_000,
            quantity=quantity,
        )


def test_expired_quote_failure_is_idempotent_and_immutable() -> None:
    registry = create_ticket_tool_registry()
    arguments = {
        "quote_id": "QUOTE-001",
        "quantity": 3,
        "idempotency_key": PURCHASE_KEYS["QUOTE-001"],
    }
    with pytest.raises(QuoteExpiredError) as first:
        registry["purchase_tickets"](**arguments)
    with pytest.raises(QuoteExpiredError) as replay:
        registry["purchase_tickets"](**arguments)
    assert str(replay.value) == str(first.value)
    with pytest.raises(InvalidTicketActionError):
        registry["purchase_tickets"](
            quote_id="QUOTE-002",
            quantity=3,
            idempotency_key=PURCHASE_KEYS["QUOTE-001"],
        )


def test_current_quote_commits_then_times_out_and_reconciles() -> None:
    registry = create_ticket_tool_registry()
    advance_to_revised_quote(registry)
    arguments = {
        "quote_id": "QUOTE-002",
        "quantity": 3,
        "idempotency_key": PURCHASE_KEYS["QUOTE-002"],
    }
    with pytest.raises(TimeoutError):
        registry["purchase_tickets"](**arguments)
    result = registry["lookup_purchase_by_idempotency_key"](
        idempotency_key=PURCHASE_KEYS["QUOTE-002"]
    )
    assert registry["purchase_tickets"](**arguments) == result
    assert result == {
        "purchase_id": "PURCHASE-001",
        "event_id": "EVT-901",
        "quote_id": "QUOTE-002",
        "quantity": 3,
        "unit_price_paise": 550_000,
        "subtotal_paise": 1_650_000,
        "fee_paise": 90_000,
        "total_price_paise": 1_740_000,
        "status": "completed",
        "idempotency_key": PURCHASE_KEYS["QUOTE-002"],
    }


@pytest.mark.parametrize("idempotency_key", [None, "", "   "])
def test_purchase_requires_an_idempotency_key(idempotency_key: Any) -> None:
    registry = create_ticket_tool_registry()
    with pytest.raises(InvalidTicketActionError):
        registry["purchase_tickets"](
            quote_id="QUOTE-001",
            quantity=3,
            idempotency_key=idempotency_key,
        )


def test_purchase_rejects_an_unknown_quote() -> None:
    registry = create_ticket_tool_registry()
    with pytest.raises(QuoteNotFoundError):
        registry["purchase_tickets"](
            quote_id="UNKNOWN",
            quantity=3,
            idempotency_key="unknown-quote-key",
        )


def test_lookup_does_not_claim_an_unverified_purchase() -> None:
    registry = create_ticket_tool_registry()
    with pytest.raises(PurchaseNotFoundError):
        registry["lookup_purchase_by_idempotency_key"](
            idempotency_key="unknown-key"
        )
    advance_to_revised_quote(registry)
    with pytest.raises(PurchaseNotFoundError):
        registry["lookup_purchase_by_idempotency_key"](
            idempotency_key=PURCHASE_KEYS["QUOTE-001"]
        )


def test_challenge_012_replans_and_verifies_purchase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    challenge = load_challenge(solution.CHALLENGE_PATH)
    responses = approved_responses()
    requests: list[dict[str, Any]] = []
    install_model(monkeypatch, responses, requests)
    executions: list[dict[str, Any]] = []
    registry = record_registry(create_ticket_tool_registry(), executions)
    monkeypatch.setattr(solution, "create_ticket_tool_registry", lambda: registry)
    monkeypatch.setattr(
        solution,
        "create_purchase_key",
        lambda quote_id: PURCHASE_KEYS[quote_id],
    )
    learner_challenge = deepcopy(challenge)
    learner_challenge.pop("expected")
    monkeypatch.setattr(solution, "load_challenge", lambda _: learner_challenge)
    prompts: list[str] = []

    def approve(prompt: str) -> str:
        prompts.append(prompt)
        assert len(executions) in (2, 5)
        return "Yes, approve this exact plan."

    actual = solution.solve(debug=False, read_user=approve)
    expected = challenge["expected"]
    assert actual["status"] == expected["status"]
    assert actual["approval_count"] == expected["approval_count"]
    assert actual["steps"] == executions
    assert actual["initial_plan"] == {
        "event_id": expected["event_id"],
        "quote_id": expected["initial_quote_id"],
        "quantity": expected["quantity"],
        "unit_price_paise": expected["initial_unit_price_paise"],
        "subtotal_paise": expected["initial_subtotal_paise"],
        "fee_paise": expected["initial_fee_paise"],
        "total_price_paise": expected["initial_total_price_paise"],
        "budget_paise": expected["budget_paise"],
    }
    assert actual["revised_plan"] == {
        "event_id": expected["event_id"],
        "quote_id": expected["revised_quote_id"],
        "quantity": expected["quantity"],
        "unit_price_paise": expected["revised_unit_price_paise"],
        "subtotal_paise": expected["revised_subtotal_paise"],
        "fee_paise": expected["revised_fee_paise"],
        "total_price_paise": expected["revised_total_price_paise"],
        "budget_paise": expected["budget_paise"],
    }
    assert actual["result"]["status"] == expected["purchase_status"]
    assert actual["result"]["quote_id"] == expected["revised_quote_id"]
    assert actual["result"]["quantity"] == expected["quantity"]
    assert actual["result"]["total_price_paise"] == expected[
        "revised_total_price_paise"
    ]
    assert actual["call_count"] == len(executions) == expected["call_count"]
    assert actual["purchase_attempt_count"] == expected[
        "purchase_attempt_count"
    ]
    assert actual["committed_purchase_count"] == expected[
        "committed_purchase_count"
    ]
    assert actual["model_turn_count"] == len(requests) == 10
    assert actual["idempotency_keys"] == PURCHASE_KEYS
    assert len(prompts) == 2
    assert [step["tool"] for step in executions] == expected["ordered_tools"]
    assert [step["status"] for step in executions] == [
        "success",
        "success",
        "error",
        "success",
        "success",
        "error",
        "success",
    ]
    assert executions[2]["error_type"] == "QuoteExpiredError"
    assert executions[5]["error_type"] == "TimeoutError"
    assert PURCHASE_KEYS["QUOTE-001"] in json.dumps(requests[3]["messages"])
    assert PURCHASE_KEYS["QUOTE-002"] not in json.dumps(requests[3]["messages"])
    assert PURCHASE_KEYS["QUOTE-002"] in json.dumps(requests[7]["messages"])
    expired_payload = json.loads(requests[4]["messages"][0]["output"])
    assert expired_payload["retry_allowed"] is False
    assert expired_payload["approval_invalidated"] is True
    timeout_payload = json.loads(requests[8]["messages"][0]["output"])
    assert timeout_payload["retry_allowed"] is False
    assert timeout_payload["reconciliation_required"] is True
