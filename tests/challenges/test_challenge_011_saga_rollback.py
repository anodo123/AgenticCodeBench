"""Offline evaluator for Challenge 011's booking saga and rollback."""

import json
from copy import deepcopy
from functools import wraps
from types import SimpleNamespace
from typing import Any

import pytest

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.trip_tools import (
    BookingNotFoundError,
    FlightNotFoundError,
    HotelNotFoundError,
    InvalidNightsError,
    InvalidTripActionError,
    RoomUnavailableError,
    create_trip_tool_registry,
)
from agentic_code_bench.solutions import challenge_011_saga_rollback as solution

ACTION_KEYS = {
    "reserve_flight": "reserve-flight-test-011",
    "reserve_hotel": "reserve-hotel-test-011",
    "cancel_flight": "cancel-flight-test-011",
}


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


def approved_responses() -> list[SimpleNamespace]:
    return [
        tool_response("get_flight_quote", {"flight_id": "F-204"}),
        tool_response("get_hotel_quote", {"hotel_id": "H-88", "nights": 2}),
        tool_response(
            "calculate_hotel_stay_total",
            {
                "nightly_price_paise": 400_000,
                "nights": 2,
            },
        ),
        tool_response(
            "calculate_trip_total",
            {
                "flight_price_paise": 1_800_000,
                "hotel_price_paise": 800_000,
            },
        ),
        message_response(
            "needs_confirmation",
            "Flight 18000, hotel 8000, total 26000, budget 30000 rupees. Approve?",
        ),
        tool_response(
            "reserve_flight",
            {
                "flight_id": "F-204",
                "idempotency_key": ACTION_KEYS["reserve_flight"],
            },
        ),
        tool_response(
            "reserve_hotel",
            {
                "hotel_id": "H-88",
                "nights": 2,
                "idempotency_key": ACTION_KEYS["reserve_hotel"],
            },
        ),
        tool_response(
            "cancel_flight",
            {
                "booking_id": "FLIGHT-BOOKING-204",
                "compensation_key": ACTION_KEYS["cancel_flight"],
            },
        ),
        tool_response(
            "lookup_flight_booking",
            {"booking_id": "FLIGHT-BOOKING-204"},
        ),
        message_response(
            "rolled_back",
            "Hotel failed permanently; flight cancellation was verified.",
        ),
    ]


def test_trip_registry_rolls_back_exactly_once() -> None:
    registry = create_trip_tool_registry()
    flight_quote = registry["get_flight_quote"](flight_id="F-204")
    hotel_quote = registry["get_hotel_quote"](hotel_id="H-88", nights=2)
    hotel_total = registry["calculate_hotel_stay_total"](
        nightly_price_paise=hotel_quote["nightly_price_paise"],
        nights=hotel_quote["nights"],
    )
    total = registry["calculate_trip_total"](
        flight_price_paise=flight_quote["flight_price_paise"],
        hotel_price_paise=hotel_total["hotel_price_paise"],
    )
    assert total["total_price_paise"] == 2_600_000
    flight_booking = registry["reserve_flight"](
        flight_id="F-204",
        idempotency_key=ACTION_KEYS["reserve_flight"],
    )
    with pytest.raises(RoomUnavailableError):
        registry["reserve_hotel"](
            hotel_id="H-88",
            nights=2,
            idempotency_key=ACTION_KEYS["reserve_hotel"],
        )
    with pytest.raises(TimeoutError):
        registry["cancel_flight"](
            booking_id=flight_booking["booking_id"],
            compensation_key=ACTION_KEYS["cancel_flight"],
        )
    cancelled = registry["lookup_flight_booking"](
        booking_id=flight_booking["booking_id"]
    )
    duplicate = registry["cancel_flight"](
        booking_id=flight_booking["booking_id"],
        compensation_key=ACTION_KEYS["cancel_flight"],
    )
    assert flight_quote["flight_price_paise"] == 1_800_000
    assert hotel_quote["nightly_price_paise"] == 400_000
    assert hotel_total["hotel_price_paise"] == 800_000
    assert duplicate == {
        "compensation_key": ACTION_KEYS["cancel_flight"],
        "booking_id": "FLIGHT-BOOKING-204",
        "status": "cancelled",
    }
    assert cancelled == {
        "booking_id": "FLIGHT-BOOKING-204",
        "flight_id": "F-204",
        "status": "cancelled",
    }


def test_hotel_quote_rejects_unknown_hotel() -> None:
    registry = create_trip_tool_registry()
    with pytest.raises(HotelNotFoundError):
        registry["get_hotel_quote"](hotel_id="UNKNOWN", nights=2)


@pytest.mark.parametrize("nights", [0, -1, 2.5, "2", True])
def test_hotel_quote_rejects_invalid_nights(nights: Any) -> None:
    registry = create_trip_tool_registry()
    with pytest.raises(InvalidNightsError):
        registry["get_hotel_quote"](hotel_id="H-88", nights=nights)


def test_flight_reservation_replays_the_original_result() -> None:
    registry = create_trip_tool_registry()
    arguments = {
        "flight_id": "F-204",
        "idempotency_key": ACTION_KEYS["reserve_flight"],
    }
    first = registry["reserve_flight"](**arguments)
    replay = registry["reserve_flight"](**arguments)
    assert first == replay == {
        "booking_id": "FLIGHT-BOOKING-204",
        "flight_id": "F-204",
        "status": "booked",
        "idempotency_key": ACTION_KEYS["reserve_flight"],
    }


@pytest.mark.parametrize("idempotency_key", [None, "", "   "])
def test_flight_reservation_requires_an_idempotency_key(
    idempotency_key: Any,
) -> None:
    registry = create_trip_tool_registry()
    with pytest.raises(InvalidTripActionError, match="non-empty"):
        registry["reserve_flight"](
            flight_id="F-204",
            idempotency_key=idempotency_key,
        )


def test_flight_reservation_rejects_an_unknown_flight() -> None:
    registry = create_trip_tool_registry()
    with pytest.raises(FlightNotFoundError):
        registry["reserve_flight"](
            flight_id="UNKNOWN",
            idempotency_key=ACTION_KEYS["reserve_flight"],
        )


def test_flight_reservation_rejects_idempotency_conflict() -> None:
    registry = create_trip_tool_registry()
    registry["reserve_flight"](
        flight_id="F-204",
        idempotency_key=ACTION_KEYS["reserve_flight"],
    )
    with pytest.raises(InvalidTripActionError, match="cannot reserve flight"):
        registry["reserve_flight"](
            flight_id="F-999",
            idempotency_key=ACTION_KEYS["reserve_flight"],
        )


def test_hotel_reservation_stores_and_replays_permanent_failure() -> None:
    registry = create_trip_tool_registry()
    arguments = {
        "hotel_id": "H-88",
        "nights": 2,
        "idempotency_key": ACTION_KEYS["reserve_hotel"],
    }
    with pytest.raises(RoomUnavailableError) as first:
        registry["reserve_hotel"](**arguments)
    with pytest.raises(RoomUnavailableError) as replay:
        registry["reserve_hotel"](**arguments)
    assert str(replay.value) == str(first.value)


@pytest.mark.parametrize("idempotency_key", [None, "", "   "])
def test_hotel_reservation_requires_an_idempotency_key(
    idempotency_key: Any,
) -> None:
    registry = create_trip_tool_registry()
    with pytest.raises(InvalidTripActionError, match="non-empty"):
        registry["reserve_hotel"](
            hotel_id="H-88",
            nights=2,
            idempotency_key=idempotency_key,
        )


@pytest.mark.parametrize("hotel_id", ["", "UNKNOWN"])
def test_hotel_reservation_rejects_an_unknown_hotel(hotel_id: str) -> None:
    registry = create_trip_tool_registry()
    with pytest.raises(HotelNotFoundError):
        registry["reserve_hotel"](
            hotel_id=hotel_id,
            nights=2,
            idempotency_key=ACTION_KEYS["reserve_hotel"],
        )


@pytest.mark.parametrize("nights", [0, -1, 2.5, "2", True])
def test_hotel_reservation_rejects_invalid_nights(nights: Any) -> None:
    registry = create_trip_tool_registry()
    with pytest.raises(InvalidNightsError):
        registry["reserve_hotel"](
            hotel_id="H-88",
            nights=nights,
            idempotency_key=ACTION_KEYS["reserve_hotel"],
        )


@pytest.mark.parametrize(
    ("hotel_id", "nights"),
    [("ANOTHER-HOTEL", 2), ("H-88", 3)],
)
def test_hotel_reservation_rejects_idempotency_conflict(
    hotel_id: str,
    nights: int,
) -> None:
    registry = create_trip_tool_registry()
    registry_call = registry["reserve_hotel"]
    with pytest.raises(RoomUnavailableError):
        registry_call(
            hotel_id="H-88",
            nights=2,
            idempotency_key=ACTION_KEYS["reserve_hotel"],
        )
    with pytest.raises(InvalidTripActionError, match="cannot reserve hotel"):
        registry_call(
            hotel_id=hotel_id,
            nights=nights,
            idempotency_key=ACTION_KEYS["reserve_hotel"],
        )


def test_flight_cancellation_commits_then_replays_stored_result() -> None:
    registry = create_trip_tool_registry()
    booking = registry["reserve_flight"](
        flight_id="F-204",
        idempotency_key=ACTION_KEYS["reserve_flight"],
    )
    arguments = {
        "booking_id": booking["booking_id"],
        "compensation_key": ACTION_KEYS["cancel_flight"],
    }
    with pytest.raises(TimeoutError):
        registry["cancel_flight"](**arguments)
    assert registry["cancel_flight"](**arguments) == {
        "compensation_key": ACTION_KEYS["cancel_flight"],
        "booking_id": "FLIGHT-BOOKING-204",
        "status": "cancelled",
    }


@pytest.mark.parametrize("compensation_key", [None, "", "   "])
def test_flight_cancellation_requires_a_compensation_key(
    compensation_key: Any,
) -> None:
    registry = create_trip_tool_registry()
    with pytest.raises(InvalidTripActionError, match="non-empty"):
        registry["cancel_flight"](
            booking_id="FLIGHT-BOOKING-204",
            compensation_key=compensation_key,
        )


@pytest.mark.parametrize("booking_id", ["", "UNKNOWN"])
def test_flight_cancellation_rejects_an_unknown_booking(
    booking_id: str,
) -> None:
    registry = create_trip_tool_registry()
    with pytest.raises(BookingNotFoundError):
        registry["cancel_flight"](
            booking_id=booking_id,
            compensation_key=ACTION_KEYS["cancel_flight"],
        )


def test_flight_cancellation_rejects_compensation_key_conflict() -> None:
    registry = create_trip_tool_registry()
    booking = registry["reserve_flight"](
        flight_id="F-204",
        idempotency_key=ACTION_KEYS["reserve_flight"],
    )
    with pytest.raises(TimeoutError):
        registry["cancel_flight"](
            booking_id=booking["booking_id"],
            compensation_key=ACTION_KEYS["cancel_flight"],
        )
    with pytest.raises(InvalidTripActionError, match="cannot cancel booking"):
        registry["cancel_flight"](
            booking_id="ANOTHER-BOOKING",
            compensation_key=ACTION_KEYS["cancel_flight"],
        )


def test_cancelled_booking_rejects_a_new_compensation_key() -> None:
    registry = create_trip_tool_registry()
    booking = registry["reserve_flight"](
        flight_id="F-204",
        idempotency_key=ACTION_KEYS["reserve_flight"],
    )
    with pytest.raises(TimeoutError):
        registry["cancel_flight"](
            booking_id=booking["booking_id"],
            compensation_key=ACTION_KEYS["cancel_flight"],
        )
    with pytest.raises(InvalidTripActionError, match="already cancelled"):
        registry["cancel_flight"](
            booking_id=booking["booking_id"],
            compensation_key="another-cancel-key",
        )


def test_flight_lookup_returns_authoritative_booking_state() -> None:
    registry = create_trip_tool_registry()
    booking = registry["reserve_flight"](
        flight_id="F-204",
        idempotency_key=ACTION_KEYS["reserve_flight"],
    )
    assert registry["lookup_flight_booking"](
        booking_id=booking["booking_id"]
    ) == {
        "booking_id": "FLIGHT-BOOKING-204",
        "flight_id": "F-204",
        "status": "booked",
    }
    with pytest.raises(TimeoutError):
        registry["cancel_flight"](
            booking_id=booking["booking_id"],
            compensation_key=ACTION_KEYS["cancel_flight"],
        )
    assert registry["lookup_flight_booking"](
        booking_id=booking["booking_id"]
    ) == {
        "booking_id": "FLIGHT-BOOKING-204",
        "flight_id": "F-204",
        "status": "cancelled",
    }


@pytest.mark.parametrize("booking_id", ["", "UNKNOWN"])
def test_flight_lookup_rejects_an_unknown_booking(booking_id: str) -> None:
    registry = create_trip_tool_registry()
    with pytest.raises(BookingNotFoundError):
        registry["lookup_flight_booking"](booking_id=booking_id)


def test_challenge_011_verified_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    challenge = load_challenge(solution.CHALLENGE_PATH)
    responses = approved_responses()
    requests: list[dict[str, Any]] = []
    install_model(monkeypatch, responses, requests)
    executions: list[dict[str, Any]] = []
    registry = record_registry(create_trip_tool_registry(), executions)
    monkeypatch.setattr(solution, "create_trip_tool_registry", lambda: registry)
    monkeypatch.setattr(
        solution,
        "create_action_key",
        lambda action: ACTION_KEYS[action],
    )
    learner_challenge = deepcopy(challenge)
    learner_challenge.pop("expected")
    monkeypatch.setattr(solution, "load_challenge", lambda _: learner_challenge)
    user_prompts: list[str] = []

    def approve(prompt: str) -> str:
        user_prompts.append(prompt)
        assert len(executions) == 4
        return "Yes, book the trip."

    actual = solution.solve(debug=False, read_user=approve)
    expected = challenge["expected"]
    assert actual["status"] == expected["status"]
    assert actual["approved"] is True
    assert actual["steps"] == executions
    assert actual["quote"] == {
        "flight_price_paise": expected["flight_price_paise"],
        "nightly_price_paise": expected["nightly_price_paise"],
        "nights": expected["nights"],
        "hotel_price_paise": expected["hotel_price_paise"],
        "total_price_paise": expected["total_price_paise"],
        "budget_paise": expected["budget_paise"],
    }
    assert actual["result"] == {
        "flight_booking_id": "FLIGHT-BOOKING-204",
        "flight_status": expected["flight_status"],
        "hotel_status": expected["hotel_status"],
        "trip_status": expected["trip_status"],
    }
    assert actual["call_count"] == len(executions) == expected["call_count"]
    assert actual["forward_action_count"] == expected["forward_action_count"]
    assert actual["compensation_count"] == expected["compensation_count"]
    assert actual["model_turn_count"] == len(requests) == 10
    assert actual["action_keys"] == ACTION_KEYS
    assert [step["tool"] for step in executions] == expected["ordered_tools"]
    assert [step["status"] for step in executions] == [
        "success",
        "success",
        "success",
        "success",
        "success",
        "error",
        "error",
        "success",
    ]
    assert executions[5]["error_type"] == "RoomUnavailableError"
    assert executions[6]["error_type"] == "TimeoutError"
    assert len(user_prompts) == 1
    initial_messages = [request["messages"] for request in requests[:5]]
    assert not any(key in json.dumps(initial_messages) for key in ACTION_KEYS.values())
    approved_input = json.dumps(requests[5]["messages"])
    assert all(key in approved_input for key in ACTION_KEYS.values())
    assert requests[0]["messages"] == challenge["messages"]
    assert requests[0]["previous_response_id"] is None
    for index, request in enumerate(requests[1:], start=1):
        assert request["previous_response_id"] == responses[index - 1].id
    handoffs = [(0, 0), (1, 1), (2, 2), (3, 3), (5, 4), (8, 7)]
    for response_index, execution_index in handoffs:
        output = requests[response_index + 1]["messages"]
        assert len(output) == 1
        assert output[0]["type"] == "function_call_output"
        assert output[0]["call_id"] == responses[response_index].output[0].call_id
        assert json.loads(output[0]["output"]) == executions[execution_index]["result"]
    hotel_error = json.loads(requests[7]["messages"][0]["output"])
    assert hotel_error["status"] == "error"
    assert hotel_error["retry_allowed"] is False
    assert hotel_error["compensation_required"] is True
    cancellation_error = json.loads(requests[8]["messages"][0]["output"])
    assert cancellation_error["status"] == "unknown"
    assert cancellation_error["retry_allowed"] is False
    assert cancellation_error["reconciliation_required"] is True


def test_challenge_011_declined_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    challenge = load_challenge(solution.CHALLENGE_PATH)
    responses = approved_responses()[:5] + [
        message_response("cancelled", "Trip booking cancelled before reservation.")
    ]
    requests: list[dict[str, Any]] = []
    install_model(monkeypatch, responses, requests)
    executions: list[dict[str, Any]] = []
    registry = record_registry(create_trip_tool_registry(), executions)
    monkeypatch.setattr(solution, "create_trip_tool_registry", lambda: registry)
    monkeypatch.setattr(
        solution,
        "create_action_key",
        lambda _: pytest.fail("A declined plan needs no action keys"),
    )
    learner_challenge = deepcopy(challenge)
    learner_challenge.pop("expected")
    monkeypatch.setattr(solution, "load_challenge", lambda _: learner_challenge)
    actual = solution.solve(
        debug=False,
        read_user=lambda _: "No, do not book it.",
    )
    assert actual["status"] == "cancelled"
    assert actual["approved"] is False
    assert actual["result"] is None
    assert actual["call_count"] == len(executions) == 4
    assert actual["forward_action_count"] == 0
    assert actual["compensation_count"] == 0
    assert actual["model_turn_count"] == len(requests) == 6
    assert actual["action_keys"] == {}
    assert [step["tool"] for step in executions] == [
        "get_flight_quote",
        "get_hotel_quote",
        "calculate_hotel_stay_total",
        "calculate_trip_total",
    ]
