"""Solution for Challenge 011: trip-booking saga rollback.

Question:
Book flight F-204 and hotel H-88 for two nights only when their actual combined
price is at most 30000 rupees. Fetch the flight price and hotel nightly price,
calculate the hotel stay total and combined trip total, and ask for explicit
confirmation. If one reservation succeeds and the next fails, compensate
successful bookings in reverse order. If cancellation times out after
committing, reconcile it and report rolled_back only after verifying that no
partial booking remains.

Learning goals:
Coordinate a saga across multiple services, separate forward and compensating
actions, enforce independent idempotency keys and action budgets, classify
permanent failures, reconcile ambiguous compensation, and distinguish
rolled_back from manual_intervention.

Output contract:
Return status, message, approved, steps, quote, result, call_count,
forward_action_count, compensation_count, model_turn_count, and action_keys.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.model import call_openai_model
from agentic_code_bench.core.trip_tools import (
    TRIP_BUDGET_PAISE,
    BookingNotFoundError,
    RoomUnavailableError,
    create_action_key,
    create_trip_tool_registry,
)

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "challenges"
    / "011_saga_rollback.json"
)
MAX_MODEL_TURNS = 12
MAX_TOOL_CALLS = 8
MAX_FORWARD_ACTIONS = 2
MAX_COMPENSATIONS = 1


def _approval_decision(reply: str) -> bool:
    normalized = reply.strip().casefold()
    affirmative = ("yes", "y", "approve", "approved", "confirm", "proceed")
    negative = ("no", "n", "decline", "declined", "cancel", "cancelled")
    if normalized.startswith(affirmative):
        return True
    if normalized.startswith(negative):
        return False
    raise RuntimeError("User reply must explicitly approve or decline the trip.")


def solve(
    model: str = "gpt-5.6-luna",
    debug: bool = True,
    read_user: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Return the final verified saga report described by the module contract."""
    def log(message: str) -> None:
        if debug:
            print(f"[challenge_011] {message}", flush=True)

    if read_user is None:
        read_user = input

    challenge = load_challenge(CHALLENGE_PATH)
    registry = create_trip_tool_registry()
    allowed_tools = {tool["name"] for tool in challenge["tools"]}
    required_order = [tool["name"] for tool in challenge["tools"]]
    messages = challenge["messages"]
    previous_response_id: str | None = None
    steps: list[dict[str, Any]] = []
    call_count = 0
    forward_action_count = 0
    compensation_count = 0
    model_turn_count = 0
    approval: bool | None = None
    action_keys: dict[str, str] = {}
    quote: dict[str, Any] | None = None
    flight_quote: dict[str, Any] | None = None
    hotel_quote: dict[str, Any] | None = None
    hotel_total: dict[str, Any] | None = None
    trip_total: dict[str, Any] | None = None
    flight_booking: dict[str, Any] | None = None
    hotel_failed = False
    cancellation_ambiguous = False
    verified_booking: dict[str, Any] | None = None
    lookup_failed = False

    for _ in range(MAX_MODEL_TURNS):
        log(
            f"Model turn {model_turn_count + 1}/{MAX_MODEL_TURNS}; "
            f"previous_response_id={previous_response_id!r}; messages={messages!r}"
        )
        response = call_openai_model(
            messages=messages,
            tools=challenge["tools"],
            model=model,
            previous_response_id=previous_response_id,
        )
        model_turn_count += 1
        log(
            f"Response id={response.id!r}; status={response.status!r}; "
            f"types={[item.type for item in response.output]!r}; "
            f"output_text={response.output_text!r}"
        )

        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        message_items = [item for item in response.output if item.type == "message"]
        if len(function_calls) > 1:
            raise RuntimeError("Only one tool call is allowed per model response.")

        if function_calls:
            if call_count >= MAX_TOOL_CALLS:
                raise RuntimeError("Tool-call budget exhausted.")
            call = function_calls[0]
            if call.name not in allowed_tools or call.name not in registry:
                raise RuntimeError(f"Unknown or unavailable tool: {call.name}")
            expected_tool = required_order[call_count]
            if call.name != expected_tool:
                raise RuntimeError(
                    f"Expected {expected_tool}, received {call.name}."
                )
            try:
                arguments = json.loads(call.arguments)
            except (TypeError, ValueError) as error:
                raise RuntimeError("Tool arguments must be valid JSON.") from error
            if not isinstance(arguments, dict):
                raise RuntimeError("Tool arguments must be a JSON object.")

            if call.name == "calculate_hotel_stay_total":
                if hotel_quote is None:
                    raise RuntimeError("A hotel quote is required before calculation.")
                expected_arguments = {
                    "nightly_price_paise": hotel_quote["nightly_price_paise"],
                    "nights": hotel_quote["nights"],
                }
                if arguments != expected_arguments:
                    raise RuntimeError("Hotel total must use the actual hotel quote.")

            if call.name == "calculate_trip_total":
                if flight_quote is None or hotel_total is None:
                    raise RuntimeError("Both quote results are required.")
                expected_arguments = {
                    "flight_price_paise": flight_quote["flight_price_paise"],
                    "hotel_price_paise": hotel_total["hotel_price_paise"],
                }
                if arguments != expected_arguments:
                    raise RuntimeError("Trip total must use the actual quote results.")

            if call.name in ("reserve_flight", "reserve_hotel"):
                if approval is not True:
                    raise RuntimeError("Reservations require explicit approval.")
                if forward_action_count >= MAX_FORWARD_ACTIONS:
                    raise RuntimeError("Forward-action budget exhausted.")
                expected_key = action_keys.get(call.name)
                if arguments.get("idempotency_key") != expected_key:
                    raise RuntimeError(
                        f"{call.name} used the wrong idempotency key."
                    )
                forward_action_count += 1

            if call.name == "cancel_flight":
                if not hotel_failed or flight_booking is None:
                    raise RuntimeError("Flight compensation is not required.")
                if compensation_count >= MAX_COMPENSATIONS:
                    raise RuntimeError("Compensation budget exhausted.")
                if arguments.get("booking_id") != flight_booking.get("booking_id"):
                    raise RuntimeError("Cancellation used the wrong booking ID.")
                if arguments.get("compensation_key") != action_keys.get(
                    "cancel_flight"
                ):
                    raise RuntimeError("Cancellation used the wrong key.")
                compensation_count += 1

            if call.name == "lookup_flight_booking":
                if not cancellation_ambiguous or flight_booking is None:
                    raise RuntimeError("Lookup requires an ambiguous cancellation.")
                if arguments.get("booking_id") != flight_booking.get("booking_id"):
                    raise RuntimeError("Lookup used the wrong booking ID.")

            call_count += 1
            log(
                f"Executing {call.name}; call_id={call.call_id!r}; "
                f"arguments={arguments!r}"
            )
            try:
                tool_result = registry[call.name](**arguments)
            except RoomUnavailableError as error:
                if call.name != "reserve_hotel":
                    raise
                steps.append(
                    {
                        "tool": call.name,
                        "arguments": arguments,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                hotel_failed = True
                payload = {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "retry_allowed": False,
                    "compensation_required": True,
                }
                log(f"Permanent hotel failure: {payload!r}")
            except TimeoutError as error:
                if call.name != "cancel_flight":
                    raise
                steps.append(
                    {
                        "tool": call.name,
                        "arguments": arguments,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                cancellation_ambiguous = True
                payload = {
                    "status": "unknown",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "retry_allowed": False,
                    "reconciliation_required": True,
                }
                log(f"Ambiguous cancellation: {payload!r}")
            except BookingNotFoundError as error:
                if call.name != "lookup_flight_booking":
                    raise
                steps.append(
                    {
                        "tool": call.name,
                        "arguments": arguments,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                lookup_failed = True
                payload = {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "retry_allowed": False,
                    "manual_intervention_required": True,
                }
                log(f"Reconciliation failed: {payload!r}")
            except Exception as error:
                steps.append(
                    {
                        "tool": call.name,
                        "arguments": arguments,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                log(f"Tool failed: {type(error).__name__}: {error}")
                raise
            else:
                steps.append(
                    {
                        "tool": call.name,
                        "arguments": arguments,
                        "status": "success",
                        "result": tool_result,
                    }
                )
                payload = tool_result
                if call.name == "get_flight_quote":
                    flight_quote = tool_result
                elif call.name == "get_hotel_quote":
                    hotel_quote = tool_result
                elif call.name == "calculate_hotel_stay_total":
                    hotel_total = tool_result
                elif call.name == "calculate_trip_total":
                    if hotel_quote is None:
                        raise RuntimeError("Hotel quote state is missing.")
                    trip_total = tool_result
                    quote = {
                        "flight_price_paise": trip_total["flight_price_paise"],
                        "nightly_price_paise": hotel_quote[
                            "nightly_price_paise"
                        ],
                        "nights": hotel_quote["nights"],
                        "hotel_price_paise": trip_total["hotel_price_paise"],
                        "total_price_paise": trip_total["total_price_paise"],
                        "budget_paise": TRIP_BUDGET_PAISE,
                    }
                elif call.name == "reserve_flight":
                    flight_booking = tool_result
                elif call.name == "cancel_flight":
                    cancellation_ambiguous = True
                elif call.name == "lookup_flight_booking":
                    verified_booking = tool_result
                    cancellation_ambiguous = False
                log(f"Tool result: {tool_result!r}")

            messages = [
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(payload),
                }
            ]
            previous_response_id = response.id
            continue

        if not message_items or not (response.output_text or "").strip():
            raise RuntimeError("Expected one tool call or a non-empty message.")
        try:
            decision = json.loads(response.output_text)
        except json.JSONDecodeError as error:
            raise RuntimeError("Malformed response: expected JSON.") from error
        if not isinstance(decision, dict):
            raise RuntimeError("Malformed response: expected a JSON object.")
        status = decision.get("status")
        message = decision.get("message")
        allowed_statuses = (
            "needs_confirmation",
            "cancelled",
            "rolled_back",
            "manual_intervention",
        )
        if status not in allowed_statuses:
            raise RuntimeError(f"Malformed response: unknown status {status!r}.")
        if not isinstance(message, str) or not message.strip():
            raise RuntimeError("Malformed response: message must be non-empty.")
        message = message.strip()
        previous_response_id = response.id

        if status == "needs_confirmation":
            if (
                call_count != 4
                or quote is None
                or approval is not None
                or quote["total_price_paise"] > TRIP_BUDGET_PAISE
            ):
                raise RuntimeError("Confirmation requested in an invalid state.")
            print(f"Agent: {message}", flush=True)
            reply = read_user(f"User: {message} ").strip()
            if not reply:
                raise RuntimeError("User approval reply must not be empty.")
            approval = _approval_decision(reply)
            if approval:
                action_keys = {
                    action: create_action_key(action)
                    for action in (
                        "reserve_flight",
                        "reserve_hotel",
                        "cancel_flight",
                    )
                }
                messages = [
                    {
                        "role": "user",
                        "content": (
                            f"{reply}\nApplication action keys: "
                            f"{json.dumps(action_keys)}"
                        ),
                    }
                ]
            else:
                messages = [{"role": "user", "content": reply}]
            log(f"Approval decision: {approval}")
            continue

        if status == "cancelled":
            if (
                approval is not False
                or call_count != 4
                or forward_action_count != 0
                or compensation_count != 0
            ):
                raise RuntimeError("Cancellation requires an explicit decline.")
            print(f"Agent: {message}", flush=True)
            return {
                "status": "cancelled",
                "message": message,
                "approved": False,
                "steps": steps,
                "quote": quote,
                "result": None,
                "call_count": call_count,
                "forward_action_count": forward_action_count,
                "compensation_count": compensation_count,
                "model_turn_count": model_turn_count,
                "action_keys": action_keys,
            }

        flight_status = (
            verified_booking.get("status")
            if verified_booking is not None
            else "unknown"
        )
        if status == "manual_intervention":
            if approval is not True or compensation_count != 1:
                raise RuntimeError("Manual intervention reported too early.")
            print(f"Agent: {message}", flush=True)
            return {
                "status": "manual_intervention",
                "message": message,
                "approved": True,
                "steps": steps,
                "quote": quote,
                "result": {
                    "flight_booking_id": (
                        flight_booking.get("booking_id")
                        if flight_booking is not None
                        else None
                    ),
                    "flight_status": flight_status,
                    "hotel_status": "not_booked",
                    "trip_status": "manual_intervention",
                },
                "call_count": call_count,
                "forward_action_count": forward_action_count,
                "compensation_count": compensation_count,
                "model_turn_count": model_turn_count,
                "action_keys": action_keys,
            }

        if (
            approval is not True
            or quote is None
            or flight_booking is None
            or not hotel_failed
            or lookup_failed
            or cancellation_ambiguous
            or verified_booking is None
            or verified_booking.get("status") != "cancelled"
            or call_count != MAX_TOOL_CALLS
            or forward_action_count != MAX_FORWARD_ACTIONS
            or compensation_count != MAX_COMPENSATIONS
        ):
            raise RuntimeError("Rolled back requires a verified cancellation.")
        print(f"Agent: {message}", flush=True)
        return {
            "status": "rolled_back",
            "message": message,
            "approved": True,
            "steps": steps,
            "quote": quote,
            "result": {
                "flight_booking_id": flight_booking["booking_id"],
                "flight_status": verified_booking["status"],
                "hotel_status": "not_booked",
                "trip_status": "rolled_back",
            },
            "call_count": call_count,
            "forward_action_count": forward_action_count,
            "compensation_count": compensation_count,
            "model_turn_count": model_turn_count,
            "action_keys": action_keys,
        }

    raise RuntimeError("Model-turn budget exhausted.")


if __name__ == "__main__":
    solve(debug=True)
