"""Solution for Challenge 012: approval invalidation.

Question:
Purchase three tickets for event EVT-901 only when the final cost is at most
18000 rupees. Fetch a versioned quote, calculate fees and total, and ask for
explicit approval. The first approved quote expires without purchasing. Fetch
the revised quote, invalidate the old approval, and ask again. The second
purchase commits before timing out; reconcile it by idempotency key without
purchasing again.

Learning goals:
Bind approval to an exact plan, invalidate stale approval, keep idempotency keys
scoped to immutable purchase intents, distinguish safe replanning from unsafe
retrying, and verify an ambiguous committed purchase.

Output contract:
Return status, message, approval_count, steps, initial_plan, revised_plan,
result, call_count, purchase_attempt_count, committed_purchase_count,
model_turn_count, and idempotency_keys.
"""

import json as json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentic_code_bench.core.challenge import load_challenge as load_challenge
from agentic_code_bench.core.model import call_openai_model as call_openai_model
from agentic_code_bench.core.ticket_tools import (
    TICKET_BUDGET_PAISE as TICKET_BUDGET_PAISE,
)
from agentic_code_bench.core.ticket_tools import (
    PurchaseNotFoundError as PurchaseNotFoundError,
)
from agentic_code_bench.core.ticket_tools import (
    QuoteExpiredError as QuoteExpiredError,
)
from agentic_code_bench.core.ticket_tools import (
    create_purchase_key as create_purchase_key,
)
from agentic_code_bench.core.ticket_tools import (
    create_ticket_tool_registry as create_ticket_tool_registry,
)

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "challenges"
    / "012_approval_invalidation.json"
)
MAX_MODEL_TURNS = 12
MAX_TOOL_CALLS = 7
MAX_PURCHASE_ATTEMPTS = 2
MAX_COMMITTED_PURCHASES = 1
REQUIRED_APPROVALS = 2


def _approval_decision(reply: str) -> bool:
    normalized = reply.strip().casefold()
    affirmative = ("yes", "y", "approve", "approved", "confirm", "proceed")
    negative = ("no", "n", "decline", "declined", "cancel", "cancelled")
    if normalized.startswith(affirmative):
        return True
    if normalized.startswith(negative):
        return False
    raise RuntimeError("User reply must explicitly approve or decline the plan.")


def solve(
    model: str = "gpt-5.6-luna",
    debug: bool = True,
    read_user: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Return the verified ticket-purchase report described above."""
    def log(message: str) -> None:
        if debug:
            print(f"[challenge_012] {message}", flush=True)

    if read_user is None:
        read_user = input

    challenge = load_challenge(CHALLENGE_PATH)
    registry = create_ticket_tool_registry()
    allowed_tools = {tool["name"] for tool in challenge["tools"]}
    required_sequence = (
        "get_ticket_quote",
        "calculate_ticket_total",
        "purchase_tickets",
        "get_ticket_quote",
        "calculate_ticket_total",
        "purchase_tickets",
        "lookup_purchase_by_idempotency_key",
    )
    messages = challenge["messages"]
    previous_response_id: str | None = None
    steps: list[dict[str, Any]] = []
    call_count = 0
    purchase_attempt_count = 0
    committed_purchase_count = 0
    approval_count = 0
    model_turn_count = 0
    approval: bool | None = None
    approved_plan: dict[str, Any] | None = None
    current_quote: dict[str, Any] | None = None
    current_plan: dict[str, Any] | None = None
    initial_plan: dict[str, Any] | None = None
    revised_plan: dict[str, Any] | None = None
    idempotency_keys: dict[str, str] = {}
    reconciliation_required = False
    lookup_failed = False
    verified_result: dict[str, Any] | None = None

    def report(
        status: str,
        message: str,
        result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "message": message,
            "approval_count": approval_count,
            "steps": steps,
            "initial_plan": initial_plan,
            "revised_plan": revised_plan,
            "result": result,
            "call_count": call_count,
            "purchase_attempt_count": purchase_attempt_count,
            "committed_purchase_count": committed_purchase_count,
            "model_turn_count": model_turn_count,
            "idempotency_keys": idempotency_keys,
        }

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
            expected_tool = required_sequence[call_count]
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

            if call.name == "calculate_ticket_total":
                if current_quote is None:
                    raise RuntimeError("A current quote is required for calculation.")
                expected_arguments = {
                    "unit_price_paise": current_quote["unit_price_paise"],
                    "quantity": 3,
                }
                if arguments != expected_arguments:
                    raise RuntimeError(
                        "Ticket total must use the current quote and quantity."
                    )

            if call.name == "purchase_tickets":
                if approval is not True or approved_plan is None:
                    raise RuntimeError(
                        "Ticket purchase requires approval for the active plan."
                    )
                quote_id = approved_plan["quote_id"]
                expected_arguments = {
                    "quote_id": quote_id,
                    "quantity": approved_plan["quantity"],
                    "idempotency_key": idempotency_keys.get(quote_id),
                }
                if arguments != expected_arguments:
                    raise RuntimeError(
                        "Purchase arguments do not match the approved plan."
                    )
                if approved_plan["total_price_paise"] > TICKET_BUDGET_PAISE:
                    raise RuntimeError("Approved plan exceeds the ticket budget.")
                if purchase_attempt_count >= MAX_PURCHASE_ATTEMPTS:
                    raise RuntimeError("Purchase-attempt budget exhausted.")
                purchase_attempt_count += 1

            if call.name == "lookup_purchase_by_idempotency_key":
                if not reconciliation_required or revised_plan is None:
                    raise RuntimeError("Lookup requires an ambiguous purchase.")
                revised_key = idempotency_keys.get(revised_plan["quote_id"])
                if arguments.get("idempotency_key") != revised_key:
                    raise RuntimeError("Lookup used the wrong idempotency key.")

            call_count += 1
            log(
                f"Executing {call.name}; call_id={call.call_id!r}; "
                f"arguments={arguments!r}"
            )
            try:
                tool_result = registry[call.name](**arguments)
            except QuoteExpiredError as error:
                if call.name != "purchase_tickets" or initial_plan is None:
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
                approval = None
                approved_plan = None
                current_quote = None
                current_plan = None
                payload = {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "retry_allowed": False,
                    "approval_invalidated": True,
                    "replan_required": True,
                }
                log(f"Approved quote expired: {payload!r}")
            except TimeoutError as error:
                if call.name != "purchase_tickets" or revised_plan is None:
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
                reconciliation_required = True
                payload = {
                    "status": "unknown",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "retry_allowed": False,
                    "reconciliation_required": True,
                }
                log(f"Ambiguous purchase: {payload!r}")
            except PurchaseNotFoundError as error:
                if call.name != "lookup_purchase_by_idempotency_key":
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
                log(f"Purchase reconciliation failed: {payload!r}")
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
                if call.name == "get_ticket_quote":
                    current_quote = tool_result
                elif call.name == "calculate_ticket_total":
                    if current_quote is None:
                        raise RuntimeError("Current quote state is missing.")
                    current_plan = {
                        "event_id": current_quote["event_id"],
                        "quote_id": current_quote["quote_id"],
                        "quantity": tool_result["quantity"],
                        "unit_price_paise": tool_result["unit_price_paise"],
                        "subtotal_paise": tool_result["subtotal_paise"],
                        "fee_paise": tool_result["fee_paise"],
                        "total_price_paise": tool_result["total_price_paise"],
                        "budget_paise": TICKET_BUDGET_PAISE,
                    }
                    if initial_plan is None:
                        initial_plan = current_plan.copy()
                    else:
                        revised_plan = current_plan.copy()
                elif call.name == "lookup_purchase_by_idempotency_key":
                    verified_result = tool_result
                    reconciliation_required = False
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
            "over_budget",
            "completed",
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
                current_plan is None
                or approval is not None
                or current_plan["total_price_paise"] > TICKET_BUDGET_PAISE
                or approval_count >= REQUIRED_APPROVALS
            ):
                raise RuntimeError("Confirmation requested in an invalid state.")
            print(f"Agent: {message}", flush=True)
            reply = read_user(f"User: {message} ").strip()
            if not reply:
                raise RuntimeError("User approval reply must not be empty.")
            approval = _approval_decision(reply)
            if approval:
                approved_plan = current_plan.copy()
                approval_count += 1
                quote_id = approved_plan["quote_id"]
                idempotency_keys[quote_id] = create_purchase_key(quote_id)
                messages = [
                    {
                        "role": "user",
                        "content": (
                            f"{reply}\nApproved plan: "
                            f"{json.dumps(approved_plan)}\nApplication "
                            f"idempotency key: {idempotency_keys[quote_id]}"
                        ),
                    }
                ]
            else:
                messages = [{"role": "user", "content": reply}]
            log(f"Approval decision: {approval}")
            continue

        if status == "cancelled":
            if approval is not False:
                raise RuntimeError("Cancellation requires an explicit decline.")
            print(f"Agent: {message}", flush=True)
            return report("cancelled", message, None)

        if status == "over_budget":
            if (
                current_plan is None
                or current_plan["total_price_paise"] <= TICKET_BUDGET_PAISE
                or approval is True
            ):
                raise RuntimeError("Over-budget reported in an invalid state.")
            print(f"Agent: {message}", flush=True)
            return report("over_budget", message, None)

        if status == "manual_intervention":
            if not reconciliation_required and not lookup_failed:
                raise RuntimeError("Manual intervention reported too early.")
            print(f"Agent: {message}", flush=True)
            return report("manual_intervention", message, verified_result)

        if revised_plan is None or verified_result is None:
            raise RuntimeError("Completion requires a verified revised purchase.")
        expected_result = {
            "event_id": revised_plan["event_id"],
            "quote_id": revised_plan["quote_id"],
            "quantity": revised_plan["quantity"],
            "unit_price_paise": revised_plan["unit_price_paise"],
            "subtotal_paise": revised_plan["subtotal_paise"],
            "fee_paise": revised_plan["fee_paise"],
            "total_price_paise": revised_plan["total_price_paise"],
            "status": "completed",
            "idempotency_key": idempotency_keys.get(revised_plan["quote_id"]),
        }
        if (
            not isinstance(verified_result.get("purchase_id"), str)
            or not verified_result["purchase_id"].strip()
            or any(
                verified_result.get(field) != value
                for field, value in expected_result.items()
            )
            or approval_count != REQUIRED_APPROVALS
            or purchase_attempt_count != MAX_PURCHASE_ATTEMPTS
            or call_count != MAX_TOOL_CALLS
            or reconciliation_required
            or lookup_failed
        ):
            raise RuntimeError("Completion requires the exact approved purchase.")
        committed_purchase_count = 1
        if committed_purchase_count != MAX_COMMITTED_PURCHASES:
            raise RuntimeError("Committed-purchase invariant failed.")
        print(f"Agent: {message}", flush=True)
        return report("completed", message, verified_result)

    raise RuntimeError("Model-turn budget exhausted.")


if __name__ == "__main__":
    solve(debug=True)
