"""Learner scaffold for Challenge 010: safe money transfer.

Question:
Transfer 2500 rupees (250000 paise) from WALLET-A to WALLET-B. First check the
source balance and create a transfer preview. Show the fee and total debit, ask
for explicit confirmation, and execute only after approval. If execution times
out after committing, do not execute again. Reconcile using the same
application-owned idempotency key and report success only after verification.

This challenge combines an interactive approval gate with state-changing tools.
The transfer commits before its first execution response is lost, so successful
orchestration must reconcile the unknown outcome without repeating the payment.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.model import call_openai_model
from agentic_code_bench.core.wallet_tools import (
    create_idempotency_key,
    create_wallet_tool_registry,
)

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "challenges"
    / "010_safe_money_transfer.json"
)
MAX_MODEL_TURNS = 8
MAX_TOOL_CALLS = 4
MAX_SIDE_EFFECTS = 1


def _approval_decision(reply: str) -> bool:
    normalized = reply.strip().casefold()
    affirmative = ("yes", "y", "approve", "approved", "confirm", "proceed")
    negative = ("no", "n", "decline", "declined", "cancel", "cancelled")
    if normalized.startswith(affirmative):
        return True
    if normalized.startswith(negative):
        return False
    raise RuntimeError("User reply must explicitly approve or decline the transfer.")


def solve(
    model: str = "gpt-5.6-luna",
    debug: bool = True,
    read_user: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Return the verified transfer report.

    The report contains status, message, approved, steps, result, call_count,
    side_effect_count, model_turn_count, and idempotency_key. A declined transfer
    returns status cancelled, approved False, result None, and no side effect.
    """
    def log(message: str) -> None:
        if debug:
            print(f"[challenge_010] {message}", flush=True)

    if read_user is None:
        read_user = input

    challenge = load_challenge(CHALLENGE_PATH)
    registry = create_wallet_tool_registry()
    allowed_tools = {tool["name"] for tool in challenge["tools"]}
    required_order = [tool["name"] for tool in challenge["tools"]]
    messages = challenge["messages"]
    previous_response_id: str | None = None
    steps: list[dict[str, Any]] = []
    call_count = 0
    side_effect_count = 0
    model_turn_count = 0
    approval: bool | None = None
    idempotency_key: str | None = None
    preview: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    reconciliation_required = False

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
        message_items = [
            item for item in response.output if item.type == "message"
        ]
        if len(function_calls) > 1:
            raise RuntimeError("Only one tool call is allowed per model response.")

        if function_calls:
            if call_count >= MAX_TOOL_CALLS:
                raise RuntimeError("Tool-call budget exhausted.")
            call = function_calls[0]
            if call.name not in allowed_tools or call.name not in registry:
                raise RuntimeError(f"Unknown or unavailable tool: {call.name}")
            if call.name != required_order[call_count]:
                raise RuntimeError(
                    f"Expected {required_order[call_count]}, received {call.name}."
                )
            try:
                arguments = json.loads(call.arguments)
            except (TypeError, ValueError) as error:
                raise RuntimeError("Tool arguments must be valid JSON.") from error
            if not isinstance(arguments, dict):
                raise RuntimeError("Tool arguments must be a JSON object.")

            if call.name == "execute_transfer":
                if approval is not True:
                    raise RuntimeError("Transfer execution requires explicit approval.")
                if idempotency_key is None or preview is None:
                    raise RuntimeError("Execution safety state is incomplete.")
                if arguments.get("idempotency_key") != idempotency_key:
                    raise RuntimeError("Model used the wrong idempotency key.")
                if arguments.get("preview_id") != preview.get("preview_id"):
                    raise RuntimeError("Model used the wrong transfer preview.")
                if side_effect_count >= MAX_SIDE_EFFECTS:
                    raise RuntimeError("Financial side-effect budget exhausted.")
                side_effect_count += 1

            if call.name == "lookup_transfer_by_idempotency_key":
                if not reconciliation_required or idempotency_key is None:
                    raise RuntimeError(
                        "Transfer lookup requires an ambiguous execution."
                    )
                if arguments.get("idempotency_key") != idempotency_key:
                    raise RuntimeError("Lookup used the wrong idempotency key.")

            call_count += 1
            log(
                f"Executing {call.name}; call_id={call.call_id!r}; "
                f"arguments={arguments!r}"
            )
            try:
                tool_result = registry[call.name](**arguments)
            except TimeoutError as error:
                if call.name != "execute_transfer":
                    raise
                step = {
                    "tool": call.name,
                    "arguments": arguments,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
                steps.append(step)
                reconciliation_required = True
                payload = {
                    "status": "unknown",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "retry_allowed": False,
                    "reconciliation_required": True,
                }
                log(f"Ambiguous execution: {payload!r}")
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
                step = {
                    "tool": call.name,
                    "arguments": arguments,
                    "status": "success",
                    "result": tool_result,
                }
                steps.append(step)
                payload = tool_result
                if call.name == "create_transfer_preview":
                    if not isinstance(tool_result, dict) or not tool_result.get(
                        "preview_id"
                    ):
                        raise RuntimeError("Preview tool returned an invalid result.")
                    preview = tool_result
                if call.name == "lookup_transfer_by_idempotency_key":
                    if not isinstance(tool_result, dict):
                        raise RuntimeError("Lookup tool returned an invalid result.")
                    result = tool_result
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
        if status not in ("needs_confirmation", "cancelled", "completed"):
            raise RuntimeError(f"Malformed response: unknown status {status!r}.")
        if not isinstance(message, str) or not message.strip():
            raise RuntimeError("Malformed response: message must be non-empty.")
        message = message.strip()
        previous_response_id = response.id

        if status == "needs_confirmation":
            if call_count != 2 or preview is None or approval is not None:
                raise RuntimeError("Confirmation requested in an invalid state.")
            print(f"Agent: {message}", flush=True)
            reply = read_user(f"User: {message} ").strip()
            if not reply:
                raise RuntimeError("User approval reply must not be empty.")
            approval = _approval_decision(reply)
            if approval:
                idempotency_key = create_idempotency_key()
                messages = [
                    {
                        "role": "user",
                        "content": (
                            f"{reply}\nApplication idempotency key: "
                            f"{idempotency_key}"
                        ),
                    }
                ]
            else:
                messages = [{"role": "user", "content": reply}]
            log(f"Approval decision: {approval}")
            continue

        if status == "cancelled":
            if approval is not False or call_count != 2 or side_effect_count != 0:
                raise RuntimeError("Cancellation requires an explicit decline.")
            print(f"Agent: {message}", flush=True)
            return {
                "status": "cancelled",
                "message": message,
                "approved": False,
                "steps": steps,
                "result": None,
                "call_count": call_count,
                "side_effect_count": side_effect_count,
                "model_turn_count": model_turn_count,
                "idempotency_key": None,
            }

        if (
            approval is not True
            or result is None
            or result.get("status") != "completed"
            or reconciliation_required
            or call_count != MAX_TOOL_CALLS
            or side_effect_count != MAX_SIDE_EFFECTS
        ):
            raise RuntimeError("Completion requires a verified completed transfer.")
        print(f"Agent: {message}", flush=True)
        return {
            "status": "completed",
            "message": message,
            "approved": True,
            "steps": steps,
            "result": result,
            "call_count": call_count,
            "side_effect_count": side_effect_count,
            "model_turn_count": model_turn_count,
            "idempotency_key": idempotency_key,
        }

    raise RuntimeError("Model-turn budget exhausted.")


if __name__ == "__main__":
    solve(debug=True)
