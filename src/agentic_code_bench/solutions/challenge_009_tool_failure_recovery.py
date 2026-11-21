"""Challenge 009: bounded recovery inside an agentic loop.

Question:
Find the delivery status of order ORD-1042. First look up the order, then use its
actual tracking ID to query the carrier. The order service times out on the first
attempt and succeeds on the second. Report the actual carrier delivery status.

Recover from temporary failures with at most two attempts per tool. Do not retry
permanent failures, invent tracking IDs, or claim success without a carrier
result. Send each actual result/error back to the model and let it choose the
next action. Enforce retry limits in Python, independently of model instructions.

Implement the orchestration TODOs yourself. Tools are local simulations.
"""

import json
from pathlib import Path
from typing import Any

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.model import call_openai_model
from agentic_code_bench.core.order_tools import create_order_tool_registry

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "challenges"
    / "009_tool_failure_recovery.json"
)
MAX_MODEL_TURNS = 8
MAX_ATTEMPTS_PER_TOOL = 2
MAX_TOOL_ATTEMPTS = 4


def solve(model: str = "gpt-5.6-luna", debug: bool = True) -> dict[str, Any]:
    """Return status, message, steps, result, call_count, model_turn_count.

    steps contains one entry per REAL execution attempt, including failures:
    {tool, arguments, status: success, result} or
    {tool, arguments, status: error, error_type, error_message, retryable}.
    call_count counts attempts, not just successful executions.
    Never read challenge["expected"] to construct outputs.
    """
    def log(message: str) -> None:
        if debug:
            print(f"[challenge_009] {message}", flush=True)

    challenge = load_challenge(CHALLENGE_PATH)
    registry = create_order_tool_registry()
    allowed_tools = {tool["name"] for tool in challenge["tools"]}
    messages = challenge["messages"]
    previous_response_id: str | None = None
    attempts_by_tool: dict[str, int] = {}
    steps: list[dict[str, Any]] = []
    call_count = 0
    model_turn_count = 0
    # Save the failed call when a retry is pending; save terminal error evidence
    # separately so the model cannot recover by switching tools after failure.
    pending_retry: dict[str, Any] | None = None
    terminal_error: dict[str, Any] | None = None
    carrier_result: dict[str, Any] | None = None
    log(f"Loaded {challenge['id']}; allowed tools={sorted(allowed_tools)}")

    for _ in range(MAX_MODEL_TURNS):
        log(f"--- Model turn {model_turn_count + 1}/{MAX_MODEL_TURNS} ---")
        log(
            f"Model inputs: messages={messages!r}, tools={challenge['tools']!r}, "
            f"model={model!r}, previous_response_id={previous_response_id!r}"
        )
        response = call_openai_model(
            messages=messages,
            tools=challenge["tools"],
            model=model,
            previous_response_id=previous_response_id,
        )
        model_turn_count += 1
        log(
            f"Response: id={response.id!r}, status={response.status!r}, "
            f"types={[item.type for item in response.output]!r}"
        )
        log(f"output_text={response.output_text!r}")
        if debug:
            for index, item in enumerate(response.output, start=1):
                log(f"Output item {index}: {item!r}")

        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        message_calls = [item for item in response.output if item.type == "message"]
        if len(function_calls) > 1:
            raise RuntimeError("Only one tool call is allowed per response.")
        if function_calls:
            call = function_calls[0]
            if call.name not in allowed_tools or call.name not in registry:
                raise RuntimeError(f"Unknown tool: {call.name}")
            if terminal_error is not None:
                raise RuntimeError("Tool calls are forbidden after a terminal error.")
            if call_count >= MAX_TOOL_ATTEMPTS:
                raise RuntimeError("Total tool attempt budget exhausted.")
            if attempts_by_tool.get(call.name, 0) >= MAX_ATTEMPTS_PER_TOOL:
                raise RuntimeError(f"Attempt budget exhausted for {call.name}.")
            try:
                arguments = json.loads(call.arguments)
            except (TypeError, ValueError) as error:
                raise RuntimeError("Tool arguments must be a JSON object.") from error
            if not isinstance(arguments, dict):
                raise RuntimeError("Tool arguments must be a JSON object.")
            if pending_retry is not None and (
                call.name != pending_retry["tool"]
                or arguments != pending_retry["arguments"]
            ):
                raise RuntimeError("A pending retry must use the same tool and arguments.")
            if any(
                step["tool"] == call.name and step["status"] == "success"
                for step in steps
            ):
                raise RuntimeError(f"Tool {call.name} already succeeded.")
            if call.name == "get_carrier_status":
                order_step = next(
                    (
                        step for step in steps
                        if step["tool"] == "get_order_status"
                        and step["status"] == "success"
                    ),
                    None,
                )
                if (
                    order_step is None
                    or not order_step["result"].get("tracking_id")
                    or arguments.get("tracking_id") != order_step["result"]["tracking_id"]
                ):
                    raise RuntimeError("Carrier calls require the successful order's tracking ID.")
            attempts_by_tool[call.name] = attempts_by_tool.get(call.name, 0) + 1
            call_count += 1
            log(
                f"Executing tool={call.name!r}, call_id={call.call_id!r}, "
                f"attempt={attempts_by_tool[call.name]}/{MAX_ATTEMPTS_PER_TOOL}, "
                f"total_attempts={call_count}/{MAX_TOOL_ATTEMPTS}, "
                f"arguments={json.dumps(arguments, ensure_ascii=False)}"
            )
            try:
                actual_result = registry[call.name](**arguments)
            except Exception as error:
                retryable = isinstance(error, (TimeoutError, ConnectionError))
                retry_allowed = (
                    retryable
                    and attempts_by_tool[call.name] < MAX_ATTEMPTS_PER_TOOL
                    and call_count < MAX_TOOL_ATTEMPTS
                )
                failure = {
                    "tool": call.name,
                    "arguments": arguments,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "retryable": retryable,
                }
                steps.append(failure)
                log(f"Tool {call.name} failed: {type(error).__name__}: {error}")
                pending_retry = failure if retry_allowed else None
                terminal_error = None if retry_allowed else failure
                payload = {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "retryable": retryable,
                    "retry_allowed": retry_allowed,
                }
            else:
                steps.append({
                    "tool": call.name,
                    "arguments": arguments,
                    "status": "success",
                    "result": actual_result,
                })
                pending_retry = None
                if call.name == "get_carrier_status":
                    carrier_result = actual_result
                payload = {"status": "success", "result": actual_result}

            log(f"Tool output: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            log(
                f"Recovery state: retry_pending={pending_retry is not None}, "
                f"terminal_error={terminal_error is not None}"
            )
            messages = [{
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(payload),
            }]
            previous_response_id = response.id
            continue

        if not message_calls or not (response.output_text or "").strip():
            raise RuntimeError("Expected a message with non-empty output text.")
        try:
            final_response = json.loads(response.output_text)
        except json.JSONDecodeError as error:
            raise RuntimeError("Malformed response: expected JSON.") from error
        if not isinstance(final_response, dict):
            raise RuntimeError("Malformed response: expected a JSON object.")
        status = final_response.get("status")
        text = final_response.get("message")
        if status not in ("completed", "failed"):
            raise RuntimeError(f"Malformed response: unknown status {status!r}.")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Malformed response: message must be a non-empty string.")
        text = text.strip()
        if status == "completed":
            if carrier_result is None or pending_retry is not None or terminal_error is not None:
                raise RuntimeError("Completion requires a carrier result and no unresolved error.")
            log("Stopped: carrier result received and recovery complete.")
        else:
            if pending_retry is not None or terminal_error is None:
                raise RuntimeError("Failure requires a terminal error with no retry pending.")
            log("Stopped: terminal tool error.")
        print(f"Agent: {text}", flush=True)
        report = {
            "status": status,
            "message": text,
            "steps": steps,
            "result": carrier_result if status == "completed" else None,
            "call_count": call_count,
            "model_turn_count": model_turn_count,
        }
        if debug:
            log(f"Final report: {json.dumps(report, indent=2, ensure_ascii=False)}")
        return report

    log(
        f"Model-turn budget exhausted: model_turn_count={model_turn_count}, "
        f"call_count={call_count}, attempts_by_tool={attempts_by_tool}"
    )
    raise RuntimeError("Model-turn budget exhausted.")


if __name__ == "__main__":
    solve(debug=True)
