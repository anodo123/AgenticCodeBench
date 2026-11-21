"""Challenge 008: interactive clarification within a bounded agent loop.

Question:
Calculate a rectangle's area, then add a waste allowance to find the total
material required. Width is 5 units. Height and waste percentage are missing:
ask the user for height first, then for the waste percentage. Never invent either
value. Use calculate_rectangle_area, calculate_percentage on the actual area,
and add_numbers to add the area and allowance, in that order. Request at most
one tool per model response and return each actual result to the model.

When asking the user a question, return JSON with status needs_input and message.
After all three tools succeed, return JSON with status completed and the final
answer in message. Preserve conversation context and enforce loop limits.

User replies come from the terminal, not the answer key. Example replies in the
challenge JSON are for manual practice only.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.model import call_openai_model
from agentic_code_bench.core.tools import TOOL_REGISTRY

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "challenges"
    / "008_resume_after_clarification.json"
)
MAX_MODEL_TURNS = 10
MAX_TOOL_CALLS = 3


def solve(
    model: str = "gpt-5.6-luna",
    debug: bool = True,
    read_user: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Route responses through tools, clarification, and completion.

    read_user defaults to terminal input. Inject a callable for offline testing.
    Never use expected or example_user_replies to construct model inputs/results.
    No fixed turn numbers: decide the next action from each model response.
    """
    def log(message: str) -> None:
        if debug:
            print(f"[challenge_008] {message}", flush=True)

    if read_user is None:
        read_user = input

    challenge = load_challenge(CHALLENGE_PATH)
    messages = challenge["messages"]
    previous_response_id: str | None = None
    model_turn_count = 0
    call_count = 0
    steps: list[dict[str, Any]] = []
    clarifications: list[dict[str, str]] = []
    allowed_tools = {tool["name"] for tool in challenge["tools"]}
    log(f"Loaded {challenge['id']}; allowed tools={sorted(allowed_tools)}")

    # TODO 1: Implement a bounded loop using MAX_MODEL_TURNS.
    # Before every request log turn number, previous_response_id, and messages.
    # Call the model with current messages, challenge tools, model, and response
    # ID. Increment model_turn_count after success. Log ID/status/output types
    # and raw output_text BEFORE validation, so failures remain visible.
    
    for _ in range(MAX_MODEL_TURNS):
        if debug:
            print(f"\n=== Model turn {model_turn_count + 1} ===")
            print(f"Previous response ID: {previous_response_id}")
            print(f"Input messages: {messages}")

        response = call_openai_model(
            messages=messages,
            tools=challenge["tools"],
            model=model,
            previous_response_id=previous_response_id,
        )
        model_turn_count += 1

        if debug:
            print(f"Response ID: {response.id}")
            print(f"Response status: {response.status}")
            print(f"Output types: {[item.type for item in response.output]}")

        # TODO 2: Inspect response.output and collect only function_call items.
        # If there are any, enter the TOOL branch:
        # - Reject multiple calls, unknown/non-exposed tools, or exhausted call budget.
        # - json.loads(call.arguments) must produce a dictionary.
        # - Resolve through TOOL_REGISTRY and increment call_count on the real attempt.
        # - Execute once, log name/call_id/arguments/result, and append execution
        #   evidence to steps. On an exception log it and re-raise; do not silently retry.
        # - Next messages: one function_call_output with matching call_id and
        #   json.dumps(actual_result). Set previous_response_id=response.id.
        # - Continue the SAME loop. Never manufacture or hardcode tool arguments.
        function_calls = [
                    item for item in response.output if item.type == "function_call"
                ]
        message_calls = [
                    item for item in response.output if item.type == "message"
                ]
        if function_calls:
            if len(function_calls) != 1:
                raise RuntimeError("Expected exactly one tool call per turn.")
            if call_count >= MAX_TOOL_CALLS:
                raise RuntimeError("Tool-call budget exhausted.")
            tool_call = function_calls[0]
            if tool_call.name not in allowed_tools or tool_call.name not in TOOL_REGISTRY:
                raise RuntimeError(f"Unknown or unavailable tool: {tool_call.name}")
            arguments = json.loads(tool_call.arguments)
            if not isinstance(arguments, dict):
                raise RuntimeError("Tool arguments must be a JSON object.")
            log(f"Tool: {tool_call.name}; call_id={tool_call.call_id}; arguments={arguments}")
            call_count += 1
            try:
                tool_call_result = TOOL_REGISTRY[tool_call.name](**arguments)
            except Exception as error:
                log(f"Tool FAILED: {type(error).__name__}: {error}")
                raise
            steps.append({
                "tool": tool_call.name,
                "arguments": arguments,
                "result": tool_call_result,
            })
            log(f"Tool result: {tool_call_result!r}")
            messages = [{
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": json.dumps(tool_call_result),
            }]
            previous_response_id = response.id
            continue

        log(f"Raw output text: {response.output_text!r}")
        if not message_calls or not (response.output_text or "").strip():
            raise RuntimeError("Expected a message with non-empty output text.")
        try:
            exact_message = json.loads(response.output_text)
        except json.JSONDecodeError as error:
            raise RuntimeError("Malformed response: expected JSON.") from error
        if not isinstance(exact_message, dict):
            raise RuntimeError("Malformed response: expected a JSON object.")
        status = exact_message.get("status")
        text = exact_message.get("message")
        if status not in ("needs_input", "completed"):
            raise RuntimeError(f"Malformed response: unknown status {status!r}.")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Malformed response: message must be a non-empty string.")
        text = text.strip()
        previous_response_id = response.id

        if status == "needs_input":
            print(f"Agent: {text}", flush=True)
            if model_turn_count >= MAX_MODEL_TURNS:
                raise RuntimeError("Model-turn budget exhausted before user reply.")
            try:
                reply = read_user(f"User Input {text}: ").strip()
            except (EOFError, KeyboardInterrupt):
                log("Stopped: waiting for user input.")
                return {
                    "status": "needs_input",
                    "message": text,
                    "steps": steps,
                    "clarifications": clarifications,
                    "result": steps[-1]["result"] if steps else None,
                    "call_count": call_count,
                    "model_turn_count": model_turn_count,
                    "previous_response_id": previous_response_id,
                }
            if not reply:
                raise RuntimeError("User reply must not be empty.")
            clarifications.append({"question": text, "reply": reply})
            messages = [{"role": "user", "content": reply}]
            log("User reply received; continuing the conversation.")
            continue

        elif status == "completed":
            required_tools = [tool["name"] for tool in challenge["tools"]]
            if [step["tool"] for step in steps] != required_tools:
                raise RuntimeError("Premature completion: required tools must execute in order.")
            print(f"Agent: {text}", flush=True)
            log("Stopped: all required tools succeeded and final answer received.")
            return {
                "status": "completed",
                "message": text,
                "steps": steps,
                "clarifications": clarifications,
                "result": steps[-1]["result"],
                "call_count": call_count,
                "model_turn_count": model_turn_count,
                "previous_response_id": previous_response_id,
            }

    log(f"Model-turn budget exhausted: turns={model_turn_count}, calls={call_count}")
    raise RuntimeError(f"Agent exceeded {MAX_MODEL_TURNS} model turns.")


if __name__ == "__main__":
    solve(debug=True)