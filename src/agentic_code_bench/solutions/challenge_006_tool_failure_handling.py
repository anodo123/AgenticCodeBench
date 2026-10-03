import json
from pathlib import Path
from typing import Any

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.model import call_openai_model
from agentic_code_bench.core.tools import TOOL_REGISTRY

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "challenges"
    / "006_tool_failure_handling.json"
)

MAX_MODEL_TURNS = 2
MAX_TOOL_ATTEMPTS = 1


def solve(model: str = "gpt-5.6-luna", debug: bool = True) -> dict[str, Any]:
    challenge = load_challenge(CHALLENGE_PATH)
    messages = challenge["messages"]
    previous_response_id = None
    steps: list[dict[str, Any]] = []
    attempts = 0
    for turn in range(1, MAX_MODEL_TURNS + 1):
        if debug:
            print("Model turn:", turn, "Previous response ID:", previous_response_id)
            print("Input messages:", messages)
        response = call_openai_model(
            messages=messages,
            tools=challenge["tools"],
            model=model,
            previous_response_id=previous_response_id,
        )
        if debug:
            print("Response:", response.id, response.status)
            print("Output types:", [item.type for item in response.output])
        calls = [item for item in response.output if item.type == "function_call"]
        if len(calls) > 1:
            raise RuntimeError("Expected at most one function call per turn")
        if not calls:
            if not steps or not any(item.type == "message" for item in response.output):
                raise RuntimeError("Expected a final message after tool work")
            if debug:
                print("Final model text:", response.output_text)
                print("Stop reason: final message; steps:", steps)
            return {
                "steps": steps,
                "result": None,
                "status": "failed",
                "call_count": attempts,
                "model_turn_count": turn,
            }
        if attempts >= MAX_TOOL_ATTEMPTS:
            raise RuntimeError("Tool attempt limit reached; retry is forbidden")
        call = calls[0]
        arguments = json.loads(call.arguments)
        function = TOOL_REGISTRY[call.name]
        if debug:
            print("Selected tool:", call.name, "Call ID:", call.call_id)
            print("Arguments:", arguments)
        attempts += 1
        try:
            function(**arguments)
        except ZeroDivisionError as exc:
            payload = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            steps.append({"tool": call.name, "arguments": arguments, **payload})
            if debug:
                print("Attempt count:", attempts, "Caught exception:", repr(exc))
                print("Failure step:", steps[-1])
        else:
            raise RuntimeError(
                "Division unexpectedly succeeded; expected a tool failure"
            )
        messages = [
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(payload),
            }
        ]
        previous_response_id = response.id
        if debug:
            print("Outgoing function-call output:", messages[0])
    raise RuntimeError("Model turn limit exhausted without a final message")
