"""Learner solution for Challenge 006: tool failure handling.

This challenge teaches the error path of a model-tool loop. The learner catches
a real exception at the tool boundary, builds evidence from that exception,
returns a structured failure to the matching model call, prevents retries, and
stops only after the model responds with a final message.
"""

import json
from pathlib import Path
from typing import Any

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.model import call_openai_model as call_openai_model
from agentic_code_bench.core.tools import TOOL_REGISTRY

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "challenges"
    / "006_tool_failure_handling.json"
)
MAX_MODEL_TURNS = 2
MAX_TOOL_ATTEMPTS = 1


def solve(model: str = "gpt-5.6-luna", debug: bool = True) -> dict[str, Any]:
    """Handle one failed tool attempt and obtain the model's final response.

    Derive error details only from the exception raised by the registered tool.
    Send them back as JSON in a ``function_call_output`` linked by ``call_id``.
    Set ``debug`` to true to print model turns, the exception, and its handoff.
    Do not read the evaluator-only ``challenge["expected"]`` section.
    """
    challenge = load_challenge(CHALLENGE_PATH)
    steps: list[dict[str, Any]] = []
    messages = challenge["messages"]
    previous_response_id: str | None = None
    model_turn_count = 0
    tool_attempt_count = 0

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
        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]

        if debug:
            print(f"Response ID: {response.id}")
            print(f"Response status: {response.status}")
            print(f"Output types: {[item.type for item in response.output]}")

        if not function_calls:
            final_messages = [
                item for item in response.output if item.type == "message"
            ]
            if not final_messages:
                raise RuntimeError(
                    "Model returned neither a function call nor a final message"
                )
            if not steps:
                raise RuntimeError("Model stopped before executing any tools")

            if debug:
                print(f"Final model text: {response.output_text!r}")
                print("Stopping: final message received after the failed tool.")
                print(f"Recorded failure steps: {steps}")

            return {
                "steps": steps,
                "result": None,
                "status": "failed",
                "call_count": tool_attempt_count,
                "model_turn_count": model_turn_count,
            }

        if len(function_calls) > 1:
            raise RuntimeError(
                f"Expected at most one function call per turn, "
                f"got {len(function_calls)}"
            )

        tool_call = function_calls[0]
        if tool_attempt_count >= MAX_TOOL_ATTEMPTS:
            raise RuntimeError("Tool attempt limit exceeded")

        arguments = json.loads(tool_call.arguments)
        tool_function = TOOL_REGISTRY[tool_call.name]
        tool_attempt_count += 1

        if debug:
            print(f"Selected tool: {tool_call.name}")
            print(f"Call ID: {tool_call.call_id}")
            print(f"Parsed arguments: {arguments}")
            print(f"Tool attempt: {tool_attempt_count}/{MAX_TOOL_ATTEMPTS}")

        try:
            tool_call_result = tool_function(**arguments)
        except ZeroDivisionError as error:
            error_step = {
                "tool": tool_call.name,
                "arguments": arguments,
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            steps.append(error_step)
            failure = {
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            messages = [
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": json.dumps(failure),
                }
            ]

            if debug:
                print(
                    f"Caught exception: {type(error).__name__}: {error}"
                )
                print(f"Recorded error step: {error_step}")
                print(
                    "Returning failure to model: "
                    f"call_id={tool_call.call_id}, output={messages[0]['output']}"
                )
        else:
            raise RuntimeError(
                f"Expected tool failure, but tool returned {tool_call_result!r}"
            )

        previous_response_id = response.id

    raise RuntimeError(
        f"Agent exceeded the maximum of {MAX_MODEL_TURNS} model turns"
    )


if __name__ == "__main__":
    import pytest

    test_path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "challenges"
        / "test_challenge_006_tool_failure_handling.py"
    )
    raise SystemExit(pytest.main([str(test_path), "-v"]))
