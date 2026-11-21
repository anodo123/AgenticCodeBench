"""Learner scaffold for Challenge 005: agent loop and stopping.

This challenge replaces a fixed two-response workflow with a bounded loop. You
will learn to carry response state between turns, return every real tool result
to its matching call, detect the model's final non-tool response, and stop safely
when the configured model-turn limit is reached.
"""

import json
from pathlib import Path
from typing import Any

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.model import call_openai_model as call_openai_model
from agentic_code_bench.core.tools import TOOL_REGISTRY

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[3] / "challenges" / "005_agent_loop.json"
)
MAX_MODEL_TURNS = 4


def solve(model: str = "gpt-5.6-luna", debug: bool = True) -> dict[str, Any]:
    """Run a bounded model-tool loop and return actual execution evidence.

    Start with the challenge messages. On each turn, make one model request and
    inspect all output items. A turn with one function call executes that tool
    once and supplies its result to the next turn. A turn with no function call
    is the successful stopping condition. Reject multiple calls in one turn.
    Never read ``challenge["expected"]`` to construct the result. Set
    ``debug`` to true to print each model turn, execution, and handoff.
    """
    challenge = load_challenge(CHALLENGE_PATH)

    # State you will need:
    steps: list[dict[str, Any]] = []
    messages = challenge["messages"]
    previous_response_id: str | None = None
    model_turn_count = 0
    # TODO: Loop for at most MAX_MODEL_TURNS.
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
            output_types = [item.type for item in response.output]
            print(f"Response ID: {response.id}")
            print(f"Response status: {response.status}")
            print(f"Output types: {output_types}")

        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]

        # TODO: If function_calls is empty, stop successfully and return:
        if not function_calls:
            final_messages = [
                item for item in response.output
                if item.type == "message"
            ]

            if not final_messages:
                raise RuntimeError(
                    "Model returned neither a function call nor a final message"
                )

            if not steps:
                raise RuntimeError("Model stopped before executing any tools")

            if debug:
                print(f"Final model text: {response.output_text!r}")
                print("Stopping: final message received with no function call.")
                print(f"Executed steps: {steps}")

            return {
                "steps": steps,
                "result": steps[-1]["result"],
                "call_count": len(steps),
                "model_turn_count": model_turn_count,
            }

        # TODO: Reject more than one function call in a single model turn.
        if len(function_calls) > 1:
            raise RuntimeError(
                f"Expected at most one function call per turn, "
                f"got {len(function_calls)}"
            )
        # TODO: For the one function call, parse arguments, resolve its name through
        # TOOL_REGISTRY, execute it exactly once, and append the execution to steps.
        tool_call = function_calls[0]
        arguments = json.loads(tool_call.arguments)

        if debug:
            print(f"Selected tool: {tool_call.name}")
            print(f"Call ID: {tool_call.call_id}")
            print(f"Parsed arguments: {arguments}")
        tool_function = TOOL_REGISTRY[tool_call.name]
        tool_call_result = tool_function(**arguments)

        if debug:
            print(f"Tool result: {tool_call_result!r}")

        steps.append(
            {
                "tool": tool_call.name,
                "arguments": arguments,
                "result": tool_call_result,
            }
        )

        #TODO: Prepare the next turn. The output must be a JSON string:
        messages = [{
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": json.dumps(tool_call_result),
        }]
        previous_response_id = response.id

        if debug:
            print(
                "Returning tool output to model: "
                f"call_id={tool_call.call_id}, output={messages[0]['output']}"
            )

    raise RuntimeError(
        f"Agent exceeded the maximum of {MAX_MODEL_TURNS} model turns"
    )


if __name__ == "__main__":
    import pytest

    test_path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "challenges"
        / "test_challenge_005_agent_loop.py"
    )
    raise SystemExit(pytest.main([str(test_path), "-v"]))
