"""Learner scaffold for Challenge 004: dependent tool calls."""

import json  # noqa: F401 - used when implementing the learner TODOs
from pathlib import Path
from typing import Any

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.model import call_openai_model as call_openai_model
from agentic_code_bench.core.tools import TOOL_REGISTRY  # noqa: F401

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "challenges"
    / "004_dependent_tool_calls.json"
)


def solve(model: str = "gpt-5.6-luna") -> dict[str, Any]:
    """Execute two dependent calls across two model responses.

    The first tool result must be returned to the model as a
    ``function_call_output`` linked by ``call_id``. Use the first response ID as
    ``previous_response_id`` for the second request. Send only challenge inputs,
    never ``challenge["expected"]``.
    """
    steps = []
    challenge = load_challenge(CHALLENGE_PATH)

    # TODO 1: Make the first model request:
    first_response = call_openai_model(
        messages=challenge["messages"],
        tools=challenge["tools"],
        model=model,
    )

    # TODO 2: Require exactly one function_call in first_response.output.
    # Parse its arguments, execute it exactly once through TOOL_REGISTRY, and
    # append {"tool", "arguments", "result"} to steps.
    function_calls = [item for item in first_response.output if item.type == "function_call"]

    if len(function_calls) != 1:
        raise RuntimeError(
            f"Expected exactly one function call, got {len(function_calls)}"
        )

    first_call = function_calls[0]

    arguments = json.loads(first_call.arguments)
    tool_function = TOOL_REGISTRY[first_call.name]
    first_result = tool_function(**arguments)
    steps.append(
        {
            "tool": first_call.name,
            "arguments": arguments,
            "result": first_result,
        }
    )
    

    # TODO 3: Send the actual first result back to the model:
    tool_output = {
        "type": "function_call_output",
        "call_id": first_call.call_id,
        "output": json.dumps(first_result),
    }
    second_response = call_openai_model(
        messages=[tool_output],
        tools=challenge["tools"],
        model=model,
        previous_response_id=first_response.id,
    )

    # TODO 4: Require exactly one function_call in second_response.output.
    # Parse and execute it once, then append its actual execution to steps.
    function_calls = [item for item in second_response.output if item.type == "function_call"]
    
    if len(function_calls) != 1:
        raise RuntimeError(
            f"Expected exactly one function call, got {len(function_calls)}"
        )

    second_call = function_calls[0]
    
    arguments = json.loads(second_call.arguments)
    tool_function = TOOL_REGISTRY[second_call.name]
    second_result = tool_function(**arguments)
    steps.append(
        {
            "tool": second_call.name,
            "arguments": arguments,
            "result": second_result,
        }
    )
    
    

    # TODO 5: Return:
    return {
        "steps": steps,
        "call_count": len(steps),
    }
    raise NotImplementedError(f"Implement orchestration for {challenge['id']}.")


if __name__ == "__main__":
    import pytest

    test_path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "challenges"
        / "test_challenge_004_dependent_tool_calls.py"
    )
    raise SystemExit(pytest.main([str(test_path), "-v"]))
