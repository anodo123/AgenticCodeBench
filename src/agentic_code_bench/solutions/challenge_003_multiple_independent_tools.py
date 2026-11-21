"""Learner scaffold for Challenge 003: multiple independent tool calls."""

import json
from pathlib import Path
from typing import Any

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.model import call_openai_model
from agentic_code_bench.core.tools import TOOL_REGISTRY

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "challenges"
    / "003_multiple_independent_tools.json"
)


def solve(model: str = "gpt-5.6-luna") -> dict[str, Any]:
    """Execute every independent function call selected by the model.

    Send only ``challenge["messages"]`` and ``challenge["tools"]`` to the
    model. Inspect every item in ``response.output``, execute each function call
    once through ``TOOL_REGISTRY``, and return actual execution evidence.

    Calls are independent, so their order is not part of the contract. Do not
    read ``challenge["expected"]`` or send it to the model.
    """
    challenge = load_challenge(CHALLENGE_PATH)

    # TODO: Call call_openai_model with challenge["messages"],
    # challenge["tools"], and model.
    response = call_openai_model(
                messages=challenge["messages"],
                tools=challenge["tools"],
                model=model,
            )

    debug_path = Path("challenge_003_response.json")

    with debug_path.open("w", encoding="utf-8") as file:
        json.dump(
            response.model_dump(mode="json"),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Response saved at: {debug_path.resolve()}")
    

    # TODO: Collect every response.output item whose type is "function_call".
    # Require exactly two calls for this challenge. Do not use
    # execute_tool_call(), because it executes only the first function call.
    function_calls = [item for item in response.output if item.type == "function_call"]

    # TODO: For each function call:
    #   1. Parse item.arguments with json.loads().
    #   2. Resolve item.name through TOOL_REGISTRY.
    #   3. Execute the resolved function exactly once.

    calls = []
    for item in function_calls:
        arguments = json.loads(item.arguments)
        tool_function = TOOL_REGISTRY[item.name]
        result = tool_function(**arguments)

        calls.append(
            {
                "tool": item.name,
                "arguments": arguments,
                "result": result,
            }
        )

    # TODO: Return actual execution evidence in this shape:
    return {
        "calls": calls,
        "call_count": len(calls),
    }
    raise NotImplementedError(f"Implement orchestration for {challenge['id']}.")


if __name__ == "__main__":
    #solve()
    import pytest

    test_path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "challenges"
        / "test_challenge_003_multiple_independent_tools.py"
    )
    raise SystemExit(pytest.main([str(test_path), "-v"]))
