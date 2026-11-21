"""Learner scaffold for Challenge 002: selecting one tool."""

from pathlib import Path
from typing import Any

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.model import call_openai_model
from agentic_code_bench.core.tool_execution import execute_tool_call
import json
from agentic_code_bench.core.tools import TOOL_REGISTRY
def format_output(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "challenges"
    / "002_tool_selection.json"
)


def solve(model: str = "gpt-5.6-luna") -> dict[str, Any]:
    """Select and execute one tool, then report the actual execution.

    Load challenges/002_tool_selection.json and send only its messages and
    tools to the model. Choose the tool and arguments from the model response,
    execute the function through TOOL_REGISTRY, and return a dictionary with:

    - tool: the executed function name.
    - arguments: the parsed arguments used for that execution.
    - result: the value returned by the function.
    - call_count: the actual number of function calls executed (an integer).

    The evaluator requires exactly one registered tool execution and an integer
    result. Do not read expected to construct output or include it in model input.
    Implement the orchestration here; the supporting environment is provided.
    """
    challenge = load_challenge(CHALLENGE_PATH)

    # TODO: Import and call core.model.call_openai_model with
    # challenge["messages"], challenge["tools"], and model.
    #
    # print(challenge["messages"])
    response = call_openai_model(
            messages=challenge["messages"],
            tools=challenge["tools"],
            model=model,
        )
    # tool_execution = execute_tool_call(response)

    # if tool_execution is None:
    #     raise RuntimeError("Model did not call any tool")

    # result = format_output(tool_execution["result"])

    # print(result)


    # TODO: Inspect the response and enforce exactly one function call.
    # core.tool_execution.execute_tool_call only executes the first call;
    # it does not verify that the response contains exactly one call.
    call_id = None
    call_count = 0
    for item in response.output:
        if item.type == "function_call":
            call_count+=1
            call_id =  item.call_id
            tool_name = item.name
            arguments = json.loads(item.arguments)
            tool_function = TOOL_REGISTRY[tool_name]

    # TODO: Parse the selected arguments and execute through TOOL_REGISTRY.
    # You can use execute_tool_call after validating the response.
    # Track the actual number of executed function calls.
    result = tool_function(**arguments)

    return {
        "call_id": call_id,
        "tool": tool_name,
        "arguments": arguments,
        "result": result,
        "call_count":call_count
    }




    # TODO: Return the actual execution report in this shape:
    # return {
    #     "tool": tool_execution["tool"],
    #     "arguments": tool_execution["arguments"],
    #     "result": tool_execution["result"],
    #     "call_count": call_count,
    # }
    # raise NotImplementedError(f"Implement orchestration for {challenge['id']}.")


if __name__ == "__main__":
    #solve()
    import pytest

    test_path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "challenges"
        / "test_challenge_002_tool_selection.py"
    )
    raise SystemExit(pytest.main([str(test_path), "-v"]))