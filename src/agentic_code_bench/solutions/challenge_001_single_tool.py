from pathlib import Path
from typing import Any

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.model import call_openai_model
from agentic_code_bench.core.tool_execution import execute_tool_call

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[3] / "challenges" / "001_single_tool.json"
)


def format_output(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value


def solve(model: str = "gpt-5.6-luna") -> int | float:
    challenge = load_challenge(CHALLENGE_PATH)

    response = call_openai_model(
        messages=challenge["messages"],
        tools=challenge["tools"],
        model=model,
    )

    tool_execution = execute_tool_call(response)

    if tool_execution is None:
        raise RuntimeError("Model did not call any tool")

    result = format_output(tool_execution["result"])

    if not isinstance(result, (int, float)):
        raise RuntimeError("Tool did not return a numeric result")
    return result


if __name__ == "__main__":
    import pytest

    test_path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "challenges"
        / "test_challenge_001_single_tool.py"
    )
    raise SystemExit(pytest.main([str(test_path), "-v"]))
