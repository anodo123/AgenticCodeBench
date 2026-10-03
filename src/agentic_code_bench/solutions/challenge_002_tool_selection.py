import json
from pathlib import Path
from typing import Any

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.model import call_openai_model
from agentic_code_bench.core.tools import TOOL_REGISTRY

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[3] / "challenges" / "002_tool_selection.json"
)


def solve(model: str = "gpt-5.6-luna") -> dict[str, Any]:
    challenge = load_challenge(CHALLENGE_PATH)
    response = call_openai_model(
        messages=challenge["messages"], tools=challenge["tools"], model=model
    )
    selected = [item for item in response.output if item.type == "function_call"]
    if len(selected) != 1:
        raise RuntimeError("Expected exactly one function call")
    calls: list[dict[str, Any]] = []
    for call in selected:
        arguments = json.loads(call.arguments)
        result = TOOL_REGISTRY[call.name](**arguments)
        calls.append({"tool": call.name, "arguments": arguments, "result": result})
    return {**calls[0], "call_count": len(calls)}
