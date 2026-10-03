import json
from pathlib import Path
from typing import Any

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.model import call_openai_model
from agentic_code_bench.core.tools import TOOL_REGISTRY

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[3] / "challenges" / "004_dependent_tool_calls.json"
)


def solve(model: str = "gpt-5.6-luna") -> dict[str, Any]:
    challenge = load_challenge(CHALLENGE_PATH)
    messages = challenge["messages"]
    previous_response_id = None
    steps: list[dict[str, Any]] = []
    for _ in range(2):
        response = call_openai_model(
            messages=messages,
            tools=challenge["tools"],
            model=model,
            previous_response_id=previous_response_id,
        )
        calls = [item for item in response.output if item.type == "function_call"]
        if len(calls) != 1:
            raise RuntimeError("Expected exactly one function call per dependent turn")
        call = calls[0]
        arguments = json.loads(call.arguments)
        result = TOOL_REGISTRY[call.name](**arguments)
        steps.append({"tool": call.name, "arguments": arguments, "result": result})
        messages = [
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result),
            }
        ]
        previous_response_id = response.id
    return {"steps": steps, "call_count": len(steps)}
