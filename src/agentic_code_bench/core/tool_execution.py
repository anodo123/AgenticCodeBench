import json
from typing import Any

from agentic_code_bench.core.tools import TOOL_REGISTRY


def execute_tool_call(response: Any) -> dict[str, Any] | None:
    for item in response.output:
        if item.type == "function_call":
            tool_name = item.name
            arguments = json.loads(item.arguments)
            tool_function = TOOL_REGISTRY[tool_name]
            result = tool_function(**arguments)

            return {
                "call_id": item.call_id,
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
            }

    return None
