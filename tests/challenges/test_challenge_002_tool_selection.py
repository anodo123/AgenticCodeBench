from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import pytest

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.tools import TOOL_REGISTRY
from agentic_code_bench.solutions.challenge_002_tool_selection import solve

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[2] / "challenges" / "002_tool_selection.json"
)


def test_challenge_002_tool_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    challenge = load_challenge(CHALLENGE_PATH)
    expected = challenge["expected"]
    calls: list[dict[str, Any]] = []

    def record_calls(
        name: str, function: Callable[..., object]
    ) -> Callable[..., object]:
        @wraps(function)
        def recorded(**arguments: Any) -> object:
            call: dict[str, Any] = {"tool": name, "arguments": arguments}
            calls.append(call)
            result = function(**arguments)
            call["result"] = result
            return result

        return recorded

    for name, function in TOOL_REGISTRY.items():
        monkeypatch.setitem(TOOL_REGISTRY, name, record_calls(name, function))

    actual = solve()

    assert isinstance(actual, dict)
    assert {"tool", "arguments", "result", "call_count"} <= actual.keys()
    assert type(actual["call_count"]) is int
    assert actual["call_count"] == len(calls) == expected["call_count"]
    assert actual["tool"] == calls[0]["tool"] == expected["tool"]
    assert actual["arguments"] == calls[0]["arguments"] == expected["arguments"]
    assert actual["result"] == calls[0]["result"] == expected["output"]
    assert type(actual["result"]) is int
    assert type(calls[0]["result"]) is int
