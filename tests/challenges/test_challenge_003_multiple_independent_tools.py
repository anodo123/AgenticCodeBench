from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import pytest

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.tools import TOOL_REGISTRY
from agentic_code_bench.solutions.challenge_003_multiple_independent_tools import solve

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "challenges"
    / "003_multiple_independent_tools.json"
)


def call_key(call: dict[str, Any]) -> tuple[str, str]:
    """Create an order-independent key without accepting swapped arguments."""
    arguments = call["arguments"]
    return call["tool"], repr(sorted(arguments.items()))


def test_challenge_003_multiple_independent_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = load_challenge(CHALLENGE_PATH)["expected"]
    recorded_calls: list[dict[str, Any]] = []

    def record_calls(
        name: str, function: Callable[..., object]
    ) -> Callable[..., object]:
        @wraps(function)
        def recorded(**arguments: Any) -> object:
            call: dict[str, Any] = {"tool": name, "arguments": arguments}
            recorded_calls.append(call)
            result = function(**arguments)
            call["result"] = result
            return result

        return recorded

    for name, function in TOOL_REGISTRY.items():
        monkeypatch.setitem(TOOL_REGISTRY, name, record_calls(name, function))

    actual = solve()

    assert isinstance(actual, dict)
    assert set(actual) == {"calls", "call_count"}
    assert type(actual["call_count"]) is int
    assert isinstance(actual["calls"], list)
    assert actual["call_count"] == len(actual["calls"])
    assert len(recorded_calls) == expected["call_count"] == actual["call_count"]

    actual_by_key = {call_key(call): call for call in actual["calls"]}
    recorded_by_key = {call_key(call): call for call in recorded_calls}
    expected_by_key = {call_key(call): call for call in expected["calls"]}

    assert len(actual_by_key) == len(actual["calls"])
    assert len(recorded_by_key) == len(recorded_calls)
    assert actual_by_key == recorded_by_key == expected_by_key

    for call in actual["calls"]:
        assert type(call["result"]) is int
