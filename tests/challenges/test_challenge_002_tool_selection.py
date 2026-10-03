"""Live integration evaluator: running this test may incur API charges."""

from pathlib import Path

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.tools import TOOL_REGISTRY
from agentic_code_bench.solutions import challenge_002_tool_selection as solution


def test_challenge_002_tool_selection(monkeypatch):
    challenge = load_challenge(
        Path(__file__).resolve().parents[2] / "challenges" / "002_tool_selection.json"
    )
    expected = challenge["expected"]
    executions = []
    events = []
    requests = []
    original_model = solution.call_openai_model

    def record_model(**kwargs):
        events.append("model")
        response = original_model(**kwargs)
        requests.append((kwargs, response))
        return response

    def wrap(name, function):
        def record(**arguments):
            events.append("tool")
            try:
                result = function(**arguments)
            except Exception as exc:
                executions.append(
                    {
                        "tool": name,
                        "arguments": arguments,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                raise
            executions.append({"tool": name, "arguments": arguments, "result": result})
            return result

        return record

    for name, function in list(TOOL_REGISTRY.items()):
        monkeypatch.setitem(TOOL_REGISTRY, name, wrap(name, function))
    monkeypatch.setattr(solution, "call_openai_model", record_model)
    actual = solution.solve()
    assert actual["call_count"] == len(executions) == expected["call_count"]
    assert len(requests) == 1
    assert actual == expected
    assert executions == [
        {key: expected[key] for key in ("tool", "arguments", "result")}
    ]
    assert type(executions[0]["result"]) is type(expected["result"])
    assert type(actual["result"]) is type(expected["result"])
