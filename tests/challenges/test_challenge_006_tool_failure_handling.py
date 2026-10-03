"""Live integration evaluator: running this test may incur API charges."""

import json
from pathlib import Path

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.tools import TOOL_REGISTRY
from agentic_code_bench.solutions import challenge_006_tool_failure_handling as solution


def test_challenge_006_tool_failure_handling(monkeypatch):
    challenge = load_challenge(
        Path(__file__).resolve().parents[2]
        / "challenges"
        / "006_tool_failure_handling.json"
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
    actual = solution.solve(debug=False)
    assert actual["call_count"] == len(executions) == expected["call_count"]
    assert actual == expected
    assert executions == expected["steps"]
    assert len(requests) == 2
    assert events == ["model", "tool", "model"]
    assert requests[0][0]["messages"] == challenge["messages"]
    assert requests[0][0].get("previous_response_id") is None
    for index, (request, _response) in enumerate(requests[1:], start=1):
        prior_response = requests[index - 1][1]
        calls = [item for item in prior_response.output if item.type == "function_call"]
        assert len(calls) == 1
        assert request["previous_response_id"] == prior_response.id
        execution = executions[index - 1]
        payload = {
            key: execution[key] for key in ("status", "error_type", "error_message")
        }
        assert request["messages"] == [
            {
                "type": "function_call_output",
                "call_id": calls[0].call_id,
                "output": json.dumps(payload),
            }
        ]
    final = requests[-1][1]
    assert not any(item.type == "function_call" for item in final.output)
    assert any(item.type == "message" for item in final.output)
    assert len(requests) <= solution.MAX_MODEL_TURNS
    assert len(executions) == solution.MAX_TOOL_ATTEMPTS == 1
