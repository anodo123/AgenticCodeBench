"""Learner scaffold for Challenge 007: missing information.

This challenge teaches when an agent must not act. The rectangle tool requires a
height, but the user supplies only a width. The learner validates that the model
asks for clarification, executes no tool, and represents the outcome as
``needs_input`` rather than completed or failed.
"""

from pathlib import Path
from typing import Any

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.model import call_openai_model as call_openai_model

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "challenges"
    / "007_missing_information.json"
)


def solve(model: str = "gpt-5.6-luna", debug: bool = True) -> dict[str, Any]:
    """Return a model-generated clarification without executing a tool."""
    def log(message: str) -> None:
        if debug:
            print(f"[challenge_007] {message}", flush=True)

    log(f"Step 1: Loading challenge from {CHALLENGE_PATH}")
    try:
        challenge = load_challenge(CHALLENGE_PATH)
    except Exception as error:
        log(f"Step 1 FAILED: {type(error).__name__}: {error}")
        raise
    log(f"Step 1 OK: Loaded {challenge['id']}; "
        f"messages={len(challenge['messages'])}, tools={len(challenge['tools'])}")

    log(f"Step 2: Sending one model request using {model}")
    try:
        response = call_openai_model(
            messages=challenge["messages"],
            tools=challenge["tools"],
            model=model,
        )
    except Exception as error:
        log(f"Step 2 FAILED: {type(error).__name__}: {error}")
        raise
    log(f"Step 2 OK: Response ID={getattr(response, 'id', None)}, "
        f"status={getattr(response, 'status', None)}")
    log(f"Output types: {[item.type for item in response.output]}")
    log(f"Raw output text: {response.output_text!r}")

    log("Step 3: Separating function calls and message items")
    function_calls = [item for item in response.output if item.type == "function_call"]
    message_calls = [item for item in response.output if item.type == "message"]
    log(f"Step 3 OK: function_calls={len(function_calls)}, messages={len(message_calls)}")

    log("Step 4: Checking that the model requested no tool calls")
    if function_calls:
        log("Step 4 FAILED: Model requested a tool despite the missing height")
        raise RuntimeError("The model must not invent the missing height, and the solution must not execute any tool")
    log("Step 4 OK: No tool calls requested or executed")

    log("Step 5: Checking for a message and non-empty clarification")
    if not message_calls:
        log("Step 5 FAILED: No message item returned")
        raise RuntimeError("The model must return at least one message item.")
    clarification = (response.output_text or "").strip()
    if not clarification:
        log("Step 5 FAILED: Output text is empty or whitespace")
        raise RuntimeError("The model must return non-empty output text.")
    log(f"Step 5 OK: Clarification={clarification!r}")

    log("Step 6: Checking clarification for 'height' or 'tall'")
    matched_keywords = [
        keyword for keyword in ("height", "tall")
        if keyword in clarification.casefold()
    ]
    if not matched_keywords:
        log("Step 6 FAILED: Neither keyword was found")
        raise RuntimeError("The clarification must mention height or tall.")
    log(f"Step 6 OK: Matched keywords={matched_keywords}")

    result = {
        "status": "needs_input",
        "message": clarification,
        "call_count": 0,
        "model_turn_count": 1,
    }
    log("Step 7: Stopped to wait for the missing height; no tool was called")
    log(f"Returning result: {result}")
    return result


if __name__ == "__main__":
    import pytest

    test_path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "challenges"
        / "test_challenge_007_missing_information.py"
    )
    raise SystemExit(pytest.main([str(test_path), "-v"]))
