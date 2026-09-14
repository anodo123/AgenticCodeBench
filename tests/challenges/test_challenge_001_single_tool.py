from pathlib import Path

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.solutions.challenge_001_single_tool import solve

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[2] / "challenges" / "001_single_tool.json"
)


def test_challenge_001_single_tool():
    challenge = load_challenge(CHALLENGE_PATH)
    actual = solve()
    expected = challenge["expected"]["output"]

    assert actual == expected
    assert type(actual) is type(expected)
