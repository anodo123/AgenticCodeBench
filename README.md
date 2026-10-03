# Agentic Code Bench

A hands-on repository for learning agent orchestration through independently
evaluable challenges. Challenges 001 through 006 are implemented. Challenge 007
is not implemented.

| Challenge | Behavior |
| --- | --- |
| 001 | Single-tool execution with integral-float output normalization |
| 002 | Select and execute exactly one tool |
| 003 | Execute multiple independent calls from one model response |
| 004 | Return one tool result to the model for a dependent call |
| 005 | Bounded model-tool loop with a final-message stopping condition |
| 006 | Handle a real tool exception and prevent retry |

Definitions live in `challenges/`, independent solutions in
`src/agentic_code_bench/solutions/`, and live evaluators in `tests/challenges/`.
The existing core provides challenge loading, the OpenAI Responses API model
layer, a single-call execution helper, and eight registered arithmetic tools.
The default model is `gpt-5.6-luna`.

Learners write agent orchestration. Expected answer data belongs only to challenge
definitions and evaluation; solutions never consult it to manufacture results.
Simple generic tools are implemented automatically. Domain-specific or substantial
tools remain learner work. No generic agent runtime is required.

Python 3.12 or newer is required. Run non-billable development checks with:

```powershell
uv sync --extra dev --locked
uv run ruff check .
uv run mypy src
```

The implemented challenge tests are live integration tests and may incur OpenAI
API charges. Ask before running live model tests or the full suite. After explicit
approval, configure `OPENAI_API_KEY` in the environment or `.env` and run a chosen
evaluator, for example:

```powershell
uv run pytest tests/challenges/test_challenge_002_tool_selection.py
```

Challenges 005 and 006 offer readable debugging via `solve(debug=True)` without
printing encrypted reasoning. Challenge 003 can optionally write
`challenge_003_response.json` via `solve(debug=True)`; generated response dumps
are ignored by Git and must not be committed.
