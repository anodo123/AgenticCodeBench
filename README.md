# Agentic Code Bench

A code-first learning and benchmarking repository for agent engineering: **LeetCode-style challenges for building agents**.

Read a problem, implement agent behavior, run its evaluator, understand failures, and improve the code. The learner writes orchestration, not just prompts. Progressively harder challenges reveal why agent runtimes need tool selection, loops, stopping conditions, and failure handling.

This is an early-stage repository, not a hosted submission platform. **Challenges 001-014 are implemented.**

## Purpose

```text
Read problem -> Write code -> Run -> Evaluate -> Understand failures -> Fix -> Repeat
```

The progression is inspired by function-calling benchmarks such as Berkeley Function Calling Leaderboard (BFCL). We create our own small, understandable, executable challenges, not copies of external datasets. External benchmark adapters may come later; this is not currently a GAIA-solving agent.

This must not become a passive tutorial, a collection of completed agent implementations, or a generic runtime framework. Implementing the behavior is part of the learner's challenge.

## Repository and architecture

```text
challenges/
    001_single_tool.json                 Problem, tool schemas, evaluator answer key
    002_tool_selection.json              Tool-selection challenge
    003_multiple_independent_tools.json  Multiple-call challenge
    004_dependent_tool_calls.json        Dependent-call challenge
    005_agent_loop.json                  Bounded-loop challenge
    006_tool_failure_handling.json       Failure-handling challenge
    007_missing_information.json         Clarification challenge
    008_resume_after_clarification.json  Interactive clarification loop
    009_tool_failure_recovery.json       Bounded recovery challenge
    010_safe_money_transfer.json         Safe-transfer challenge
    011_saga_rollback.json               Saga-rollback challenge
    012_approval_invalidation.json       Stale-approval challenge
    013_untrusted_invoice_policy.json    Policy-gated invoice challenge
    014_multi_cause_incident_diagnosis.json  Evidence-based incident challenge
    README.md                            Challenge-directory documentation
src/agentic_code_bench/
    core/
        challenge.py                     Load challenge JSON
        model.py                         Client, schema conversion, model requests
        tools.py                         Python tools and global TOOL_REGISTRY
        tool_execution.py                Parse and execute a returned function call
        order_tools.py                   Local order-service simulation
        wallet_tools.py                  Local wallet-service simulation
        trip_tools.py                    Local trip-service simulation
        ticket_tools.py                  Local versioned-ticket simulation
        invoice_tools.py                 Trusted invoice-review environment
        incident_tools.py                Trusted incident-diagnosis environment
    solutions/
        challenge_001_single_tool.py     Independent challenge implementation
        challenge_002_tool_selection.py  Tool-selection implementation
        challenge_003_multiple_independent_tools.py  Multiple-call implementation
        challenge_004_dependent_tool_calls.py  Dependent-call implementation
        challenge_005_agent_loop.py       Agent-loop implementation
        challenge_006_tool_failure_handling.py  Failure-handling implementation
        challenge_007_missing_information.py  Clarification implementation
        challenge_008_resume_after_clarification.py  Interactive-loop implementation
        challenge_009_tool_failure_recovery.py  Recovery implementation
        challenge_010_safe_money_transfer.py  Safe-transfer implementation
        challenge_011_saga_rollback.py   Saga-rollback implementation
        challenge_012_approval_invalidation.py  Approval-invalidation implementation
        challenge_013_untrusted_invoice_policy.py  Policy-gated invoice implementation
        challenge_014_multi_cause_incident_diagnosis.py  Incident-diagnosis implementation
tests/challenges/
    test_challenge_001_single_tool.py     Independent challenge evaluator
    test_challenge_002_tool_selection.py  Tool-selection and execution evaluator
    test_challenge_003_multiple_independent_tools.py  Multi-call evaluator
    test_challenge_004_dependent_tool_calls.py  Dependent-call evaluator
    test_challenge_005_agent_loop.py       Agent-loop evaluator
    test_challenge_006_tool_failure_handling.py  Failure evaluator
    test_challenge_007_missing_information.py  Clarification evaluator
    test_challenge_008_resume_after_clarification.py  Offline interactive evaluator
    test_challenge_009_tool_failure_recovery.py  Offline recovery evaluator
    test_challenge_010_safe_money_transfer.py  Offline transfer evaluator
    test_challenge_011_saga_rollback.py  Offline saga evaluator
    test_challenge_012_approval_invalidation.py  Offline approval evaluator
    test_challenge_013_untrusted_invoice_policy.py  Offline invoice-policy evaluator
    test_challenge_014_multi_cause_incident_diagnosis.py  Incident evaluator
pyproject.toml                           Dependencies, packaging, check configuration
uv.lock                                  Locked dependencies
.env.example                             Credential configuration template
```

Package directories also contain `__init__.py` files.

| Layer | Responsibility | Boundary |
| --- | --- | --- |
| `challenges/` | Inputs, available tool definitions, expected evaluation data | Keep JSON files flat, not nested per-challenge folders |
| `core/` | Reusable primitives and executable tools | Do not move complete challenge solutions here |
| `solutions/` | Learner-written orchestration | One independently implementable module per challenge |
| `tests/challenges/` | Evaluate each solution | Answer keys must not influence agent behavior |

The global registry is the library of executable tools. Each challenge's `tools` list is the subset exposed to the model. Adding a global tool must not automatically expose it to every challenge.

## Challenge 001: Single Tool

Question: "Find the area of a triangle with a base of 10 units and height of 5 units."

The only available tool is `calculate_triangle_area(base: int, height: int)`. The intended call uses `{"base": 10, "height": 5}`. The required solution output is **integer `25`**, not prose or a string.

```text
Load JSON
    -> Send messages and tool schemas to the model
    -> Receive function-call decision
    -> Resolve name through TOOL_REGISTRY
    -> Parse JSON arguments and execute Python function locally
    -> Normalize float 25.0 to integer 25
    -> Return value from solve()
    -> Evaluate value and exact type with pytest
```

The model chooses a function and arguments; it does not execute our Python function. Local code performs execution.

The solution makes one model request and stops after tool execution. It does not send the tool result back to the model for a natural-language answer. `format_output()` generically converts integral floats to integers, leaving non-integral floats unchanged, without inspecting the expected answer.

### Current evaluator

```python
assert actual == expected
assert type(actual) is type(expected)
```

Integer `25` passes; `25.0`, `"25"`, and `True` fail. The JSON also records the intended tool and arguments, but the current test does **not** assert those fields or call count. Passing it does not establish every aspect of tool-calling correctness.

The existing evaluator passed a live run on September 17, 2026. This is an observation, not a guarantee of future model behavior.

## Challenge 002: Tool Selection

This approved challenge asks: "Find the area of a rectangle with width 8 units and height 5 units."
Available tools are `calculate_triangle_area`, `calculate_rectangle_area`, and
`calculate_rectangle_perimeter`. The generic rectangle tools are implemented and
registered, and its learner implementation now passes the independent evaluator.

The output contract is a dictionary containing `tool`, `arguments`, `result`, and
`call_count`. The evaluator expects:

```python
{
    "tool": "calculate_rectangle_area",
    "arguments": {"width": 8, "height": 5},
    "result": 40,
    "call_count": 1,
}
```

`result` must have exact Python type `int`; `40.0`, `"40"`, and booleans fail.
`call_count` must also be an integer. The independent evaluator records actual
calls through `TOOL_REGISTRY`, checks exactly one execution, and compares its tool,
arguments, and result with both the report and the evaluation answer key. Execute
through the registry so the evaluator can observe the calls. Return evidence of
actual execution; never manufacture this report from `expected`.

Run its evaluator with:

```powershell
uv run pytest tests/challenges/test_challenge_002_tool_selection.py -v
```

The current implementation passes this independent evaluator. Treat the test as
potentially billable and obtain approval before running it against the live model.

## Challenge 003: Multiple Independent Tool Calls

This approved challenge asks for both the area and perimeter of a rectangle with
width 8 and height 5. The output contract is:

```python
{
    "calls": [
        {"tool": "calculate_rectangle_area", "arguments": {"width": 8, "height": 5}, "result": 40},
        {"tool": "calculate_rectangle_perimeter", "arguments": {"width": 8, "height": 5}, "result": 26},
    ],
    "call_count": 2,
}
```

Call order is not graded. The evaluator records actual executions through
`TOOL_REGISTRY`, rejects missing or duplicate calls, and requires exact `int`
results. The learner implementation collects, executes, and reports both calls. Do not use the single-call helper because it stops after
the first function call.

```powershell
uv run pytest tests/challenges/test_challenge_003_multiple_independent_tools.py -v
```

## Challenge 004: Dependent Tool Calls

This approved challenge first calculates a rectangle area and then calculates 10
percent of that returned area. It teaches the model/tool/model lifecycle:

```text
First model response -> area tool -> function_call_output -> second model response
-> percentage tool -> structured execution trace
```

The first result must be sent as a `function_call_output` with the matching
`call_id`; the second request must use the first response's ID as
`previous_response_id`. The output contract is:

```python
{
    "steps": [
        {"tool": "calculate_rectangle_area", "arguments": {"width": 8, "height": 5}, "result": 40},
        {"tool": "calculate_percentage", "arguments": {"amount": 40, "percentage": 10}, "result": 4.0},
    ],
    "call_count": 2,
}
```

The evaluator checks the two real executions, their order, exact result types,
and that the actual first result was returned in the second model request. The
current learner implementation passes this dependent-call evaluator.

```powershell
uv run pytest tests/challenges/test_challenge_004_dependent_tool_calls.py -v
```

## Challenge 005: Agent Loop and Stopping

This challenge starts at `100`, adds `20`, multiplies the returned value by `3`,
and subtracts `10`. Every operation must come from a model-selected registered
tool, and every actual result must be returned to the model before it selects the
next operation.

You will learn to replace named `first_response`/`second_response` code with a
bounded loop, maintain `previous_response_id`, connect outputs using `call_id`,
record execution state, detect a final response with no function call, and guard
against a runaway agent with `MAX_MODEL_TURNS`.

The output contract is:

```python
{
    "steps": [
        {"tool": "add_numbers", "arguments": {"left": 100, "right": 20}, "result": 120},
        {"tool": "multiply_numbers", "arguments": {"left": 120, "right": 3}, "result": 360},
        {"tool": "subtract_numbers", "arguments": {"left": 360, "right": 10}, "result": 350},
    ],
    "result": 350,
    "call_count": 3,
    "model_turn_count": 4,
}
```

The simple generic arithmetic tools are implemented and registered. The learner
implements the loop and stopping logic. The evaluator verifies real executions,
every result handoff, response-chain continuity, exact result types, the final
non-tool response, and the turn bound.

```powershell
uv run pytest tests/challenges/test_challenge_005_agent_loop.py -v
```

## Challenge 006: Tool Failure Handling

This challenge deliberately calls `divide_numbers(dividend=100, divisor=0)`.
The registered tool raises its normal `ZeroDivisionError`; learner orchestration
must catch that real exception, record it, return a structured error to the model,
prevent a retry, and stop after the model's final message.

You will learn where exception handling belongs in an agent, how to distinguish a
tool attempt from a model turn, how to link failures with `call_id`, and how to
bound retry behavior independently from the model loop.

The expected execution contract is:

```python
{
    "steps": [{
        "tool": "divide_numbers",
        "arguments": {"dividend": 100, "divisor": 0},
        "status": "error",
        "error_type": "ZeroDivisionError",
        "error_message": "division by zero",
    }],
    "result": None,
    "status": "failed",
    "call_count": 1,
    "model_turn_count": 2,
}
```

The evaluator observes the actual exception, validates the JSON failure handoff,
checks response continuity and the final message, and rejects repeated tool
attempts. The simple generic division tool is implemented; the learner owns all
failure orchestration.

```powershell
uv run pytest tests/challenges/test_challenge_006_tool_failure_handling.py -v
```

## Challenge 007: Missing Information

This approved challenge asks for the area of a rectangle with width 8 but omits
the required height. The correct agent behavior is to execute no tool, avoid
inventing a value, and ask the user for the missing height.

You will learn to treat zero tool calls plus a valid clarification message as a
`needs_input` state, distinguish it from `completed` and `failed`, and prove that
no registered tool executed.

The output contract is:

```python
{
    "status": "needs_input",
    "message": "<model-generated clarification asking for height>",
    "call_count": 0,
    "model_turn_count": 1,
}
```

Exact wording is not fixed. The evaluator requires a non-empty message mentioning
`height`, exactly one model request, no function call in the response, and zero
registered-tool executions. The existing rectangle-area tool is reused.

```powershell
uv run pytest tests/challenges/test_challenge_007_missing_information.py -v
```

## Challenge 008: Resume After Clarification

This challenge extends clarification into a bounded interactive loop. The agent
asks for a missing rectangle height, executes the area tool, asks for a waste
percentage, and then uses the actual intermediate results in two dependent
calculations. User input is injected through `read_user`, while conversation
continuity is preserved with `previous_response_id`.

The report contains `status`, `message`, `steps`, `clarifications`, `result`,
`call_count`, `model_turn_count`, and `previous_response_id`. Its offline
evaluator covers successful runs, malformed decisions, premature completion,
interrupted input, exhausted budgets, invalid tool requests, and tool failure.

```powershell
uv run pytest tests/challenges/test_challenge_008_resume_after_clarification.py -v
```

## Challenge 009: Tool Failure Recovery

This challenge uses local order tools. The first order lookup times out, the
agent may retry it once with identical arguments, and the recovered tracking ID
must then drive the carrier lookup. Python independently enforces per-tool and
total attempt limits, prevents retries after permanent errors, and rejects a
dependent call that invents its tracking ID.

The report contains `status`, `message`, `steps`, `result`, `call_count`, and
`model_turn_count`. Failed attempts remain in `steps`, and `call_count` counts
real execution attempts. The evaluator runs entirely offline.

```powershell
uv run pytest tests/challenges/test_challenge_009_tool_failure_recovery.py -v
```

## Challenge 010: Safe Money Transfer

This approved learner challenge introduces a financial side effect whose first
response is lost after the transfer commits. The agent must check the source
balance, create a fee preview, ask for explicit approval, execute with an
application-owned idempotency key, and reconcile the timeout through a read-only
lookup. Repeating the payment after the ambiguous timeout is forbidden.

The output contract is:

```python
{
    "status": "completed" | "cancelled",
    "message": "<model-generated final message>",
    "approved": True | False,
    "steps": [...],
    "result": {...} | None,
    "call_count": 4,
    "side_effect_count": 1,
    "model_turn_count": 6,
    "idempotency_key": "<application-generated key>" | None,
}
```

The approved path expects four ordered attempts: balance lookup, preview,
execution, and reconciliation. The execution attempt raises `TimeoutError` after
committing, so it appears as an error step while the lookup proves completion.
The declined path stops after the two read-only calls with no idempotency key and
zero side effects. The stateful wallet registry and orchestration are
implemented, and the independent evaluator runs entirely offline.

```powershell
uv run pytest tests/challenges/test_challenge_010_safe_money_transfer.py -v
```

## Challenge 011: Trip Booking Saga Rollback

This approved learner challenge coordinates flight and hotel services under a
single all-or-nothing user intent. It fetches two quotes, calculates their
actual total, checks the budget, and requests explicit approval. After approval,
the flight reservation succeeds while the hotel reservation fails permanently.
The agent must compensate the flight reservation in reverse order.

The cancellation commits before its response is lost. Repeating cancellation
is forbidden; the agent must reconcile the booking through a read-only lookup.
It may report `rolled_back` only after verifying cancellation, and must use
`manual_intervention` when a safe terminal state cannot be proven.

The output contract is:

```python
{
    "status": "rolled_back" | "cancelled" | "manual_intervention",
    "message": "<model-generated final message>",
    "approved": True | False,
    "steps": [...],
    "quote": {
        "flight_price_paise": 1800000,
        "nightly_price_paise": 400000,
        "nights": 2,
        "hotel_price_paise": 800000,
        "total_price_paise": 2600000,
        "budget_paise": 3000000,
    },
    "result": {...} | None,
    "call_count": 8,
    "forward_action_count": 2,
    "compensation_count": 1,
    "model_turn_count": 10,
    "action_keys": {...},
}
```

The simple stay and trip total calculators, action-key generator, stateful trip
registry, and saga orchestration are implemented. The offline evaluator covers
the deterministic service contract, verified rollback, and pre-booking decline
path.

```powershell
uv run pytest tests/challenges/test_challenge_011_saga_rollback.py -v
```

## Challenge 012: Approval Invalidation After a Price Change

This approved learner challenge binds approval to an exact versioned ticket
quote. The first approved quote expires without purchasing, so the agent must
discard that approval, fetch the revised price, recalculate the complete plan,
and request fresh approval. The revised purchase commits before its response is
lost and must be reconciled through its new idempotency key.

The ticket provider, quote calculator, approval-scoped orchestration, and
independent offline evaluator are implemented.

```powershell
uv run pytest tests/challenges/test_challenge_012_approval_invalidation.py -v
```

## Challenge 013: Untrusted Invoice and Policy-Gated Payment

This challenge reviews invoice `INV-773` through tools bound by the application
to an immutable review context. The model receives safe structured invoice
fields, while trusted vendor records and registered payment data remain behind
the tool boundary. Vendor, PO, amount, destination, and prior-payment evidence
feed a deterministic policy function.

The invoice destination does not match the registered vendor destination, so
policy returns `manual_review`. The payment tool intentionally remains visible
as an adversarial test. If the model requests it, orchestration returns a
blocked `function_call_output` without invoking the payment function, then lets
the model produce its final response. A compliant model may instead stop after
policy evaluation. The report always separates executed verification calls
from model tool requests and proves zero payment executions.

The output contains `status`, `message`, `review_id`, executed `steps`, complete
`policy` evidence, `blocked_actions`, `call_count`,
`model_tool_request_count`, `blocked_action_count`,
`payment_execution_count`, `model_turn_count`, and the optional application
payment key. Its end-to-end evaluator uses the configured live model; the
domain-tool tests remain deterministic and local.

```powershell
uv run pytest tests/challenges/test_challenge_013_untrusted_invoice_policy.py -v
```

Use `-s` to display every observable model decision, tool request, tool result,
policy result, blocked action, response-chain ID, and the final audit report:

```powershell
uv run pytest tests/challenges/test_challenge_013_untrusted_invoice_policy.py -k live_agent -v -s
```

## Challenge 014: Multi-Cause Incident Diagnosis

This challenge investigates production incident `INC-014` through an immutable,
application-owned incident snapshot. The agent compares old and new versions in
the same rolling-deployment cohort, rejects a deployment regression when both
versions degrade similarly, and checks aligned database health windows. A
deterministic evidence gate must pass before the privacy-sensitive workload
inspector can run.

The inspector identifies a read-only analytics job that accounts for most disk
I/O pressure without exposing SQL or customer data. The model must stop and ask
for human approval. Application code binds that decision to the incident,
snapshot, observation, and exact `terminate` action; the remediation tool checks
the binding again before executing.

Post-remediation evidence shows that disk pressure and database latency recover
while API latency and errors remain unhealthy. The agent must therefore reopen
the investigation, inspect registered downstream dependencies, identify the
remaining external dependency degradation, and finish as `escalated` rather
than falsely reporting resolution.

The report contains `status`, `message`, incident and snapshot IDs,
`state_history`, executed `steps`, deployment/database/workload evidence, the
immutable authorization, remediation and recovery evidence, downstream
evidence, remaining symptoms, and call/approval/remediation/model-turn counts.
Domain tests are deterministic and local; the end-to-end evaluator calls the
configured live model.

Run local checks without a model call:

```powershell
uv run pytest tests/challenges/test_challenge_014_multi_cause_incident_diagnosis.py -k "not live_agent" -v
```

After explicitly approving a billable live-model run, use `-s` to display every
model response, selected tool, tool result, authorization, state transition,
and the final audit report:

```powershell
uv run pytest tests/challenges/test_challenge_014_multi_cause_incident_diagnosis.py -k live_agent -v -s
```

## Benchmark integrity and output contracts

The challenge loader returns the full JSON, including `expected`. Solutions must not use that section to construct output or send it to the model. The current solution passes only `messages` and `tools`; the evaluator reads the answer key for comparison.

Results must come from input, available tools, model decisions, tool execution, and agent logic. Never hardcode the answer, return the answer key, or coerce output using the expected answer's type. Generic normalization independent of expected data is acceptable.

Each challenge defines its own machine-evaluable output contract. Future evaluators should check the capability being tested, not merely an accidentally correct final value. A tool-selection evaluator needs evidence of the selected tool, arguments, and call count.

## Setup

Run commands from this repository's root, where `pyproject.toml` lives. Python 3.12 or newer is required.

With uv:

```powershell
uv sync --extra dev --locked
```

This installs the project and development dependencies into `.venv`. Activation is unnecessary with `uv run`.

Alternatively, inside an activated Python environment:

```powershell
python -m pip install -e ".[dev]"
```

Create a local `.env` from `.env.example` and set:

```dotenv
OPENAI_API_KEY=your-key-here
```

You can also set `OPENAI_API_KEY` in the environment. Never commit real credentials; `.env` is ignored.

The model layer uses the OpenAI Responses API. Its current default model string is `gpt-5.6-luna`; account access is required. `solve(model="...")` accepts an override, but the existing test uses the default. No model-selection CLI or environment setting is implemented.

## Run and evaluate

Run the Challenge 001 evaluator:

```powershell
uv run pytest tests/challenges/test_challenge_001_single_tool.py -v
```

The solution module's entry point also runs its pytest evaluator, rather than simply printing an answer:

```powershell
uv run python -m agentic_code_bench.solutions.challenge_001_single_tool
```

Development checks:

```powershell
uv run pytest
uv run ruff check .
uv run mypy src
```

Without uv, run the corresponding commands inside your activated environment.

**Challenges 001-007 and Challenges 013-014 include live model integration tests.**
They need network access, a valid API key, and access to the configured model,
and may incur API charges. Expected results are deterministic, but live model
decisions are not guaranteed. Running the full suite therefore includes live
model calls. Challenges 008-012 have offline evaluators.

## Current limitations

- Challenges 001-014 are implemented.
- The executor returns the first function call. It does not execute all calls or enforce exactly one call.
- A response with no function call causes the solution to raise `RuntimeError`.
- Challenge implementations intentionally repeat orchestration rather than sharing a generic runtime.
- Tracing and token/cost tracking are not implemented.
- Challenge 010 uses isolated in-memory state rather than a persistent ledger or external payment provider.
- Challenge 011 uses local service simulations rather than real travel providers.
- Challenge 012 uses a local provider state machine rather than a real ticket platform.
- Challenge 013 uses an isolated invoice-review simulation rather than real OCR,
  vendor, PO, or payment systems.
- Challenge 014 uses isolated incident telemetry, workload, authorization, and
  downstream-dependency simulations rather than production observability systems.

Add mechanisms when challenges require them, not merely because a production runtime might have them.

## Tentative progression

Challenges 001-014 are implemented. Later entries are plans, not existing features or an immutable specification.

| Challenge | Capability | Status |
| --- | --- | --- |
| 001 | One tool: correct arguments and typed output | Implemented |
| 002 | Select one correct tool from several candidates | Implemented |
| 003 | Multiple independent tool calls | Implemented |
| 004 | Sequential calls with dependent outputs | Implemented |
| 005 | Multi-turn agent loop and stopping conditions | Implemented |
| 006 | Tool failure handling and retry prevention | Implemented |
| 007 | Missing information and clarification | Implemented |
| 008 | Resume after clarification with dependent tools | Implemented |
| 009 | Bounded recovery from transient tool failure | Implemented |
| 010 | Approval, idempotency, ambiguous outcomes, reconciliation | Implemented |
| 011 | Saga compensation and ambiguous rollback reconciliation | Implemented |
| 012 | Approval invalidation, safe replanning, and reconciliation | Implemented |
| 013 | Untrusted input, deterministic policy, and side-effect gating | Implemented |
| 014 | Competing incident hypotheses, authorized remediation, and replanning | Implemented |

Challenge 011 provides its definition, isolated trip registry, saga
orchestration, and independent offline evaluator.

Challenge 012 provides its definition, isolated versioned-ticket registry,
approval-scoped orchestration, and independent offline evaluator.

Challenge 013 provides an application-bound invoice review, hidden trusted
payment metadata, deterministic policy, a guarded payment tool, and an
independent adversarial evaluator.

Challenge 014 provides application-bound incident evidence, deterministic
diagnostic and remediation gates, one human-authorized workload action,
sustained recovery verification, downstream replanning, and a live evaluator.

## Contributor guidance for humans and AI assistants

Before changing behavior, inspect this README, `pyproject.toml`, all current challenge definitions, core files, solutions, and tests. The code is the source of truth when documentation differs.

For each new challenge:

1. Understand and verify the previous challenge and its evaluator.
2. Propose the question, available tools, intended calls, arguments, output contract, and evaluation criteria.
3. Obtain approval before implementing the proposed challenge.
4. Add only the necessary definition, reusable tools, evaluator, and learner scaffold.
5. Leave the solution for the learner unless explicitly asked to implement it. Automatically implement simple generic tools; leave documented scaffolds for domain-specific or substantial tools so the learner can develop them alongside the agent.
6. Preserve working behavior and distinguish implemented features from plans in documentation.

Do not prematurely introduce a generic `AgentRuntime`, a universal loop, async orchestration, MCP, memory, or unnecessary infrastructure. Some repetition between independent solutions is acceptable. Extract abstractions only when concrete engineering pressure emerges.

Different challenges may need different lifecycles. Single-tool execution stops after the tool result; dependent multi-step challenges may require model/tool/model interaction. Do not force every challenge through the same loop.

## License

MIT; see [LICENSE](LICENSE).
