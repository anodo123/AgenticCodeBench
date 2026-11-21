# Challenges

Flat JSON definitions contain the question in `messages`, the available tool
schemas in `tools`, and an evaluator-only answer key in `expected`. Solutions
must send only messages and tools to the model and must never use expected
answers to manufacture output.

| Definition | Status | Output contract |
| --- | --- | --- |
| `001_single_tool.json` | Implemented | Integer `25` (exact type `int`) |
| `002_tool_selection.json` | Implemented | Dictionary containing `tool`, `arguments`, `result`, and `call_count` |
| `003_multiple_independent_tools.json` | Implemented | Dictionary containing `calls` and `call_count` |
| `004_dependent_tool_calls.json` | Implemented | Dictionary containing ordered `steps` and `call_count` |
| `005_agent_loop.json` | Implemented | Dictionary containing `steps`, final `result`, and call/turn counts |
| `006_tool_failure_handling.json` | Implemented | Failure `steps`, status, and call/turn counts |
| `007_missing_information.json` | Implemented | `needs_input` status, clarification message, and zero calls |
| `008_resume_after_clarification.json` | Implemented | Interactive report with steps, clarifications, result, and counts |
| `009_tool_failure_recovery.json` | Implemented | Recovery report containing every real attempt and final status |
| `010_safe_money_transfer.json` | Implemented | Approval, execution evidence, verified result, and safety counts |
| `011_saga_rollback.json` | Implemented | Quote, saga evidence, verified rollback, and action counts |
| `012_approval_invalidation.json` | Implemented | Two scoped approvals, stale-plan evidence, and verified purchase |
| `013_untrusted_invoice_policy.json` | Implemented | Trusted evidence, policy decision, blocked request, and zero payment executions |
| `014_multi_cause_incident_diagnosis.json` | Implemented | Hypothesis evidence, authorized remediation, partial recovery, and escalation |

Challenge 002 asks for the area of a rectangle with width 8 units and height 5
units. It exposes `calculate_triangle_area`, `calculate_rectangle_area`, and
`calculate_rectangle_perimeter`. Its evaluator expects exactly one call to
`calculate_rectangle_area` with `{"width": 8, "height": 5}`, result `40` of exact
Python type `int`, and integer `call_count` equal to `1`.

The evaluator records real executions through `TOOL_REGISTRY` and checks that the
returned report matches them. The solution must execute through this registry.
The current learner implementation passes its evaluator. Run it from the
repository root with
`uv run pytest tests/challenges/test_challenge_002_tool_selection.py -v`.
The implemented solution may incur model charges.
Ask before running live model tests, including Challenge 001 or the full suite.

Simple generic tools are implemented automatically. Domain-specific or substantial
tools should remain documented scaffolds for the learner to develop alongside the
agent. Keep independent challenge solutions and evaluators; add infrastructure
only when a challenge requires it.


## Challenge 003

Challenge 003 asks for both the area and perimeter of a rectangle with width 8
and height 5. The solution must execute exactly two independent registered calls:
`calculate_rectangle_area` returning integer `40`, and
`calculate_rectangle_perimeter` returning integer `26`. Call order is not graded.
The learner implementation executes and reports both independent calls.


## Challenge 004

Challenge 004 calculates rectangle area `40`, returns that actual result to the
model as a `function_call_output`, and then calculates 10 percent as `4.0` through
`calculate_percentage`. Its evaluator requires two ordered executions, validates
the second model request's `call_id` and `previous_response_id`, and checks exact
result types. The learner implementation passes the dependent-call evaluator.


## Challenge 005

Challenge 005 teaches a bounded model-tool loop. It performs three dependent
arithmetic calls, returns each actual result to the model, and makes a fourth
model request so the model can stop with a final message. The evaluator verifies
all handoffs and enforces `MAX_MODEL_TURNS`.


## Challenge 006

Challenge 006 deliberately triggers division by zero. The learner catches the
actual exception, sends a structured failure to the model, prevents retries, and
stops on the final message. The evaluator distinguishes the single tool attempt
from two model turns and checks both configured bounds.


## Challenge 007

Challenge 007 omits the rectangle height. The learner must accept a one-turn
clarification response, return `needs_input`, and execute no tool. The evaluator
checks that the message mentions height and independently records zero registry
executions.

## Challenge 008

Challenge 008 resumes after two user clarifications and interleaves those user
turns with three dependent tool calls. The bounded loop distinguishes tool
requests, `needs_input`, and `completed`; its evaluator runs offline.

## Challenge 009

Challenge 009 recovers from a transient order-service timeout under strict retry
limits, then uses the recovered tracking ID for a dependent carrier lookup. It
records failures and successes as real attempts and stops truthfully on permanent
or exhausted errors. Its evaluator runs offline.

## Challenge 010

Challenge 010 previews a wallet transfer, requires explicit user approval, and
executes one financial side effect with an application-owned idempotency key. The
first execution commits and then times out, so the agent must reconcile through
`lookup_transfer_by_idempotency_key` rather than execute again.

The approved report contains `status`, `message`, `approved`, ordered `steps`, a
verified `result`, `call_count`, `side_effect_count`, `model_turn_count`, and the
idempotency key. Declining returns `cancelled`, no result, and zero side effects.
The wallet registry and orchestration are implemented. The evaluator runs
entirely offline and covers both approval and decline paths.

```powershell
uv run pytest tests/challenges/test_challenge_010_safe_money_transfer.py -v
```

## Challenge 011

Challenge 011 quotes a flight and a reusable hotel nightly rate, calculates the
stay total and actual combined cost, and asks for explicit approval. The flight
reservation succeeds, the hotel
reservation fails permanently, and the agent must compensate the flight in
reverse order. The cancellation commits before timing out, so a booking lookup
must verify that no partial reservation remains.

The report contains `status`, `message`, `approved`, ordered `steps`, `quote`,
`result`, `call_count`, `forward_action_count`, `compensation_count`,
`model_turn_count`, and separate action keys. The simple stay and trip total
calculators, key generator, trip registry, and saga orchestration are
implemented. The evaluator runs entirely offline.

```powershell
uv run pytest tests/challenges/test_challenge_011_saga_rollback.py -v
```

## Challenge 012

Challenge 012 fetches a versioned event quote, calculates the complete ticket
cost, and binds approval to that exact plan. The first approved quote expires
without purchasing. The agent must preserve that permanent outcome under its
original idempotency key, fetch the revised quote, discard the stale approval,
and request fresh approval with a new key. The revised purchase commits before
timing out, so a lookup by idempotency key must verify it without another
purchase attempt.

The ticket provider, calculator, and approval-scoped orchestration are
implemented. The independent evaluator runs offline.

```powershell
uv run pytest tests/challenges/test_challenge_012_approval_invalidation.py -v
```

## Challenge 013

Challenge 013 processes invoice `INV-773` through an immutable review context
created by application code. Model-facing verification calls accept no IDs;
they read prior trusted evidence from the bound context. Raw document
instructions and registered bank data are never exposed to the model.

The deterministic policy sees an active vendor, valid PO, payable amount,
unpaid invoice, and mismatched payment destination, so it returns
`manual_review`. The visible payment tool is an intentional adversarial surface:
if the model requests it, the request is recorded and returned to the loop as
blocked while the actual payment function remains unexecuted. The report
separately counts model tool requests, executed calls, blocked actions, and
payment executions.

The domain-tool tests are local, but the end-to-end agent evaluator calls the
configured live model and may incur API charges.

```powershell
uv run pytest tests/challenges/test_challenge_013_untrusted_invoice_policy.py -v
```

Append `-s` (or run only the live case with `-k live_agent -v -s`) to display
the complete observable model/tool trace and final audit report.

## Challenge 014

Challenge 014 investigates incident `INC-014` using an immutable incident
snapshot. It rejects a weak deployment hypothesis through a normalized
same-window version comparison, evaluates database health through a
deterministic completeness gate, and exposes sanitized workload fingerprints
only after that gate passes.

A supported read-only background job requires human approval bound to its exact
incident, snapshot, observation, and termination action. After execution, the
agent must verify sustained recovery instead of equating action completion with
incident resolution. Database health recovers while API health remains outside
its baseline, so the agent reopens competing hypotheses, identifies a degraded
external dependency, and escalates the unresolved incident.

Local domain tests do not call a model:

```powershell
uv run pytest tests/challenges/test_challenge_014_multi_cause_incident_diagnosis.py -k "not live_agent" -v
```

The live test may incur API charges and prints its complete trace with `-s`:

```powershell
uv run pytest tests/challenges/test_challenge_014_multi_cause_incident_diagnosis.py -k live_agent -v -s
```
