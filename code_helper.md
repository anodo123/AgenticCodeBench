# Agentic Code Bench Continuation Prompt

Use this document as the complete handoff prompt when continuing this work in a
different Codex setup. Treat the repository as an existing project with a long
learning history. Do not recreate it as a new project.

## Workspace

The expected repository root is:

```text
C:\Users\AhmadFaiz\Downloads\Learning\AgenticCodeBench-main\AgenticCodeBench-main
```

At the beginning of a new session:

1. Confirm that the current workspace is this directory.
2. Read `README.md`, `challenges/README.md`, `pyproject.toml`, and any
   `AGENTS.md` instructions.
3. Inspect the current challenge definitions, core tools, solutions, and tests
   before editing.
4. Treat the checked-out files as the source of truth if this handoff differs
   from the repository.

This copy of the repository was downloaded without a `.git` directory. Commands
such as `git status` report that it is not a Git repository. Make direct file
changes and report them precisely. Do not assume that commits, branches, or
diffs are available.

The project uses Python 3.12 or newer, `uv`, pytest, Ruff, mypy, the OpenAI
Responses API, and a `src/` package layout. The currently configured default
model in the solution modules is `gpt-5.6-luna`. Never reveal or copy secrets
from `.env`.

## Project Purpose

Agentic Code Bench is a progressive, code-first environment for learning agent
engineering. Each challenge adds one meaningful capability. The learner studies
the architecture, makes important design decisions, and then asks Codex to
implement the agreed design. The goal is to become an expert in agentic concepts
and production workflow design, not merely to test whether a model can answer a
prompt.

The repository intentionally keeps independent challenge implementations. Do
not introduce a universal runtime, framework, MCP layer, or abstraction unless a
specific advanced challenge creates real engineering pressure for it.

The main layers are:

```text
challenges/                     Challenge definitions and evaluator-only answers
src/agentic_code_bench/core/    Reusable tools and local domain environments
src/agentic_code_bench/solutions/ Independent challenge orchestration
tests/challenges/               Deterministic and live evaluators
```

The `expected` section of a challenge JSON belongs only to evaluation. A
solution must never read it, send it to the model, or manufacture output from
it. Results must come from model decisions, actual tool executions,
application-owned state, and deterministic policy.

## Collaboration and Teaching Style

Follow these rules throughout the continuation:

1. Communicate in English unless the user explicitly asks otherwise. The user
   may write Hindi or mixed Hindi-English; answer clearly and warmly.
2. Explain the challenge and obtain approval before finalizing or implementing
   it. The user explicitly corrected the previous assistant for choosing a
   challenge too quickly.
3. Do not jump directly into implementation. First discuss the scenario,
   learning objective, important tool contracts, state transitions, safety
   boundaries, output contract, and evaluator behavior.
4. Ask the user meaningful architecture and orchestration questions. Examples:
   - How many meaningful states or cases can this function enter?
   - What arguments should the tool accept, and why?
   - Which data belongs to the model and which belongs to application-owned
     state?
   - What evidence authorizes a transition or side effect?
   - How should partial recovery, ambiguity, or failure change the plan?
5. When designing a tool, discuss its name, number and purpose of arguments,
   return structure, important failure states, trust boundary, and side effects.
   Do not silently assume function arguments.
6. Skip trivial questions that do not provide learning value, such as arbitrary
   constant names or routine identifier spelling. Make reasonable choices for
   those details.
7. If the user's first answer is incomplete, provide a conceptual hint and let
   them retry. Do not immediately reveal the full answer. Explain the concept
   directly when the user says they lack the required domain knowledge.
8. The user no longer considers hand-typing implementation the main learning
   activity. Their learning comes from giving correct architectural guidance,
   reasoning about flows, and reviewing generated code. Implement the agreed
   design after the design discussion is complete and the user asks for it.
9. Automatically implement small generic tools. Discuss substantial,
   domain-specific tools and their contracts before implementation.
10. Every new solution file must begin with a module docstring containing the
    actual challenge question, learning goals, and output contract.
11. Boilerplate must include the basic imports required to run it.
12. Keep debugging detailed for new live-agent challenges. With `debug=True`,
    log every model turn, response ID, previous response ID, complete observable
    model response, selected tool, arguments, call ID, tool result, state
    transition, authorization, blocked action, and final report.
13. New end-to-end agent tests should call the real configured model. Do not mock
    `call_openai_model` for the live path. Local deterministic tests for the
    domain environment are expected and should not call the model.
14. Ask before running a live model test because it may consume billable tokens.
    Run deterministic checks without a model call whenever possible.
15. Report what changed, what was verified, and what remains for the user.

## Learning Principles Established So Far

Preserve these concepts in later challenges:

- The model selects tools; application code executes them.
- A tool schema being present in a response means the tool was available. An
  actual call is represented by a `function_call` item and recorded execution
  evidence.
- Tool output should return to the model with the matching `call_id` and
  `previous_response_id` when another model decision depends on it.
- The absence of a function call is only a stopping signal when the returned
  text is valid for the current state.
- Agent loops require independent bounds for model turns, tool calls, retries,
  and irreversible side effects.
- Application code owns immutable identity, evidence provenance, authorization,
  idempotency keys, policy decisions, and side-effect limits.
- A model may propose an action, but deterministic policy decides whether it is
  permitted.
- Tool completion does not prove business or system recovery. Recovery needs a
  separate read-only verification step.
- Timing overlap establishes correlation. Stronger causal evidence requires
  aligned resource contribution and the predicted change after remediation.
- Partial recovery must reopen surviving hypotheses. It must not be reported as
  complete resolution.
- Sensitive data, raw SQL, untrusted instructions, credentials, and customer
  data should remain behind tool boundaries. The model receives sanitized
  evidence and immutable evidence references.

## Completed Challenges

Challenges 001 through 014 are implemented. Preserve their behavior.

### Challenge 001: Single Tool

Select and execute the triangle-area tool and return an exact integer result.

### Challenge 002: Tool Selection

Choose rectangle area from triangle area, rectangle area, and rectangle
perimeter. Return executed tool, arguments, result, and call count.

### Challenge 003: Multiple Independent Tools

Call rectangle area and perimeter, execute both, and return their evidence.

### Challenge 004: Dependent Tool Calls

Calculate rectangle area, return the actual result to the model, and calculate a
percentage from that result.

### Challenge 005: Agent Loop

Execute a bounded sequence of dependent arithmetic tools and stop only after a
valid final model response.

### Challenge 006: Tool Failure Handling

Catch a real divide-by-zero failure, return structured failure evidence to the
model, prevent an unsafe retry, and stop truthfully.

### Challenge 007: Missing Information

Ask for the missing rectangle height without inventing it or executing a tool.

### Challenge 008: Resume After Clarification

Resume a bounded interactive flow after multiple user clarifications and use the
supplied values in dependent calculations.

### Challenge 009: Tool Failure Recovery

Recover from one transient order-service timeout under strict retry limits and
prevent retries after permanent failures.

### Challenge 010: Safe Money Transfer

Bind approval and an application-generated idempotency key to a money transfer.
When execution commits before timing out, reconcile through a read-only lookup
instead of repeating the side effect.

### Challenge 011: Saga Rollback

Coordinate flight and hotel reservations. When the second action fails,
compensate the first action in reverse order and reconcile an ambiguous
cancellation before reporting rollback.

### Challenge 012: Approval Invalidation

Invalidate approval when a ticket quote changes, obtain fresh approval for the
new immutable plan, use a fresh idempotency key, and reconcile the committed
purchase after a lost response.

### Challenge 013: Untrusted Invoice Policy

Review structured but untrusted invoice fields against application-owned vendor,
PO, amount, destination, and payment-status evidence. A deterministic policy
returns `manual_review` for the destination mismatch. The visible payment tool
cannot execute because both orchestration and the tool independently enforce the
policy.

### Challenge 014: Multi-Cause Incident Diagnosis

Challenge 014 is fully implemented in:

```text
challenges/014_multi_cause_incident_diagnosis.json
src/agentic_code_bench/core/incident_tools.py
src/agentic_code_bench/solutions/challenge_014_multi_cause_incident_diagnosis.py
tests/challenges/test_challenge_014_multi_cause_incident_diagnosis.py
```

It investigates production incident `INC-014` using an immutable,
application-owned incident snapshot. The live flow is:

```text
get_incident_details
    -> diagnose_deployment_changes
    -> inspect_database_health
    -> deterministic database inspection gate
    -> inspect_database_workloads
    -> model requests human approval
    -> application records immutable authorization
    -> execute_workload_remediation
    -> verify_remediation_outcome
    -> partial recovery
    -> investigate_downstream_dependency_issue
    -> escalated
```

The deployment hypothesis is rejected because old and new versions degrade
similarly in the same rolling-deployment window and matched traffic cohort.

Database health shows elevated query latency and disk I/O pressure. The deeper
workload inspector is allowed only after deterministic checks confirm sufficient
coverage, acceptable missing-window gaps, trusted metric references, aligned
database/API timing, high p95 query latency in at least 70 percent of windows,
and at least one breached pressure indicator.

The workload inspector returns sanitized evidence. It identifies
`OBS-JOB-014`, a read-only analytics background job with 72 percent disk I/O
contribution. It does not expose raw SQL or customer data.

Human authorization is uniquely bound to:

```text
incident ID
incident snapshot ID
observation ID
exact proposed action
decision
reviewer comment
creation and expiry times
```

Approval to terminate the specific workload cannot authorize pause, throttle,
or an action on another observation.

After termination, three recovery windows show recovered disk I/O and database
query latency, while API latency and error rate remain outside the healthy
baseline. Recovery is therefore `partial`. The agent reopens investigation and
finds that `bedrock-runtime` explains most remaining endpoint wait time. With no
authorized local remediation for the external dependency, the final state is
`escalated`.

The expected state history is:

```text
investigating
-> awaiting_approval
-> monitoring_recovery
-> investigating
-> escalated
```

The final execution has seven model-selected tool calls, one human approval,
one remediation execution, and nine model turns.

Verification already completed:

- Seven deterministic Challenge 014 tests passed.
- An offline regression across Challenges 008 through 014 passed 92 tests, with
  the two live-agent tests deselected.
- Ruff formatting and lint passed for the new files.
- Strict mypy passed for the Challenge 014 source modules.
- The Challenge 014 live-agent test passed: one test passed and seven local
  tests were deselected in that invocation.

The live test command is:

```powershell
uv run pytest tests/challenges/test_challenge_014_multi_cause_incident_diagnosis.py -k live_agent -v -s
```

The `-s` output includes complete response dumps, tool schemas, tool calls,
results, authorization, recovery evidence, state history, token usage, and the
final audit report. The model is live; the incident systems and telemetry are
deterministic local simulations.

## Finalized Roadmap for Challenges 015–024

The user approved this roadmap as the working plan. Keep the sequence unless the
user explicitly revises it. Each individual challenge still requires detailed
discussion and approval before implementation.

| Challenge | Expert concept |
| --- | --- |
| 015 | Cost-aware planning: choose diagnostics by information value, cost, risk, and remaining budget |
| 016 | Dynamic tool discovery: find and load relevant tools from a large catalog without placing every tool in model context |
| 017 | Concurrent orchestration: independent work, deadlines, cancellation, and partial results |
| 018 | Durable agent workflows: checkpoints, crash recovery, delayed events, and exactly-once side effects |
| 019 | Scoped multi-agent delegation: specialist agents with restricted tools, context, and budgets |
| 020 | Multi-agent coordination: conflicting conclusions and shared-state version conflicts |
| 021 | Agent memory and context engineering: provenance, stale-fact invalidation, safe summarization, and retrieval |
| 022 | Dynamic authorization and agent security: temporary capabilities, revocation, confused-deputy prevention, and adversarial tool output |
| 023 | Agent evaluation engineering: trace-based evals, counterfactual replay, failure classification, routing, and regression measurement |
| 024 | Autonomous production-change capstone: validated planning, approval, canary execution, monitoring, rollback, and post-action evaluation |

The progression is:

```text
planning
-> tool discovery
-> orchestration
-> durability
-> delegation
-> coordination
-> memory
-> security
-> evaluation
-> end-to-end autonomy
```

## Current Position: Challenge 015 Design

Work is paused during the conceptual design of Challenge 015. No Challenge 015
files have been created. Do not implement it until the design discussion is
complete and the user explicitly asks for implementation.

The provisional title is:

```text
Challenge 015: Cost-Aware Diagnostic Planning
```

The scenario is an online shopping checkout service with an 18 percent failure
rate. A checkout service coordinates inventory validation, database state,
payment processing, feature flags, order creation, and purchase confirmation
when a customer places an order.

The currently plausible hypotheses are:

- **H1:** Database lock contention
- **H2:** Payment-provider throttling
- **H3:** Inventory-service degradation
- **H4:** Feature-flag misconfiguration

The agent has six investigation credits. The current diagnostic catalog is:

| Diagnostic | Cost | Hypotheses it can investigate |
| --- | ---: | --- |
| Distributed request traces | 3 credits | H1, H2, and H3 |
| Database lock metrics | 2 credits | H1 |
| Payment-provider metrics | 2 credits | H2 |
| Inventory-service metrics | 2 credits | H3 |
| Feature-flag audit | 1 credit | H4 |

Running every diagnostic would cost ten credits, so the agent must select tools
adaptively. After each result, it must update the competing hypotheses, account
for the remaining budget, avoid equivalent or low-value evidence, and decide
whether another diagnostic is worth its cost.

Potential terminal states are currently:

```text
supported
inconclusive
escalated
```

Any remediation will remain separately approval-gated. The diagnostic budget
must be enforced by application code, not trusted to model arithmetic. Evidence
provenance and tool costs should also be application-owned.

Duration was initially included in the diagnostic metadata. The user said the
duration concept was unclear. After discussion, duration was removed from
Challenge 015 so this challenge can focus on credit-constrained information
value. Deadlines, duration, cancellation, and concurrency belong to Challenge
017.

The last conceptual explanation defined a checkout service and its dependencies.
The user then paused the work with: `we halt here sire`.

The exact next learning question, when the user resumes, is:

> Given only that checkout failures increased to 18 percent and the diagnostic
> catalog above, which diagnostic should the agent call first, and why does it
> provide the best reduction in uncertainty for its credit cost?

Do not ask the user to guess the actual root cause. Ask them to reason about the
diagnostic's ability to distinguish multiple plausible hypotheses. Let the user
attempt the answer before explaining the full value-of-information reasoning.

The distributed trace diagnostic appears broad because it can distinguish H1,
H2, and H3, while focused metrics test only one hypothesis. This observation is
guidance for evaluating the user's reasoning, not a reason to reveal the answer
immediately. The feature-flag audit is cheap and may also be valuable depending
on prior probability and evidence quality. Challenge design should eventually
make those tradeoffs explicit so the evaluator does not encode an arbitrary
"optimal" call.

## Next Challenge After 015

Challenge 016 is planned as dynamic tool discovery. Its purpose is to teach an
agent how to find and load a small relevant subset from a large capability
catalog. It should cover tool metadata, schema retrieval, trust, context cost,
and deterministic access control. Do not begin Challenge 016 until Challenge
015 is complete and the user asks to move forward.

## Expected Challenge 015 Design Work Still Remaining

When the user resumes, guide them through the following substantive decisions
without turning each routine detail into a question:

1. Decide how the agent estimates diagnostic information value without claiming
   fake mathematical precision.
2. Define the diagnostic catalog metadata visible before execution: cost,
   hypotheses tested, scope, freshness, evidence source, and operational risk.
3. Decide which diagnostic results are returned for the fixed scenario,
   including at least one noisy or conflicting observation.
4. Define the application-owned budget ledger and the behavior of an
   over-budget tool request.
5. Define a structured hypothesis state and how evidence supports,
   contradicts, or leaves it unresolved.
6. Define when evidence is strong enough for `supported`, when the agent must
   return `inconclusive`, and when escalation is appropriate.
7. Define whether the agent may request remediation and what independent policy
   threshold and human approval are required.
8. Agree on the report/output contract and evaluator invariants.
9. Only after approval, implement the challenge JSON, substantial domain
   environment, solution orchestration, deterministic tests, one real-model
   end-to-end test with full logs, and both README updates.

The design must preserve genuine agent choice. Avoid hard-coding one strict tool
sequence like Challenge 014. The evaluator should judge budget compliance,
evidence quality, truthful state transitions, and acceptable diagnostic paths.
Multiple efficient paths may be valid if they produce sufficient evidence
within budget.

## Verification Rules for Future Work

For new challenges:

1. Run Ruff formatting and lint on changed Python files.
2. Run strict mypy on changed source modules.
3. Run deterministic tests while excluding live-agent tests.
4. Run an appropriate offline regression across recent advanced challenges.
5. Do not run the live-model evaluator until the user explicitly approves the
   billable call.
6. When approved, run the live test with `-s` so the complete observable trace is
   visible.

Useful command pattern:

```powershell
uv run pytest tests/challenges/test_challenge_015_*.py -k "not live_agent" -v
uv run pytest tests/challenges/test_challenge_015_*.py -k live_agent -v -s
```

Adjust the exact filename after Challenge 015 is named and implemented.

## Resume Instruction to the Next Codex

Start by telling the user that the handoff was understood and that work is
paused at the first Challenge 015 value-of-information decision. Briefly restate
the checkout scenario only if needed. Continue with one meaningful design
question at a time. Preserve the user's opportunity to reason, correct
incomplete answers with hints, and wait for explicit implementation approval
after the design becomes concrete and reviewable.
