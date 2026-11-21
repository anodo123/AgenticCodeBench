"""Solution for Challenge 014: multi-cause incident diagnosis.

Question:
Investigate production incident INC-014 without treating correlation as a root
cause. Compare rolling-deployment versions, gate deeper database inspection on
complete health evidence, identify a privacy-safe suspected workload, obtain
human approval bound to one exact remediation, and verify recovery across
multiple windows. When database health recovers but API health does not, reopen
the investigation, test downstream dependencies, and escalate the remaining
supported external cause instead of falsely reporting resolution.

Learning goals:
Maintain competing falsifiable hypotheses, keep authoritative evidence and
approval state outside model-controlled arguments, use deterministic policy for
diagnostic and action gates, separate action completion from system recovery,
and replan after partial recovery.

Output contract:
Return status, message, incident and snapshot IDs, state history, executed
steps, hypothesis evidence, authorization record, remediation and recovery
evidence, downstream evidence, call/approval/remediation/model-turn counts, and
the final remaining symptoms.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.incident_tools import (
    DATABASE_OBSERVATION_ID,
    INCIDENT_ID,
    PROPOSED_ACTION,
    create_incident_tool_registry,
)
from agentic_code_bench.core.model import call_openai_model

CHALLENGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "challenges"
    / "014_multi_cause_incident_diagnosis.json"
)
MAX_MODEL_TURNS = 9
MAX_TOOL_CALLS = 7
MAX_REMEDIATION_EXECUTIONS = 1


def _authorization_decision(reply: str) -> str:
    """Classify a human reply without allowing the model to approve itself."""
    normalized = " ".join(reply.strip().lower().split())
    if any(token in normalized for token in ("reject", "decline", "do not")):
        return "rejected"
    if normalized in {"no", "n"}:
        return "rejected"
    if any(token in normalized for token in ("approve", "approved", "yes")):
        return "approved"
    return "questions_required"


def solve(
    model: str = "gpt-5.6-luna",
    debug: bool = True,
    read_user: Callable[[str], str] = input,
) -> dict[str, Any]:
    """Return the evidence-based incident investigation described above."""

    def log(message: str) -> None:
        if debug:
            print(f"[challenge_014] {message}", flush=True)

    challenge = load_challenge(CHALLENGE_PATH)
    registry = create_incident_tool_registry(INCIDENT_ID)
    allowed_tools = {tool["name"] for tool in challenge["tools"]}
    required_sequence = (
        "get_incident_details",
        "diagnose_deployment_changes",
        "inspect_database_health",
        "inspect_database_workloads",
        "execute_workload_remediation",
        "verify_remediation_outcome",
        "investigate_downstream_dependency_issue",
    )
    messages = challenge["messages"]
    previous_response_id: str | None = None
    steps: list[dict[str, Any]] = []
    state_history = ["investigating"]
    call_count = 0
    approval_count = 0
    remediation_execution_count = 0
    model_turn_count = 0
    incident_details: dict[str, Any] | None = None
    deployment_evidence: dict[str, Any] | None = None
    database_health: dict[str, Any] | None = None
    workload_evidence: dict[str, Any] | None = None
    authorization: dict[str, Any] | None = None
    remediation: dict[str, Any] | None = None
    recovery: dict[str, Any] | None = None
    downstream_evidence: dict[str, Any] | None = None

    def report(status: str, message: str) -> dict[str, Any]:
        snapshot_id = (
            incident_details.get("snapshot_id")
            if incident_details is not None
            else None
        )
        remaining_symptoms = (
            recovery.get("remaining_symptoms", []) if recovery is not None else []
        )
        return {
            "status": status,
            "message": message,
            "incident_id": INCIDENT_ID,
            "snapshot_id": snapshot_id,
            "state_history": state_history,
            "steps": steps,
            "deployment_evidence": deployment_evidence,
            "database_health": database_health,
            "workload_evidence": workload_evidence,
            "authorization": authorization,
            "remediation": remediation,
            "recovery": recovery,
            "downstream_evidence": downstream_evidence,
            "remaining_symptoms": remaining_symptoms,
            "call_count": call_count,
            "approval_count": approval_count,
            "remediation_execution_count": remediation_execution_count,
            "model_turn_count": model_turn_count,
        }

    for _ in range(MAX_MODEL_TURNS):
        log(
            f"Model turn {model_turn_count + 1}/{MAX_MODEL_TURNS}; "
            f"previous_response_id={previous_response_id!r}; messages={messages!r}"
        )
        response = call_openai_model(
            messages=messages,
            tools=challenge["tools"],
            model=model,
            previous_response_id=previous_response_id,
        )
        model_turn_count += 1
        log(
            f"Response id={response.id!r}; status={response.status!r}; "
            f"types={[item.type for item in response.output]!r}; "
            f"output_text={response.output_text!r}"
        )
        if debug:
            log(
                "Full model response:\n"
                + json.dumps(response.model_dump(mode="json"), indent=2)
            )

        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        message_items = [item for item in response.output if item.type == "message"]
        if len(function_calls) > 1:
            raise RuntimeError("Only one tool call is allowed per model response.")

        if function_calls:
            if call_count >= MAX_TOOL_CALLS:
                raise RuntimeError("Tool-call budget exhausted.")
            call = function_calls[0]
            if call.name not in allowed_tools or call.name not in registry:
                raise RuntimeError(f"Unknown or unavailable tool: {call.name}")
            expected_tool = required_sequence[call_count]
            if call.name != expected_tool:
                raise RuntimeError(f"Expected {expected_tool}, received {call.name}.")
            if call.name == "execute_workload_remediation" and authorization is None:
                raise RuntimeError("Remediation requires prior human authorization.")
            try:
                arguments = json.loads(call.arguments)
            except (TypeError, ValueError) as error:
                raise RuntimeError("Tool arguments must be valid JSON.") from error
            if not isinstance(arguments, dict):
                raise RuntimeError("Tool arguments must be a JSON object.")
            log(
                f"Model requested tool={call.name!r}; call_id={call.call_id!r}; "
                f"arguments={arguments!r}"
            )

            if call.name in {
                "get_incident_details",
                "diagnose_deployment_changes",
                "inspect_database_health",
                "inspect_database_workloads",
            }:
                if arguments != {"incident_id": INCIDENT_ID}:
                    raise RuntimeError(f"{call.name} must use the bound incident ID.")
            elif call.name == "execute_workload_remediation":
                if authorization is None:
                    raise RuntimeError("Authorization state is missing.")
                expected_arguments = {
                    "observation_id": DATABASE_OBSERVATION_ID,
                    "proposed_action": PROPOSED_ACTION,
                    "authorization_id": authorization["authorization_id"],
                }
                if arguments != expected_arguments:
                    raise RuntimeError(
                        "Remediation must match the exact human authorization."
                    )
            else:
                if remediation is None:
                    raise RuntimeError("Remediation execution state is missing.")
                expected_arguments = {
                    "incident_id": INCIDENT_ID,
                    "remediation_execution_id": remediation["remediation_execution_id"],
                }
                if arguments != expected_arguments:
                    raise RuntimeError(
                        f"{call.name} must use the verified remediation execution."
                    )

            result = registry[call.name](**arguments)
            call_count += 1
            step = {
                "tool": call.name,
                "arguments": arguments,
                "status": "success",
                "result": result,
            }
            steps.append(step)
            if call.name == "get_incident_details":
                incident_details = result
            elif call.name == "diagnose_deployment_changes":
                deployment_evidence = result
            elif call.name == "inspect_database_health":
                database_health = result
            elif call.name == "inspect_database_workloads":
                workload_evidence = result
            elif call.name == "execute_workload_remediation":
                remediation = result
                remediation_execution_count += 1
                state_history.append("monitoring_recovery")
            elif call.name == "verify_remediation_outcome":
                recovery = result
                if result["recovery_state"] == "partial":
                    state_history.append("investigating")
            else:
                downstream_evidence = result
            log(f"Tool result: {result!r}")

            messages = [
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                }
            ]
            previous_response_id = response.id
            log(
                f"Returning function_call_output for call_id={call.call_id!r}; "
                f"next_previous_response_id={previous_response_id!r}"
            )
            continue

        if not message_items or not (response.output_text or "").strip():
            raise RuntimeError("Expected one tool call or a non-empty message.")
        try:
            decision = json.loads(response.output_text)
        except json.JSONDecodeError as error:
            raise RuntimeError("Malformed response: expected JSON.") from error
        if not isinstance(decision, dict):
            raise RuntimeError("Malformed response: expected a JSON object.")
        status = decision.get("status")
        message = decision.get("message")
        if not isinstance(message, str) or not message.strip():
            raise RuntimeError("Malformed response: message must be non-empty.")
        message = message.strip()
        previous_response_id = response.id

        if status == "needs_approval":
            if (
                workload_evidence is None
                or authorization is not None
                or call_count != 4
                or workload_evidence["hypothesis"]["state"] != "supported"
            ):
                raise RuntimeError("Approval requested in an invalid state.")
            state_history.append("awaiting_approval")
            print(f"Agent: {message}", flush=True)
            reply = read_user(f"Human remediation decision: {message} ").strip()
            if not reply:
                raise RuntimeError("Human remediation decision must not be empty.")
            authorization = registry["record_remediation_authorization"](
                observation_id=DATABASE_OBSERVATION_ID,
                proposed_action=PROPOSED_ACTION,
                decision=_authorization_decision(reply),
                reviewer_comment=reply,
            )
            log(f"Human authorization record: {authorization!r}")
            if authorization["decision"] != "approved":
                state_history.append("awaiting_user_direction")
                final_report = report("awaiting_user_direction", message)
                log(f"Final report:\n{json.dumps(final_report, indent=2)}")
                return final_report
            approval_count += 1
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Human decision recorded by the application: "
                        + json.dumps(authorization)
                        + ". Continue using only this exact authorization."
                    ),
                }
            ]
            continue

        if status != "escalated":
            raise RuntimeError(f"Unexpected terminal status: {status!r}.")
        if (
            call_count != MAX_TOOL_CALLS
            or approval_count != 1
            or remediation_execution_count != MAX_REMEDIATION_EXECUTIONS
            or deployment_evidence is None
            or deployment_evidence["hypothesis_state"] != "rejected"
            or workload_evidence is None
            or workload_evidence["hypothesis"]["state"] != "supported"
            or authorization is None
            or authorization["decision"] != "approved"
            or recovery is None
            or recovery["recovery_state"] != "partial"
            or downstream_evidence is None
            or downstream_evidence["hypothesis"]["state"] != "supported"
            or downstream_evidence["recommended_incident_state"] != "escalated"
        ):
            raise RuntimeError("Escalation requires the complete evidence chain.")
        state_history.append("escalated")
        final_report = report("escalated", message)
        log(f"Final report:\n{json.dumps(final_report, indent=2)}")
        return final_report

    raise RuntimeError("Model-turn budget exhausted.")


if __name__ == "__main__":
    solve(debug=True)
