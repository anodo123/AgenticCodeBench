"""Evaluator for Challenge 014's evidence-based incident diagnosis."""

import json
from functools import wraps
from typing import Any

import pytest

from agentic_code_bench.core.challenge import load_challenge
from agentic_code_bench.core.incident_tools import (
    DATABASE_OBSERVATION_ID,
    DOWNSTREAM_DEPENDENCY_ID,
    INCIDENT_ID,
    PROPOSED_ACTION,
    InvalidIncidentContextError,
    InvalidRemediationActionError,
    RemediationPolicyError,
    create_incident_tool_registry,
    evaluate_database_inspection_gate,
)
from agentic_code_bench.solutions import (
    challenge_014_multi_cause_incident_diagnosis as solution,
)


def execute_through_workload_inspection(
    registry: dict[str, Any],
) -> dict[str, Any]:
    registry["get_incident_details"](incident_id=INCIDENT_ID)
    registry["diagnose_deployment_changes"](incident_id=INCIDENT_ID)
    registry["inspect_database_health"](incident_id=INCIDENT_ID)
    return registry["inspect_database_workloads"](incident_id=INCIDENT_ID)


def approve_and_remediate(
    registry: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    execute_through_workload_inspection(registry)
    authorization = registry["record_remediation_authorization"](
        observation_id=DATABASE_OBSERVATION_ID,
        proposed_action=PROPOSED_ACTION,
        decision="approved",
        reviewer_comment="Approved after reviewing the read-only job impact.",
    )
    remediation = registry["execute_workload_remediation"](
        observation_id=DATABASE_OBSERVATION_ID,
        proposed_action=PROPOSED_ACTION,
        authorization_id=authorization["authorization_id"],
    )
    return authorization, remediation


def record_model_tools(
    registry: dict[str, Any],
    model_tool_names: set[str],
    executions: list[dict[str, Any]],
) -> dict[str, Any]:
    recorded = dict(registry)
    for name in model_tool_names:
        function = registry[name]

        def make_wrapper(tool_name: str, tool_function: Any) -> Any:
            @wraps(tool_function)
            def wrapper(**arguments: Any) -> Any:
                evidence = {"tool": tool_name, "arguments": arguments}
                try:
                    result = tool_function(**arguments)
                except Exception as error:
                    evidence.update(
                        status="error",
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                    executions.append(evidence)
                    raise
                evidence.update(status="success", result=result)
                executions.append(evidence)
                return result

            return wrapper

        recorded[name] = make_wrapper(name, function)
    return recorded


def test_incident_environment_requires_application_owned_context() -> None:
    with pytest.raises(InvalidIncidentContextError):
        create_incident_tool_registry("")


def test_deployment_hypothesis_uses_normalized_same_window_comparison() -> None:
    registry = create_incident_tool_registry(INCIDENT_ID)
    registry["get_incident_details"](incident_id=INCIDENT_ID)
    result = registry["diagnose_deployment_changes"](incident_id=INCIDENT_ID)
    assert result["hypothesis_state"] == "rejected"
    assert result["matched_cohort"] == {
        "endpoint": "POST /v1/recommendations",
        "region": "ap-south-1",
        "request_path": "standard-recommendation",
    }
    assert result["versions"]["previous"]["error_rate_percent"] == 3.0
    assert result["versions"]["current"]["error_rate_percent"] == 3.1
    assert result["observations"][0]["relationship"] == "contradicts"


def test_database_health_passes_the_deterministic_inspection_gate() -> None:
    registry = create_incident_tool_registry(INCIDENT_ID)
    registry["get_incident_details"](incident_id=INCIDENT_ID)
    health = registry["inspect_database_health"](incident_id=INCIDENT_ID)
    gate = evaluate_database_inspection_gate(health)
    assert gate["workload_inspection_allowed"] is True
    assert gate["high_latency_window_percent"] == 100
    assert set(gate["checks"].values()) == {"passed"}
    assert health["data_quality"]["coverage_percent"] == 100


def test_workload_inspector_returns_sanitized_causal_evidence() -> None:
    registry = create_incident_tool_registry(INCIDENT_ID)
    result = execute_through_workload_inspection(registry)
    supported = result["observations"][0]
    assert result["hypothesis"]["state"] == "supported"
    assert supported["observation_type"] == "background_job"
    assert supported["observation_id"] == DATABASE_OBSERVATION_ID
    assert supported["operation_type"] == "read"
    assert supported["resource_contribution"]["disk_io_percent"] == 72
    serialized = json.dumps(result).lower()
    assert "raw_sql" not in serialized
    assert "customer_data" not in serialized
    assert "select *" not in serialized


def test_authorization_is_unique_and_bound_to_the_exact_action() -> None:
    registry = create_incident_tool_registry(INCIDENT_ID)
    execute_through_workload_inspection(registry)
    first = registry["record_remediation_authorization"](
        observation_id=DATABASE_OBSERVATION_ID,
        proposed_action=PROPOSED_ACTION,
        decision="approved",
        reviewer_comment="Approve termination.",
    )
    second = registry["record_remediation_authorization"](
        observation_id=DATABASE_OBSERVATION_ID,
        proposed_action=PROPOSED_ACTION,
        decision="approved",
        reviewer_comment="A separate approval record.",
    )
    assert first["authorization_id"] != second["authorization_id"]
    assert first["incident_id"] == INCIDENT_ID
    assert first["observation_id"] == DATABASE_OBSERVATION_ID
    assert first["proposed_action"] == PROPOSED_ACTION
    with pytest.raises(InvalidRemediationActionError):
        registry["execute_workload_remediation"](
            observation_id=DATABASE_OBSERVATION_ID,
            proposed_action="pause",
            authorization_id=first["authorization_id"],
        )


def test_rejected_authorization_cannot_execute_remediation() -> None:
    registry = create_incident_tool_registry(INCIDENT_ID)
    execute_through_workload_inspection(registry)
    authorization = registry["record_remediation_authorization"](
        observation_id=DATABASE_OBSERVATION_ID,
        proposed_action=PROPOSED_ACTION,
        decision="rejected",
        reviewer_comment="Do not interrupt this reporting job.",
    )
    with pytest.raises(RemediationPolicyError):
        registry["execute_workload_remediation"](
            observation_id=DATABASE_OBSERVATION_ID,
            proposed_action=PROPOSED_ACTION,
            authorization_id=authorization["authorization_id"],
        )


def test_recovery_is_partial_and_reopens_downstream_diagnosis() -> None:
    registry = create_incident_tool_registry(INCIDENT_ID)
    _, remediation = approve_and_remediate(registry)
    execution_id = remediation["remediation_execution_id"]
    recovery = registry["verify_remediation_outcome"](
        incident_id=INCIDENT_ID,
        remediation_execution_id=execution_id,
    )
    assert recovery["recovery_state"] == "partial"
    assert (
        recovery["post_remediation_metrics"]["disk_io_wait_ms"]["state"] == "recovered"
    )
    assert (
        recovery["post_remediation_metrics"]["api_latency_p95_ms"]["state"]
        == "unhealthy"
    )
    assert recovery["observation_window_count"] == 3

    downstream = registry["investigate_downstream_dependency_issue"](
        incident_id=INCIDENT_ID,
        remediation_execution_id=execution_id,
    )
    assert downstream["hypothesis"]["state"] == "supported"
    assert (
        downstream["hypothesis"]["supported_dependency_id"] == DOWNSTREAM_DEPENDENCY_ID
    )
    assert downstream["recommended_incident_state"] == "escalated"
    serialized = json.dumps(downstream).lower()
    assert "request_payload" not in serialized
    assert "credentials" not in serialized
    assert "customer_data" not in serialized


def test_challenge_014_live_agent_replans_after_partial_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    challenge = load_challenge(solution.CHALLENGE_PATH)
    executions: list[dict[str, Any]] = []
    real_factory = create_incident_tool_registry
    model_tool_names = {tool["name"] for tool in challenge["tools"]}

    def observed_factory(incident_id: str) -> dict[str, Any]:
        registry = real_factory(incident_id)
        return record_model_tools(registry, model_tool_names, executions)

    monkeypatch.setattr(solution, "create_incident_tool_registry", observed_factory)
    prompts: list[str] = []

    def approve(prompt: str) -> str:
        prompts.append(prompt)
        assert len(executions) == 4
        return "Yes, approve termination of this exact read-only workload."

    actual = solution.solve(debug=True, read_user=approve)
    expected = challenge["expected"]
    assert actual["status"] == expected["status"]
    assert actual["incident_id"] == expected["incident_id"]
    assert actual["steps"] == executions
    assert [step["tool"] for step in executions] == expected["ordered_executed_tools"]
    assert actual["call_count"] == len(executions) == expected["call_count"]
    assert actual["approval_count"] == expected["approval_count"]
    assert (
        actual["remediation_execution_count"] == expected["remediation_execution_count"]
    )
    assert (
        expected["minimum_model_turn_count"]
        <= actual["model_turn_count"]
        <= expected["maximum_model_turn_count"]
    )
    assert (
        actual["deployment_evidence"]["hypothesis_state"]
        == expected["deployment_hypothesis_state"]
    )
    assert (
        actual["workload_evidence"]["hypothesis"]["state"]
        == expected["database_hypothesis_state"]
    )
    assert (
        actual["workload_evidence"]["hypothesis"]["supported_observation_id"]
        == expected["supported_observation_id"]
    )
    assert actual["authorization"]["decision"] == expected["authorization_decision"]
    assert actual["authorization"]["proposed_action"] == expected["approved_action"]
    assert actual["recovery"]["recovery_state"] == expected["recovery_state"]
    assert (
        actual["downstream_evidence"]["hypothesis"]["state"]
        == expected["downstream_hypothesis_state"]
    )
    assert (
        actual["downstream_evidence"]["hypothesis"]["supported_dependency_id"]
        == expected["supported_dependency_id"]
    )
    assert actual["state_history"] == [
        "investigating",
        "awaiting_approval",
        "monitoring_recovery",
        "investigating",
        "escalated",
    ]
    assert len(prompts) == 1
    serialized = json.dumps(actual).lower()
    assert "raw_sql" not in serialized
    assert "customer_data" not in serialized
