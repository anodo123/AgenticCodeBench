"""Trusted incident-diagnosis environment for Challenge 014.

The application binds every tool to one immutable incident context. Model-facing
arguments identify records but never carry authoritative incident evidence.
Diagnostic tools return sanitized measurements and immutable evidence references;
raw SQL, log text, request payloads, credentials, and customer data stay behind
the tool boundary. Remediation requires a separate application-owned approval
record bound to the incident, observation, snapshot, and exact action.
"""

from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

INCIDENT_ID = "INC-014"
INCIDENT_SNAPSHOT_ID = "SNAPSHOT-INC-014-V1"
DATABASE_OBSERVATION_ID = "OBS-JOB-014"
DATABASE_WORKLOAD_ID = "workload-fingerprint-analytics-014"
DOWNSTREAM_DEPENDENCY_ID = "bedrock-runtime"
PROPOSED_ACTION = "terminate"


class IncidentNotFoundError(ValueError):
    """The requested incident does not exist."""


class InvalidIncidentContextError(ValueError):
    """The application-owned incident context is invalid."""


class IncidentStateError(RuntimeError):
    """A diagnostic or action was requested before its prerequisites existed."""


class InvalidRemediationActionError(ValueError):
    """A remediation request does not match its immutable authorization."""


class RemediationPolicyError(RuntimeError):
    """Deterministic policy does not permit the requested remediation."""


def create_authorization_id() -> str:
    """Return an application-owned identifier for one remediation decision."""
    return f"remediation-authorization-{uuid4()}"


def evaluate_database_inspection_gate(
    health: dict[str, Any],
) -> dict[str, Any]:
    """Deterministically decide whether deeper workload inspection is allowed."""
    windows = health.get("windows")
    baseline = health.get("healthy_baseline")
    quality = health.get("data_quality")
    timing = health.get("timing")
    if not isinstance(windows, list) or not windows:
        return {
            "workload_inspection_allowed": False,
            "checks": {"measurement_windows_present": "failed"},
            "reason_codes": ["DATABASE_EVIDENCE_INCOMPLETE"],
        }
    if not isinstance(baseline, dict) or not isinstance(quality, dict):
        return {
            "workload_inspection_allowed": False,
            "checks": {"baseline_and_quality_present": "failed"},
            "reason_codes": ["DATABASE_EVIDENCE_INCOMPLETE"],
        }

    latency_bound = baseline["query_latency_p95_ms"]["normal_upper_bound"]
    disk_bound = baseline["disk_io_wait_ms"]["normal_upper_bound"]
    lock_bound = baseline["lock_wait_ms"]["normal_upper_bound"]
    queue_bound = baseline["query_queue_depth"]["normal_upper_bound"]
    high_latency_windows = sum(
        1
        for window in windows
        if window["query_latency_p95_ms"] > 500
        and window["query_latency_p95_ms"] > latency_bound
    )
    high_latency_percent = high_latency_windows * 100 / len(windows)
    pressure_breach = any(
        window["disk_io_wait_ms"] > disk_bound
        or window["lock_wait_ms"] > lock_bound
        or window["connection_utilization_percent"] > 90
        or window["query_queue_depth"] > queue_bound
        for window in windows
    )
    coverage_ok = quality.get("coverage_percent", 0) >= 90
    gaps_ok = quality.get("maximum_consecutive_missing_windows", 99) <= 2
    timing_ok = isinstance(timing, dict) and timing.get("aligned_with_api") is True
    evidence_refs_ok = all(
        isinstance(window.get("evidence_refs"), list) and bool(window["evidence_refs"])
        for window in windows
    )
    checks = {
        "query_latency_high_in_70_percent_of_windows": (
            "passed" if high_latency_percent >= 70 else "failed"
        ),
        "pressure_indicator_breached": "passed" if pressure_breach else "failed",
        "database_and_api_timing_aligned": "passed" if timing_ok else "failed",
        "coverage_at_least_90_percent": "passed" if coverage_ok else "failed",
        "maximum_missing_gap_within_two_windows": ("passed" if gaps_ok else "failed"),
        "trusted_metric_references_present": (
            "passed" if evidence_refs_ok else "failed"
        ),
    }
    allowed = all(value == "passed" for value in checks.values())
    return {
        "workload_inspection_allowed": allowed,
        "high_latency_window_percent": high_latency_percent,
        "checks": checks,
        "reason_codes": [] if allowed else ["DATABASE_INSPECTION_GATE_FAILED"],
    }


def create_incident_tool_registry(
    incident_id: str,
) -> dict[str, Callable[..., Any]]:
    """Return diagnostic and remediation tools bound to one incident snapshot."""
    if not isinstance(incident_id, str) or not incident_id.strip():
        raise InvalidIncidentContextError("A non-empty incident ID is required")
    if incident_id != INCIDENT_ID:
        raise IncidentNotFoundError(f"Incident {incident_id!r} was not found")

    evidence: dict[str, dict[str, Any]] = {}
    authorizations: dict[str, dict[str, Any]] = {}
    executions_by_authorization: dict[str, dict[str, Any]] = {}
    stopped_workloads: set[str] = set()

    def remember(name: str, result: dict[str, Any]) -> dict[str, Any]:
        existing = evidence.get(name)
        if existing is None:
            evidence[name] = deepcopy(result)
            existing = evidence[name]
        return deepcopy(existing)

    def require(name: str) -> dict[str, Any]:
        result = evidence.get(name)
        if result is None:
            raise IncidentStateError(f"Required incident evidence is missing: {name}")
        return deepcopy(result)

    def validate_incident(requested_incident_id: str) -> None:
        if requested_incident_id != incident_id:
            raise InvalidIncidentContextError(
                "The requested incident does not match the bound context"
            )

    def get_incident_details(incident_id: str) -> dict[str, Any]:
        """Return symptoms and a historical healthy baseline for the incident."""
        validate_incident(incident_id)
        return remember(
            "incident",
            {
                "incident_id": incident_id,
                "snapshot_id": INCIDENT_SNAPSHOT_ID,
                "state": "investigating",
                "service": "recommendations-api",
                "region": "ap-south-1",
                "incident_started_at": "2026-11-10T10:00:00Z",
                "affected_endpoints": ["POST /v1/recommendations"],
                "healthy_baseline": {
                    "selection_policy": (
                        "matching time windows from previous healthy days; "
                        "known incidents excluded"
                    ),
                    "aggregation": "median",
                    "sample_count": 14,
                    "api_latency_p95_ms": {
                        "median": 240,
                        "normal_upper_bound": 400,
                    },
                    "api_error_rate_percent": {
                        "median": 0.4,
                        "normal_upper_bound": 1.0,
                    },
                },
                "current_symptoms": {
                    "api_latency_p95_ms": 1_480,
                    "api_error_rate_percent": 3.2,
                    "request_rate_per_minute": 1_020,
                },
                "slo": {
                    "api_latency_p95_ms": 500,
                    "api_error_rate_percent": 1.0,
                },
                "evidence_refs": [
                    "metric://api-latency/INC-014",
                    "metric://api-errors/INC-014",
                    "alert://recommendations-api/INC-014",
                ],
            },
        )

    def diagnose_deployment_changes(incident_id: str) -> dict[str, Any]:
        """Compare old and new versions in the same rolling-deployment window."""
        validate_incident(incident_id)
        require("incident")
        return remember(
            "deployment",
            {
                "incident_id": incident_id,
                "snapshot_id": INCIDENT_SNAPSHOT_ID,
                "hypothesis_type": "deployment_regression",
                "comparison_window": {
                    "start": "2026-11-10T10:00:00Z",
                    "end": "2026-11-10T10:20:00Z",
                },
                "matched_cohort": {
                    "endpoint": "POST /v1/recommendations",
                    "region": "ap-south-1",
                    "request_path": "standard-recommendation",
                },
                "versions": {
                    "previous": {
                        "version": "2026.11.09.4",
                        "request_count": 9_000,
                        "error_count": 270,
                        "error_rate_percent": 3.0,
                        "latency_p95_ms": 1_410,
                    },
                    "current": {
                        "version": "2026.11.10.1",
                        "request_count": 1_000,
                        "error_count": 31,
                        "error_rate_percent": 3.1,
                        "latency_p95_ms": 1_450,
                    },
                },
                "changed_components": [
                    {
                        "component": "recommendation-response-mapper",
                        "deployment_record_id": "DEPLOY-014",
                        "commit_id": "COMMIT-014",
                    }
                ],
                "observations": [
                    {
                        "description": (
                            "Old and new versions degrade similarly in the same "
                            "time window and matched traffic cohort."
                        ),
                        "relationship": "contradicts",
                        "evidence_refs": [
                            "metric://version-comparison/INC-014",
                            "deployment://DEPLOY-014",
                        ],
                    }
                ],
                "hypothesis_state": "rejected",
            },
        )

    def inspect_database_health(incident_id: str) -> dict[str, Any]:
        """Return aligned database health metrics without query contents."""
        validate_incident(incident_id)
        require("incident")
        windows = []
        values = (
            (680, 910, 78, 18, 76, 16),
            (720, 960, 82, 20, 79, 18),
            (700, 940, 80, 19, 81, 17),
            (740, 990, 84, 22, 83, 19),
            (710, 950, 81, 21, 80, 18),
        )
        for index, value in enumerate(values):
            minute = index * 2
            windows.append(
                {
                    "start": f"2026-11-10T10:{minute:02d}:00Z",
                    "end": f"2026-11-10T10:{minute + 2:02d}:00Z",
                    "query_latency_p95_ms": value[0],
                    "query_latency_p99_ms": value[1],
                    "disk_io_wait_ms": value[2],
                    "lock_wait_ms": value[3],
                    "connection_utilization_percent": value[4],
                    "query_queue_depth": value[5],
                    "evidence_refs": [f"metric://database/window-{index + 1}/INC-014"],
                }
            )
        result = {
            "incident_id": incident_id,
            "snapshot_id": INCIDENT_SNAPSHOT_ID,
            "observation_period": {
                "start": "2026-11-10T10:00:00Z",
                "end": "2026-11-10T10:10:00Z",
                "window_duration_seconds": 120,
                "expected_window_count": 5,
            },
            "data_quality": {
                "received_window_count": 5,
                "coverage_percent": 100,
                "maximum_consecutive_missing_windows": 0,
            },
            "healthy_baseline": {
                "query_latency_p95_ms": {
                    "median": 115,
                    "normal_upper_bound": 200,
                },
                "disk_io_wait_ms": {"median": 12, "normal_upper_bound": 30},
                "lock_wait_ms": {"median": 3, "normal_upper_bound": 10},
                "query_queue_depth": {"median": 2, "normal_upper_bound": 6},
            },
            "timing": {
                "api_degradation_started_at": "2026-11-10T10:00:00Z",
                "database_degradation_started_at": "2026-11-10T10:01:00Z",
                "aligned_with_api": True,
            },
            "windows": windows,
        }
        result["workload_inspection_gate"] = evaluate_database_inspection_gate(result)
        return remember("database_health", result)

    def inspect_database_workloads(incident_id: str) -> dict[str, Any]:
        """Return sanitized suspected workloads after the deterministic gate."""
        validate_incident(incident_id)
        health = require("database_health")
        gate = evaluate_database_inspection_gate(health)
        if gate["workload_inspection_allowed"] is not True:
            raise RemediationPolicyError(
                "Database evidence does not permit workload inspection"
            )
        return remember(
            "database_workloads",
            {
                "incident_id": incident_id,
                "snapshot_id": INCIDENT_SNAPSHOT_ID,
                "inspection_gate": gate,
                "observations": [
                    {
                        "observation_type": "background_job",
                        "observation_id": DATABASE_OBSERVATION_ID,
                        "workload_id": DATABASE_WORKLOAD_ID,
                        "owner_service": "analytics-service",
                        "operation_type": "read",
                        "started_at": "2026-11-10T09:59:30Z",
                        "runtime_seconds": 1_230,
                        "incident_overlap": True,
                        "resource_contribution": {
                            "disk_io_percent": 72,
                            "lock_wait_percent": 8,
                            "connection_usage_percent": 18,
                        },
                        "affected_components": ["recommendations-primary-db"],
                        "relationship": "supports",
                        "evidence_refs": [
                            "metric://workload-fingerprint-analytics-014/io",
                            "trace://database-pressure/OBS-JOB-014",
                        ],
                    },
                    {
                        "observation_type": "slow_query",
                        "observation_id": "OBS-QUERY-014",
                        "workload_id": "workload-fingerprint-query-014",
                        "owner_service": "recommendations-api",
                        "operation_type": "read",
                        "started_at": "2026-11-10T10:04:00Z",
                        "runtime_seconds": 20,
                        "incident_overlap": True,
                        "resource_contribution": {
                            "disk_io_percent": 3,
                            "lock_wait_percent": 1,
                            "connection_usage_percent": 2,
                        },
                        "affected_components": ["recommendations-primary-db"],
                        "relationship": "contradicts",
                        "evidence_refs": ["metric://workload-fingerprint-query-014/io"],
                    },
                ],
                "hypothesis": {
                    "type": "database_workload_pressure",
                    "state": "supported",
                    "supported_observation_id": DATABASE_OBSERVATION_ID,
                    "summary": (
                        "The analytics background job overlaps the incident and "
                        "accounts for most observed disk I/O pressure."
                    ),
                    "next_required_action": "human_authorized_remediation",
                },
            },
        )

    def record_remediation_authorization(
        observation_id: str,
        proposed_action: str,
        decision: str,
        reviewer_comment: str,
    ) -> dict[str, Any]:
        """Store an immutable human decision outside the model tool surface."""
        workloads = require("database_workloads")
        valid_ids = {
            observation["observation_id"] for observation in workloads["observations"]
        }
        if observation_id not in valid_ids:
            raise InvalidRemediationActionError("Unknown workload observation")
        if proposed_action not in {"terminate", "pause", "throttle"}:
            raise InvalidRemediationActionError("Unsupported remediation action")
        if decision not in {"approved", "rejected", "questions_required"}:
            raise InvalidRemediationActionError("Unknown authorization decision")
        if not isinstance(reviewer_comment, str) or not reviewer_comment.strip():
            raise InvalidRemediationActionError("A reviewer comment is required")
        authorization_id = create_authorization_id()
        created_at = datetime.now(UTC)
        record = {
            "authorization_id": authorization_id,
            "incident_id": incident_id,
            "incident_snapshot_id": INCIDENT_SNAPSHOT_ID,
            "observation_id": observation_id,
            "proposed_action": proposed_action,
            "decision": decision,
            "reviewer_comment": reviewer_comment.strip(),
            "created_at": created_at.isoformat(),
            "expires_at": (created_at + timedelta(minutes=15)).isoformat(),
        }
        authorizations[authorization_id] = deepcopy(record)
        return deepcopy(record)

    def execute_workload_remediation(
        observation_id: str,
        proposed_action: str,
        authorization_id: str,
    ) -> dict[str, Any]:
        """Execute only the exact action authorized for the bound observation."""
        require("database_workloads")
        authorization = authorizations.get(authorization_id)
        if authorization is None:
            raise InvalidRemediationActionError("Authorization was not found")
        if authorization["decision"] != "approved":
            raise RemediationPolicyError("Remediation was not approved")
        expected = {
            "incident_id": incident_id,
            "incident_snapshot_id": INCIDENT_SNAPSHOT_ID,
            "observation_id": observation_id,
            "proposed_action": proposed_action,
        }
        if any(authorization.get(key) != value for key, value in expected.items()):
            raise InvalidRemediationActionError(
                "Remediation request does not match its authorization"
            )
        existing = executions_by_authorization.get(authorization_id)
        if existing is not None:
            return deepcopy(existing)
        if proposed_action != PROPOSED_ACTION:
            raise RemediationPolicyError(
                "This simulation supports only the approved termination action"
            )
        stopped_workloads.add(observation_id)
        result = {
            "remediation_execution_id": f"REMEDIATION-{uuid4()}",
            "incident_id": incident_id,
            "observation_id": observation_id,
            "authorization_id": authorization_id,
            "action": proposed_action,
            "execution_status": "completed",
            "workload_state": "terminated",
            "evidence_refs": [f"action://workload/{observation_id}/terminated"],
        }
        executions_by_authorization[authorization_id] = deepcopy(result)
        evidence["remediation"] = deepcopy(result)
        return deepcopy(result)

    def verify_remediation_outcome(
        incident_id: str,
        remediation_execution_id: str,
    ) -> dict[str, Any]:
        """Compare sustained post-action metrics with the healthy baseline."""
        validate_incident(incident_id)
        remediation = require("remediation")
        if remediation["remediation_execution_id"] != remediation_execution_id:
            raise InvalidRemediationActionError("Unknown remediation execution")
        if DATABASE_OBSERVATION_ID not in stopped_workloads:
            raise IncidentStateError("The suspected workload is still running")
        return remember(
            "recovery",
            {
                "incident_id": incident_id,
                "remediation_execution_id": remediation_execution_id,
                "remediation_status": "completed",
                "observation_window_count": 3,
                "post_remediation_metrics": {
                    "disk_io_wait_ms": {
                        "observed_median": 18,
                        "normal_upper_bound": 30,
                        "state": "recovered",
                    },
                    "database_query_latency_p95_ms": {
                        "observed_median": 145,
                        "normal_upper_bound": 200,
                        "state": "recovered",
                    },
                    "api_latency_p95_ms": {
                        "observed_median": 860,
                        "normal_upper_bound": 400,
                        "state": "unhealthy",
                    },
                    "api_error_rate_percent": {
                        "observed_median": 2.4,
                        "normal_upper_bound": 1.0,
                        "state": "unhealthy",
                    },
                },
                "recovery_state": "partial",
                "remaining_symptoms": [
                    "api_latency_above_healthy_bound",
                    "api_error_rate_above_healthy_bound",
                ],
                "next_required_action": "reopen_remaining_hypotheses",
                "evidence_refs": [
                    "metric://post-remediation/database/INC-014",
                    "metric://post-remediation/api/INC-014",
                ],
            },
        )

    def investigate_downstream_dependency_issue(
        incident_id: str,
        remediation_execution_id: str,
    ) -> dict[str, Any]:
        """Inspect registered dependencies after verified partial recovery."""
        validate_incident(incident_id)
        recovery = require("recovery")
        if recovery["remediation_execution_id"] != remediation_execution_id:
            raise InvalidRemediationActionError("Unknown remediation execution")
        if recovery["recovery_state"] != "partial":
            raise IncidentStateError(
                "Downstream diagnosis requires verified partial recovery"
            )
        return remember(
            "downstream",
            {
                "incident_id": incident_id,
                "snapshot_id": INCIDENT_SNAPSHOT_ID,
                "dependency_registry_version": "dependency-registry-v4",
                "observations": [
                    {
                        "dependency_id": DOWNSTREAM_DEPENDENCY_ID,
                        "dependency_type": "third_party_model_api",
                        "affected_endpoint": "POST /v1/recommendations",
                        "on_affected_execution_path": True,
                        "healthy_latency_p95_ms": 210,
                        "healthy_latency_normal_upper_bound_ms": 320,
                        "observed_latency_p95_ms": 710,
                        "observed_failure_rate_percent": 3.8,
                        "endpoint_wait_time_contribution_percent": 68,
                        "unaffected_endpoints_healthy": True,
                        "relationship": "supports",
                        "evidence_refs": [
                            "metric://dependency/bedrock-runtime/INC-014",
                            "trace://dependency-wait/bedrock-runtime/INC-014",
                        ],
                    },
                    {
                        "dependency_id": "s3-config-store",
                        "dependency_type": "cloud_storage",
                        "affected_endpoint": "POST /v1/recommendations",
                        "on_affected_execution_path": True,
                        "healthy_latency_p95_ms": 35,
                        "healthy_latency_normal_upper_bound_ms": 60,
                        "observed_latency_p95_ms": 38,
                        "observed_failure_rate_percent": 0.1,
                        "endpoint_wait_time_contribution_percent": 4,
                        "unaffected_endpoints_healthy": True,
                        "relationship": "contradicts",
                        "evidence_refs": [
                            "metric://dependency/s3-config-store/INC-014"
                        ],
                    },
                ],
                "hypothesis": {
                    "type": "downstream_dependency_degradation",
                    "state": "supported",
                    "supported_dependency_id": DOWNSTREAM_DEPENDENCY_ID,
                    "summary": (
                        "Bedrock runtime degradation explains most remaining "
                        "latency on the affected endpoint."
                    ),
                },
                "recommended_incident_state": "escalated",
                "escalation_reason": (
                    "API health remains outside baseline after local database "
                    "recovery, and the remaining supported cause is external."
                ),
            },
        )

    return {
        "get_incident_details": get_incident_details,
        "diagnose_deployment_changes": diagnose_deployment_changes,
        "inspect_database_health": inspect_database_health,
        "inspect_database_workloads": inspect_database_workloads,
        "record_remediation_authorization": record_remediation_authorization,
        "execute_workload_remediation": execute_workload_remediation,
        "verify_remediation_outcome": verify_remediation_outcome,
        "investigate_downstream_dependency_issue": (
            investigate_downstream_dependency_issue
        ),
    }
