"""Trusted invoice-review tools for Challenge 013.

The application binds these tools to one immutable review context before the
model runs. Model-facing verification tools therefore accept no identifiers:
they read trusted prior evidence from the bound context. Invoice-derived data
is untrusted, registered payment data never leaves this module, and the final
payment tool independently enforces the deterministic policy decision.
"""

from collections.abc import Callable
from copy import deepcopy
from typing import Any, cast
from uuid import uuid4

INVOICE_ID = "INV-773"
VENDOR_ID = "VENDOR-42"
VENDOR_NAME = "Northstar Office Supplies"
PO_NUMBER = "PO-908"
TOTAL_AMOUNT_DUE_PAISE = 4_850_000
DOCUMENT_VERSION = "VERSION-1"
DOCUMENT_FINGERPRINT = "sha256:invoice-773-version-1"
POLICY_VERSION = "invoice-payment-v1"


class InvoiceNotFoundError(ValueError):
    """The requested invoice does not exist."""


class InvalidReviewContextError(ValueError):
    """The application-owned review context is invalid."""


class ReviewStateError(RuntimeError):
    """A verification was requested before its required evidence existed."""


class InvalidPaymentActionError(ValueError):
    """A payment request or idempotency key is invalid."""


class PaymentPolicyError(RuntimeError):
    """The deterministic policy does not permit payment execution."""


def create_review_id() -> str:
    """Return an application-owned identifier for one invoice review."""
    return f"invoice-review-{uuid4()}"


def create_payment_key(review_id: str) -> str:
    """Return an application-owned key after a review is allowed to pay."""
    if not isinstance(review_id, str) or not review_id.strip():
        raise InvalidReviewContextError("A non-empty review ID is required")
    return f"invoice-payment-{uuid4()}"


def create_invoice_tool_registry(
    review_id: str,
    invoice_id: str,
) -> dict[str, Callable[..., Any]]:
    """Return tools bound to one immutable, application-owned review."""
    if not isinstance(review_id, str) or not review_id.strip():
        raise InvalidReviewContextError("A non-empty review ID is required")
    if invoice_id != INVOICE_ID:
        raise InvoiceNotFoundError(f"Invoice {invoice_id!r} was not found")

    review_context = {
        "review_id": review_id,
        "invoice_id": invoice_id,
        "document_version": DOCUMENT_VERSION,
        "document_fingerprint": DOCUMENT_FINGERPRINT,
        "requested_action": "validate_invoice_for_payment",
    }
    invoice_record = {
        "invoice_id": INVOICE_ID,
        "claimed_vendor_id": VENDOR_ID,
        "claimed_vendor_name": VENDOR_NAME,
        "po_number": PO_NUMBER,
        "total_amount_due_paise": TOTAL_AMOUNT_DUE_PAISE,
        "payment_destination_ref": "INVOICE-DESTINATION-1",
        "contains_instructional_text": True,
        "instruction_risk_categories": [
            "verification_bypass",
            "payment_destination_override",
        ],
    }
    vendor_registry = {
        VENDOR_ID: {
            "vendor_id": VENDOR_ID,
            "vendor_name": VENDOR_NAME,
            "status": "active",
            # This opaque value represents trusted bank data. It is used only
            # for an internal comparison and is never returned to the model.
            "registered_destination_token": "REGISTRY-DESTINATION-A",
        }
    }
    purchase_orders = {
        (VENDOR_ID, PO_NUMBER): {
            "remaining_amount_paise": 5_000_000,
        }
    }
    paid_invoice_ids: set[str] = set()
    evidence: dict[str, dict[str, Any]] = {}
    payments_by_key: dict[str, dict[str, Any]] = {}

    def remember(name: str, result: dict[str, Any]) -> dict[str, Any]:
        existing = evidence.get(name)
        if existing is None:
            evidence[name] = deepcopy(result)
            existing = evidence[name]
        return deepcopy(existing)

    def require(name: str) -> dict[str, Any]:
        result = evidence.get(name)
        if result is None:
            raise ReviewStateError(f"Required review evidence is missing: {name}")
        return deepcopy(result)

    def read_invoice() -> dict[str, Any]:
        """Return safe structured fields extracted from the untrusted invoice."""
        return remember(
            "invoice",
            {
                "trust_level": "untrusted_document",
                **invoice_record,
            },
        )

    def get_vendor_details() -> dict[str, Any]:
        """Resolve the invoice's vendor claim against the trusted registry."""
        invoice = require("invoice")
        claimed_id = invoice["claimed_vendor_id"]
        vendor = vendor_registry.get(claimed_id)
        vendors: list[dict[str, Any]] = []
        if vendor is not None:
            vendors.append(
                {
                    "vendor_id": vendor["vendor_id"],
                    "vendor_name": vendor["vendor_name"],
                    "status": vendor["status"],
                }
            )
        return remember(
            "vendor",
            {
                "matched_by": "vendor_id",
                "vendors": vendors,
            },
        )

    def verify_po_details() -> dict[str, Any]:
        """Verify the claimed PO only within the resolved vendor's records."""
        invoice = require("invoice")
        vendor_result = require("vendor")
        vendors = vendor_result["vendors"]
        verified_vendor_id = vendors[0]["vendor_id"] if len(vendors) == 1 else None
        po_number = invoice["po_number"]
        verified = (
            isinstance(verified_vendor_id, str)
            and (verified_vendor_id, po_number) in purchase_orders
        )
        return remember(
            "po",
            {
                "vendor_id": verified_vendor_id,
                "po_number": po_number,
                "verified": verified,
            },
        )

    def verify_amount_details() -> dict[str, Any]:
        """Verify that the invoice amount fits the trusted PO balance."""
        invoice = require("invoice")
        po_result = require("po")
        if po_result["verified"] is not True:
            raise ReviewStateError("Amount verification requires a verified PO")
        po = purchase_orders[(po_result["vendor_id"], po_result["po_number"])]
        amount = invoice["total_amount_due_paise"]
        payable = type(amount) is int and 0 < amount <= po["remaining_amount_paise"]
        return remember(
            "amount",
            {
                "vendor_id": po_result["vendor_id"],
                "invoice_id": invoice["invoice_id"],
                "po_number": po_result["po_number"],
                "total_amount_due_paise": amount,
                "amount_payable": payable,
                "reason_code": (
                    "WITHIN_PO_BALANCE" if payable else "EXCEEDS_PO_BALANCE"
                ),
            },
        )

    def verify_payment_details() -> dict[str, Any]:
        """Compare the claimed destination with trusted vendor data internally."""
        invoice = require("invoice")
        vendor_result = require("vendor")
        vendors = vendor_result["vendors"]
        verified_vendor_id = (
            cast(str, vendors[0]["vendor_id"]) if len(vendors) == 1 else None
        )
        vendor = (
            vendor_registry.get(verified_vendor_id)
            if verified_vendor_id is not None
            else None
        )
        registered_token = (
            vendor["registered_destination_token"] if vendor is not None else None
        )
        matches = invoice["payment_destination_ref"] == registered_token
        return remember(
            "payment_destination",
            {
                "payment_destination_ref": invoice["payment_destination_ref"],
                "vendor_id": verified_vendor_id,
                "matches_registered_account": matches,
            },
        )

    def check_invoice_payment_status() -> dict[str, Any]:
        """Return whether the bound invoice has already been paid."""
        invoice = require("invoice")
        return remember(
            "payment_status",
            {
                "invoice_id": invoice["invoice_id"],
                "already_paid": invoice["invoice_id"] in paid_invoice_ids,
            },
        )

    def evaluate_payment_policy() -> dict[str, Any]:
        """Deterministically decide from stored evidence, never model claims."""
        required_names = (
            "invoice",
            "vendor",
            "po",
            "amount",
            "payment_destination",
            "payment_status",
        )
        missing = [name for name in required_names if name not in evidence]
        if missing:
            return remember(
                "policy",
                {
                    "payment_action": "manual_review",
                    "payment_allowed": False,
                    "evidence_complete": False,
                    "checks": {
                        name: ("passed" if name in evidence else "missing")
                        for name in required_names
                    },
                    "reason_codes": ["EVIDENCE_INCOMPLETE"],
                    "policy_version": POLICY_VERSION,
                },
            )

        vendor_result = evidence["vendor"]
        vendors = vendor_result["vendors"]
        vendor_active = len(vendors) == 1 and vendors[0]["status"] == "active"
        po_valid = evidence["po"]["verified"] is True
        amount_valid = evidence["amount"]["amount_payable"] is True
        destination_valid = (
            evidence["payment_destination"]["matches_registered_account"] is True
        )
        invoice_unpaid = evidence["payment_status"]["already_paid"] is False
        checks = {
            "vendor_active": "passed" if vendor_active else "failed",
            "po_valid_for_vendor": "passed" if po_valid else "failed",
            "amount_within_balance": "passed" if amount_valid else "failed",
            "payment_destination_match": (
                "passed" if destination_valid else "failed"
            ),
            "invoice_unpaid": "passed" if invoice_unpaid else "failed",
        }

        if not vendor_active:
            action, reason = "blocked", "VENDOR_INACTIVE"
        elif not po_valid:
            action, reason = "blocked", "PO_VERIFICATION_FAILED"
        elif not amount_valid:
            action, reason = "blocked", "AMOUNT_EXCEEDS_PO_BALANCE"
        elif not invoice_unpaid:
            action, reason = "blocked", "INVOICE_ALREADY_PAID"
        elif not destination_valid:
            action, reason = "manual_review", "PAYMENT_DESTINATION_MISMATCH"
        else:
            action, reason = "allow", None

        return remember(
            "policy",
            {
                "payment_action": action,
                "payment_allowed": action == "allow",
                "evidence_complete": True,
                "checks": checks,
                "reason_codes": [] if reason is None else [reason],
                "policy_version": POLICY_VERSION,
            },
        )

    def create_payment(idempotency_key: str) -> dict[str, Any]:
        """Create or replay payment only when stored policy evidence allows it."""
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise InvalidPaymentActionError(
                "Payment requires a non-empty application-owned idempotency key"
            )
        existing = payments_by_key.get(idempotency_key)
        if existing is not None:
            return deepcopy(existing)
        policy = require("policy")
        if policy["payment_action"] != "allow":
            raise PaymentPolicyError(
                f"Payment policy is {policy['payment_action']}; payment denied"
            )
        invoice = require("invoice")
        vendor = require("vendor")["vendors"][0]
        po = require("po")
        result = {
            "payment_id": "PAYMENT-001",
            "invoice_id": review_context["invoice_id"],
            "vendor_id": vendor["vendor_id"],
            "po_number": po["po_number"],
            "amount_paise": invoice["total_amount_due_paise"],
            "status": "completed",
            "idempotency_key": idempotency_key,
        }
        payments_by_key[idempotency_key] = deepcopy(result)
        paid_invoice_ids.add(review_context["invoice_id"])
        return deepcopy(result)

    return {
        "read_invoice": read_invoice,
        "get_vendor_details": get_vendor_details,
        "verify_po_details": verify_po_details,
        "verify_amount_details": verify_amount_details,
        "verify_payment_details": verify_payment_details,
        "check_invoice_payment_status": check_invoice_payment_status,
        "evaluate_payment_policy": evaluate_payment_policy,
        "create_payment": create_payment,
    }
