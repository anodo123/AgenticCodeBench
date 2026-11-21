"""Domain tools for Challenge 012's versioned ticket purchase.

Event EVT-901 begins with QUOTE-001 at 500000 paise per ticket. Its first
purchase attempt permanently expires that quote without buying tickets and
publishes QUOTE-002 at 550000 paise. Purchasing against QUOTE-002 commits before
raising TimeoutError. Replaying either attempt with its original idempotency key
must reproduce the stored outcome without repeating the operation.
"""

from collections.abc import Callable
from typing import Any, cast
from uuid import uuid4

EVENT_ID = "EVT-901"
INITIAL_QUOTE_ID = "QUOTE-001"
REVISED_QUOTE_ID = "QUOTE-002"
INITIAL_UNIT_PRICE_PAISE = 500_000
REVISED_UNIT_PRICE_PAISE = 550_000
FEE_PER_TICKET_PAISE = 30_000
TICKET_BUDGET_PAISE = 1_800_000


class EventNotFoundError(ValueError):
    """The requested event does not exist."""


class QuoteNotFoundError(ValueError):
    """The requested quote does not exist."""


class QuoteExpiredError(RuntimeError):
    """The exact approved quote expired before purchase."""


class InvalidQuantityError(ValueError):
    """The requested ticket quantity is invalid."""


class InvalidTicketActionError(ValueError):
    """The requested ticket action or idempotency key is invalid."""


class PurchaseNotFoundError(ValueError):
    """No committed purchase exists for the requested key."""


def create_purchase_key(quote_id: str) -> str:
    """Return a fresh application-owned key for one approved quote."""
    return f"purchase-{quote_id.lower()}-{uuid4()}"


def calculate_ticket_total(
    unit_price_paise: int,
    quantity: int,
) -> dict[str, int]:
    """Return subtotal, fixed ticket fees, and final total in paise."""
    if type(unit_price_paise) is not int or unit_price_paise < 0:
        raise ValueError("Unit price must be a non-negative integer")
    if type(quantity) is not int or quantity <= 0:
        raise InvalidQuantityError("Quantity must be a positive integer")
    subtotal_paise = unit_price_paise * quantity
    fee_paise = FEE_PER_TICKET_PAISE * quantity
    return {
        "unit_price_paise": unit_price_paise,
        "quantity": quantity,
        "subtotal_paise": subtotal_paise,
        "fee_paise": fee_paise,
        "total_price_paise": subtotal_paise + fee_paise,
    }


def create_ticket_tool_registry() -> dict[str, Callable[..., Any]]:
    """Return an isolated versioned-quote and purchase simulation."""
    quotes_by_id: dict[str, dict[str, Any]] = {
        INITIAL_QUOTE_ID: {
            "event_id": EVENT_ID,
            "quote_id": INITIAL_QUOTE_ID,
            "unit_price_paise": INITIAL_UNIT_PRICE_PAISE,
            "status": "current",
        }
    }
    current_quote_id = INITIAL_QUOTE_ID
    attempts_by_key: dict[str, dict[str, Any]] = {}
    purchase_sequence = 0

    def publish_revised_quote() -> None:
        nonlocal current_quote_id
        quotes_by_id[INITIAL_QUOTE_ID]["status"] = "expired"
        quotes_by_id[REVISED_QUOTE_ID] = {
            "event_id": EVENT_ID,
            "quote_id": REVISED_QUOTE_ID,
            "unit_price_paise": REVISED_UNIT_PRICE_PAISE,
            "status": "current",
        }
        current_quote_id = REVISED_QUOTE_ID

    def get_ticket_quote(event_id: str) -> dict[str, Any]:
        """Return the provider's current quote for one event."""
        if event_id != EVENT_ID:
            raise EventNotFoundError(f"Event {event_id} was not found")
        quote = quotes_by_id[current_quote_id]
        return {
            "event_id": quote["event_id"],
            "quote_id": quote["quote_id"],
            "unit_price_paise": quote["unit_price_paise"],
        }

    def purchase_tickets(
        quote_id: str,
        quantity: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Purchase or replay tickets against one immutable request key."""
        nonlocal purchase_sequence
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise InvalidTicketActionError(
                "A ticket purchase requires a non-empty idempotency key"
            )

        existing = attempts_by_key.get(idempotency_key)
        if existing is not None:
            if (
                existing["quote_id"] != quote_id
                or existing["quantity"] != quantity
            ):
                raise InvalidTicketActionError(
                    f"Idempotency key {idempotency_key!r} belongs to quote "
                    f"{existing['quote_id']!r} and quantity "
                    f"{existing['quantity']!r}"
                )
            if existing["outcome"] == "quote_expired":
                raise QuoteExpiredError(
                    f"Quote {quote_id} expired before purchase"
                )
            return cast(dict[str, Any], existing["result"]).copy()

        quote = quotes_by_id.get(quote_id)
        if quote is None:
            raise QuoteNotFoundError(f"Quote {quote_id} was not found")
        if type(quantity) is not int or quantity <= 0:
            raise InvalidQuantityError("Quantity must be a positive integer")

        if quote["status"] != "current" or quote_id == INITIAL_QUOTE_ID:
            if quote_id == INITIAL_QUOTE_ID and current_quote_id == INITIAL_QUOTE_ID:
                publish_revised_quote()
            attempts_by_key[idempotency_key] = {
                "quote_id": quote_id,
                "quantity": quantity,
                "outcome": "quote_expired",
            }
            raise QuoteExpiredError(f"Quote {quote_id} expired before purchase")

        totals = calculate_ticket_total(
            unit_price_paise=quote["unit_price_paise"],
            quantity=quantity,
        )
        purchase_sequence += 1
        result = {
            "purchase_id": f"PURCHASE-{purchase_sequence:03d}",
            "event_id": quote["event_id"],
            "quote_id": quote_id,
            "quantity": quantity,
            "unit_price_paise": totals["unit_price_paise"],
            "subtotal_paise": totals["subtotal_paise"],
            "fee_paise": totals["fee_paise"],
            "total_price_paise": totals["total_price_paise"],
            "status": "completed",
            "idempotency_key": idempotency_key,
        }
        attempts_by_key[idempotency_key] = {
            "quote_id": quote_id,
            "quantity": quantity,
            "outcome": "completed",
            "result": result.copy(),
        }
        raise TimeoutError(
            f"Purchase response timed out for idempotency key {idempotency_key}"
        )

    def lookup_purchase_by_idempotency_key(
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Return the committed purchase stored under an idempotency key."""
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise InvalidTicketActionError(
                "Purchase lookup requires a non-empty idempotency key"
            )
        attempt = attempts_by_key.get(idempotency_key)
        if attempt is None or attempt["outcome"] != "completed":
            raise PurchaseNotFoundError(
                f"No committed purchase exists for key {idempotency_key}"
            )
        return cast(dict[str, Any], attempt["result"]).copy()

    return {
        "get_ticket_quote": get_ticket_quote,
        "calculate_ticket_total": calculate_ticket_total,
        "purchase_tickets": purchase_tickets,
        "lookup_purchase_by_idempotency_key": (
            lookup_purchase_by_idempotency_key
        ),
    }
