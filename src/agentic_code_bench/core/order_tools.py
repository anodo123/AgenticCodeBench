"""Deterministic, local order tools for the recovery exercise; no network calls."""

from collections.abc import Callable
from typing import Any


class OrderNotFoundError(ValueError):
    """Permanent failure: the requested order does not exist."""


class TrackingNotFoundError(ValueError):
    """Permanent failure: the requested tracking ID does not exist."""


def create_order_tool_registry() -> dict[str, Callable[..., Any]]:
    """Create isolated tools: the known order times out once per new registry.

    Call once per solve(), not once per model turn. State must survive retries
    within one run and reset between runs.
    """
    order_attempts = 0

    def get_order_status(order_id: str) -> dict[str, str]:
        nonlocal order_attempts
        if order_id != "ORD-1042":
            raise OrderNotFoundError(f"Order {order_id} was not found")
        order_attempts += 1
        if order_attempts == 1:
            raise TimeoutError("Order service timed out temporarily")
        return {
            "order_id": order_id,
            "status": "shipped",
            "tracking_id": "TRK-1042",
        }

    def get_carrier_status(tracking_id: str) -> dict[str, str]:
        if tracking_id != "TRK-1042":
            raise TrackingNotFoundError(f"Tracking ID {tracking_id} was not found")
        return {
            "tracking_id": tracking_id,
            "delivery_status": "out_for_delivery",
        }

    return {
        "get_order_status": get_order_status,
        "get_carrier_status": get_carrier_status,
    }