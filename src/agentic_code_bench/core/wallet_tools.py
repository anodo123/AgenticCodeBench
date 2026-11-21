"""Deterministic domain tools for Challenge 010's wallet simulation.

Each registry has isolated in-memory state. WALLET-A starts with 500000 paise
and WALLET-B starts with 100000 paise. A preview charges a one-percent fee.
Executing a new idempotency key commits the transfer exactly once and then
raises TimeoutError to simulate a lost response. Looking up that key returns
the committed transfer. Repeating execution with the same key returns the
original record without moving money again.
"""

from collections.abc import Callable
from typing import Any
from uuid import uuid4

INITIAL_WALLET_BALANCES = {
    "WALLET-A": 500_000,
    "WALLET-B": 100_000,
}
TRANSFER_FEE_PERCENT = 1
PREVIEW_ID = "PREVIEW-1042"
TRANSFER_ID = "TRANSFER-1042"


class WalletNotFoundError(ValueError):
    """The requested wallet does not exist."""


class InvalidTransferError(ValueError):
    """The requested transfer is structurally invalid."""


class InsufficientFundsError(ValueError):
    """The source wallet cannot cover the amount and fee."""


class PreviewNotFoundError(ValueError):
    """The requested transfer preview does not exist."""


class TransferNotFoundError(ValueError):
    """No transfer exists for the requested idempotency key."""


def create_idempotency_key() -> str:
    """Return a fresh application-owned transfer idempotency key."""
    return f"transfer-{uuid4()}"


def create_wallet_tool_registry() -> dict[str, Callable[..., Any]]:
    """Return isolated wallet tools implementing the module contract."""
    balances = INITIAL_WALLET_BALANCES.copy()
    previews: dict[str, dict[str, Any]] = {}
    transfers_by_key: dict[str, dict[str, Any]] = {}
    preview_keys: dict[str, str] = {}

    def get_wallet_balance(wallet_id: str) -> dict[str, Any]:
        """Return one wallet's current balance in paise."""
        if wallet_id not in balances:
            raise WalletNotFoundError(f"Wallet {wallet_id} was not found")
        return {
            "wallet_id": wallet_id,
            "balance_paise": balances[wallet_id],
        }

    def create_transfer_preview(
        source_wallet_id: str,
        destination_wallet_id: str,
        transfer_amount_paise: int,
    ) -> dict[str, Any]:
        """Return a non-mutating transfer preview."""
        if source_wallet_id not in balances:
            raise WalletNotFoundError(
                f"Wallet {source_wallet_id} was not found"
            )
        if destination_wallet_id not in balances:
            raise WalletNotFoundError(
                f"Wallet {destination_wallet_id} was not found"
            )
        if source_wallet_id == destination_wallet_id:
            raise InvalidTransferError(
                "Source and destination wallets must be different"
            )
        if (
            type(transfer_amount_paise) is not int
            or transfer_amount_paise <= 0
        ):
            raise InvalidTransferError(
                "Transfer amount must be a positive integer number of paise"
            )

        service_fee_paise = (
            transfer_amount_paise * TRANSFER_FEE_PERCENT // 100
        )
        source_total_debit_paise = (
            transfer_amount_paise + service_fee_paise
        )
        if balances[source_wallet_id] < source_total_debit_paise:
            raise InsufficientFundsError(
                f"Wallet {source_wallet_id} has insufficient funds"
            )

        preview = {
            "preview_id": PREVIEW_ID,
            "source_wallet_id": source_wallet_id,
            "destination_wallet_id": destination_wallet_id,
            "transfer_amount_paise": transfer_amount_paise,
            "service_fee_paise": service_fee_paise,
            "source_total_debit_paise": source_total_debit_paise,
            "destination_credit_paise": transfer_amount_paise,
        }
        previews[PREVIEW_ID] = preview
        return preview.copy()

    def execute_transfer(
        preview_id: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Commit or replay an idempotent transfer."""
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise InvalidTransferError("Idempotency key must not be empty")

        existing_transfer = transfers_by_key.get(idempotency_key)
        if existing_transfer is not None:
            if existing_transfer["preview_id"] != preview_id:
                raise InvalidTransferError(
                    "Idempotency key belongs to a different preview"
                )
            return existing_transfer.copy()

        preview = previews.get(preview_id)
        if preview is None:
            raise PreviewNotFoundError(
                f"Transfer preview {preview_id} was not found"
            )

        existing_key = preview_keys.get(preview_id)
        if existing_key is not None:
            raise InvalidTransferError(
                "Transfer preview was already executed with a different key"
            )

        source_wallet_id = preview["source_wallet_id"]
        destination_wallet_id = preview["destination_wallet_id"]
        source_total_debit_paise = preview["source_total_debit_paise"]
        destination_credit_paise = preview["destination_credit_paise"]
        if balances[source_wallet_id] < source_total_debit_paise:
            raise InsufficientFundsError(
                f"Wallet {source_wallet_id} has insufficient funds"
            )

        balances[source_wallet_id] -= source_total_debit_paise
        balances[destination_wallet_id] += destination_credit_paise
        transfer = {
            "transfer_id": TRANSFER_ID,
            "preview_id": preview_id,
            "idempotency_key": idempotency_key,
            "status": "completed",
            "source_wallet_id": source_wallet_id,
            "destination_wallet_id": destination_wallet_id,
            "transfer_amount_paise": preview["transfer_amount_paise"],
            "service_fee_paise": preview["service_fee_paise"],
            "source_total_debit_paise": source_total_debit_paise,
            "destination_credit_paise": destination_credit_paise,
            "source_balance_paise": balances[source_wallet_id],
            "destination_balance_paise": balances[destination_wallet_id],
        }
        transfers_by_key[idempotency_key] = transfer
        preview_keys[preview_id] = idempotency_key
        raise TimeoutError("Transfer committed but its response was lost")

    def lookup_transfer_by_idempotency_key(
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Return the authoritative record for an idempotency key."""
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise TransferNotFoundError(
                "No transfer exists for an empty idempotency key"
            )
        transfer = transfers_by_key.get(idempotency_key)
        if transfer is None:
            raise TransferNotFoundError(
                f"No transfer exists for idempotency key {idempotency_key}"
            )
        return transfer.copy()

    return {
        "get_wallet_balance": get_wallet_balance,
        "create_transfer_preview": create_transfer_preview,
        "execute_transfer": execute_transfer,
        "lookup_transfer_by_idempotency_key": (
            lookup_transfer_by_idempotency_key
        ),
    }
