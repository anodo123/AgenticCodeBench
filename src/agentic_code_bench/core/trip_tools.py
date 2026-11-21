"""Domain tools for Challenge 011's trip-booking saga.

Flight F-204 costs 1800000 paise. Hotel H-88 costs 400000 paise per night.
Both appear available during quoting. Flight reservation succeeds,
hotel reservation fails permanently with RoomUnavailableError, and flight
cancellation commits before raising TimeoutError. A booking lookup must reveal
the committed cancellation. Replaying any side effect with its original key
must not repeat the state change.
"""

from collections.abc import Callable
from typing import Any
from uuid import uuid4

FLIGHT_ID = "F-204"
HOTEL_ID = "H-88"
FLIGHT_PRICE_PAISE = 1_800_000
HOTEL_NIGHTLY_PRICE_PAISE = 400_000
TRIP_BUDGET_PAISE = 3_000_000
FLIGHT_BOOKING_ID = "FLIGHT-BOOKING-204"


class FlightNotFoundError(ValueError):
    """The requested flight does not exist."""


class HotelNotFoundError(ValueError):
    """The requested hotel does not exist."""


class InvalidNightsError(ValueError):
    """The requested number of hotel nights is invalid."""


class RoomUnavailableError(RuntimeError):
    """The quoted hotel room is permanently unavailable at reservation time."""


class BookingNotFoundError(ValueError):
    """The requested booking does not exist."""


class InvalidTripActionError(ValueError):
    """The requested booking or compensation action is invalid."""


def create_action_key(action: str) -> str:
    """Return a fresh application-owned idempotency key for one saga action."""
    return f"{action}-{uuid4()}"


def calculate_trip_total(
    flight_price_paise: int,
    hotel_price_paise: int,
) -> dict[str, int]:
    """Return the combined price of two valid non-negative integer quotes."""
    if (
        type(flight_price_paise) is not int
        or type(hotel_price_paise) is not int
        or flight_price_paise < 0
        or hotel_price_paise < 0
    ):
        raise ValueError("Trip prices must be non-negative integer paise values")
    return {
        "flight_price_paise": flight_price_paise,
        "hotel_price_paise": hotel_price_paise,
        "total_price_paise": flight_price_paise + hotel_price_paise,
    }


def calculate_hotel_stay_total(
    nightly_price_paise: int,
    nights: int,
) -> dict[str, int]:
    """Return a hotel stay total from a nightly price and number of nights."""
    if type(nightly_price_paise) is not int or nightly_price_paise < 0:
        raise ValueError("Nightly price must be a non-negative integer")
    if type(nights) is not int or nights <= 0:
        raise InvalidNightsError("Nights must be a positive integer")
    return {
        "nightly_price_paise": nightly_price_paise,
        "nights": nights,
        "hotel_price_paise": nightly_price_paise * nights,
    }


def create_trip_tool_registry() -> dict[str, Callable[..., Any]]:
    """Return isolated trip tools implementing the module contract."""
    flight_bookings: dict[str, dict[str, Any]] = {}
    flight_reservations_by_key: dict[str, dict[str, Any]] = {}
    hotel_reservations_by_key: dict[str, dict[str, Any]] = {}
    flight_cancellations_by_key: dict[str, dict[str, Any]] = {}

    def get_flight_quote(flight_id: str) -> dict[str, Any]:
        """Return the current flight quote."""
        if flight_id != FLIGHT_ID:
            raise FlightNotFoundError(f"Flight {flight_id} was not found")
        return {
            "flight_id": flight_id,
            "flight_price_paise": FLIGHT_PRICE_PAISE,
            "available": True,
        }

    def get_hotel_quote(hotel_id: str, nights: int) -> dict[str, Any]:
        """Return the current hotel quote."""
        if hotel_id != HOTEL_ID:
            raise HotelNotFoundError(f"Hotel {hotel_id} was not found")
        if type(nights) is not int or nights <= 0:
            raise InvalidNightsError("Nights must be a positive integer")
        return {
            "hotel_id": hotel_id,
            "nightly_price_paise": HOTEL_NIGHTLY_PRICE_PAISE,
            "nights": nights,
            "available": True,
        }

    def reserve_flight(
        flight_id: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Create or replay an idempotent flight reservation."""
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise InvalidTripActionError(
                "A flight reservation requires a non-empty idempotency key"
            )

        existing = flight_reservations_by_key.get(idempotency_key)
        if existing is not None:
            if existing["flight_id"] != flight_id:
                raise InvalidTripActionError(
                    f"Idempotency key {idempotency_key!r} belongs to flight "
                    f"{existing['flight_id']!r} and booking "
                    f"{existing['booking_id']!r}; it cannot reserve flight "
                    f"{flight_id!r}"
                )
            return existing.copy()

        if flight_id != FLIGHT_ID:
            raise FlightNotFoundError(f"Flight {flight_id} was not found")

        result = {
            "booking_id": FLIGHT_BOOKING_ID,
            "flight_id": flight_id,
            "status": "booked",
            "idempotency_key": idempotency_key,
        }
        flight_reservations_by_key[idempotency_key] = result.copy()
        flight_bookings[FLIGHT_BOOKING_ID] = {
            "booking_id": FLIGHT_BOOKING_ID,
            "flight_id": flight_id,
            "status": "booked",
        }
        return result

    def reserve_hotel(
        hotel_id: str,
        nights: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Attempt or replay an idempotent hotel reservation."""
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise InvalidTripActionError(
                "A hotel reservation requires a non-empty idempotency key"
            )

        existing = hotel_reservations_by_key.get(idempotency_key)
        if existing is not None:
            if (
                existing["hotel_id"] != hotel_id
                or existing["nights"] != nights
            ):
                raise InvalidTripActionError(
                    f"Idempotency key {idempotency_key!r} belongs to hotel "
                    f"{existing['hotel_id']!r} for {existing['nights']} nights; "
                    f"it cannot reserve hotel {hotel_id!r} for {nights!r} nights"
                )
            raise RoomUnavailableError(
                f"Hotel {hotel_id} is unavailable for {nights} nights"
            )

        if hotel_id != HOTEL_ID:
            raise HotelNotFoundError(f"Hotel {hotel_id} was not found")
        if type(nights) is not int or nights <= 0:
            raise InvalidNightsError("Nights must be a positive integer")

        hotel_reservations_by_key[idempotency_key] = {
            "booking_id": None,
            "hotel_id": hotel_id,
            "nights": nights,
            "idempotency_key": idempotency_key,
            "available": False,
            "status": "failed",
        }
        raise RoomUnavailableError(
            f"Hotel {hotel_id} is unavailable for {nights} nights"
        )

    def cancel_flight(
        booking_id: str,
        compensation_key: str,
    ) -> dict[str, Any]:
        """Commit or replay an idempotent flight cancellation."""
        if not isinstance(compensation_key, str) or not compensation_key.strip():
            raise InvalidTripActionError(
                "A flight cancellation requires a non-empty compensation key"
            )

        existing = flight_cancellations_by_key.get(compensation_key)
        if existing is not None:
            if existing["booking_id"] != booking_id:
                raise InvalidTripActionError(
                    f"Compensation key {compensation_key!r} belongs to booking "
                    f"{existing['booking_id']!r}; it cannot cancel booking "
                    f"{booking_id!r}"
                )
            return existing.copy()

        booking = flight_bookings.get(booking_id)
        if booking is None:
            raise BookingNotFoundError(f"Booking {booking_id} was not found")
        if booking["status"] == "cancelled":
            raise InvalidTripActionError(
                f"Booking {booking_id!r} was already cancelled with another "
                "compensation key"
            )
        if booking["status"] != "booked":
            raise InvalidTripActionError(
                f"Booking {booking_id!r} cannot be cancelled from status "
                f"{booking['status']!r}"
            )

        booking["status"] = "cancelled"
        result = {
            "compensation_key": compensation_key,
            "booking_id": booking_id,
            "status": "cancelled",
        }
        flight_cancellations_by_key[compensation_key] = result.copy()
        raise TimeoutError(
            f"Cancellation response timed out for booking {booking_id}"
        )

    def lookup_flight_booking(booking_id: str) -> dict[str, Any]:
        """Return the authoritative flight-booking state."""
        booking = flight_bookings.get(booking_id)
        if booking is None:
            raise BookingNotFoundError(f"Booking {booking_id} was not found")
        return booking.copy()

    return {
        "get_flight_quote": get_flight_quote,
        "get_hotel_quote": get_hotel_quote,
        "calculate_hotel_stay_total": calculate_hotel_stay_total,
        "calculate_trip_total": calculate_trip_total,
        "reserve_flight": reserve_flight,
        "reserve_hotel": reserve_hotel,
        "cancel_flight": cancel_flight,
        "lookup_flight_booking": lookup_flight_booking,
    }
