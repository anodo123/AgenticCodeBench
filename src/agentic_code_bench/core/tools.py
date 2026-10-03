from collections.abc import Callable


def calculate_triangle_area(base: int, height: int) -> float:
    return 0.5 * base * height


def calculate_rectangle_area(width: int, height: int) -> int:
    return width * height


def calculate_rectangle_perimeter(width: int, height: int) -> int:
    return 2 * (width + height)


def calculate_percentage(amount: int | float, percentage: int | float) -> float:
    return amount * percentage / 100


def add_numbers(left: int | float, right: int | float) -> int | float:
    return left + right


def multiply_numbers(left: int | float, right: int | float) -> int | float:
    return left * right


def subtract_numbers(left: int | float, right: int | float) -> int | float:
    return left - right


def divide_numbers(dividend: int | float, divisor: int | float) -> float:
    return dividend / divisor


TOOL_REGISTRY: dict[str, Callable[..., object]] = {
    "calculate_triangle_area": calculate_triangle_area,
    "calculate_rectangle_area": calculate_rectangle_area,
    "calculate_rectangle_perimeter": calculate_rectangle_perimeter,
    "calculate_percentage": calculate_percentage,
    "add_numbers": add_numbers,
    "multiply_numbers": multiply_numbers,
    "subtract_numbers": subtract_numbers,
    "divide_numbers": divide_numbers,
}
