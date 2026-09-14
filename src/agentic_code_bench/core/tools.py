from collections.abc import Callable


def calculate_triangle_area(base: int, height: int) -> float:
    return 0.5 * base * height


TOOL_REGISTRY: dict[str, Callable[..., object]] = {
    "calculate_triangle_area": calculate_triangle_area,
}
