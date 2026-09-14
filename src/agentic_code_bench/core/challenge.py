import json
from pathlib import Path
from typing import Any, cast


def load_challenge(file_path: str | Path) -> dict[str, Any]:
    with Path(file_path).open(encoding="utf-8") as file:
        return cast(dict[str, Any], json.load(file))
