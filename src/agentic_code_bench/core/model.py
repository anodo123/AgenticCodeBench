import os
from functools import cache
from typing import Any, cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import Response

load_dotenv()


@cache
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment "
            "or project .env file."
        )

    return OpenAI(api_key=api_key)


def convert_tool_definitions(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        }
        for tool in tools
    ]


def call_openai_model(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str = "gpt-5.6-luna",
    previous_response_id: str | None = None,
) -> Response:
    request: dict[str, Any] = {
        "model": model,
        "input": messages,
        "tools": convert_tool_definitions(tools),
    }
    if previous_response_id is not None:
        request["previous_response_id"] = previous_response_id

    return cast(Response, get_openai_client().responses.create(**request))
