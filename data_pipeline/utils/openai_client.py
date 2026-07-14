from __future__ import annotations

import base64
import mimetypes
from typing import Any

from .io import first_image, message_text


def make_async_client(base_url: str, api_key: str) -> Any:
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def image_data_url(path: str) -> str:
    mime_type = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def vision_user_message(prompt: str, image_path: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
        ],
    }


def text_user_message(prompt: str) -> dict[str, str]:
    return {"role": "user", "content": prompt}


def row_to_openai_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [vision_user_message(message_text(row["messages"]), first_image(row))]


async def async_chat_completion(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    request: dict[str, Any],
    samples: int = 1,
) -> list[str]:
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        n=samples,
        **request,
    )
    return [choice.message.content or "" for choice in response.choices]

