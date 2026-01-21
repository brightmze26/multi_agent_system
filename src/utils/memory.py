from __future__ import annotations

from typing import Literal


Role = Literal["user", "assistant"]


def append_message(history: list[dict[str, str]], role: Role, content: str) -> list[dict[str, str]]:
    history = list(history or [])
    history.append({"role": role, "content": content})
    return history


def trim_history(history: list[dict[str, str]], k: int) -> list[dict[str, str]]:
    if not history:
        return []
    if k <= 0:
        return []
    return history[-k:]