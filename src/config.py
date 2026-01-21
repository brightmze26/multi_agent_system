from __future__ import annotations

import os
from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class AppConfig:
    max_upload_mb: int = 10
    chat_memory_k: int = 5

    gemini_api_key: str = ""
    groq_api_key: str = ""

    planner_model: str = "gemini-2.0-flash"

    # PandasAI via LiteLLM model strings
    pandasai_litellm_model: str = "gemini/gemini-2.0-flash"
    pandasai_litellm_fallback_model: str = "groq/llama-3.3-70b-versatile"


def _get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default) or default)
    except Exception:
        return default


def get_config() -> AppConfig:
    gemini_key = _get_secret("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    groq_key = _get_secret("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

    cfg = AppConfig(
        max_upload_mb=int(_get_secret("MAX_UPLOAD_MB", os.getenv("MAX_UPLOAD_MB", "10"))),
        chat_memory_k=int(_get_secret("CHAT_MEMORY_K", os.getenv("CHAT_MEMORY_K", "5"))),
        gemini_api_key=gemini_key,
        groq_api_key=groq_key,
        planner_model=_get_secret("PLANNER_MODEL", os.getenv("PLANNER_MODEL", "gemini-2.0-flash")),
        pandasai_litellm_model=_get_secret(
            "PANDASAI_PRIMARY_MODEL", os.getenv("PANDASAI_PRIMARY_MODEL", "gemini/gemini-2.0-flash")
        ),
        pandasai_litellm_fallback_model=_get_secret(
            "PANDASAI_FALLBACK_MODEL", os.getenv("PANDASAI_FALLBACK_MODEL", "groq/llama-3.3-70b-versatile")
        ),
    )

    if not cfg.gemini_api_key and not cfg.groq_api_key:
        raise ValueError(
            "Missing API keys. Set GEMINI_API_KEY and/or GROQ_API_KEY in .streamlit/secrets.toml "
            "or environment variables."
        )

    return cfg