from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandasai import SmartDataframe

# gemini/gemini-2.0-flash / groq/llama-3.3-70b-versatile
from pandasai_litellm.litellm import LiteLLM  


@dataclass
class ExecutorResult:
    response: Any
    model_used: str
    last_code_executed: str | None = None


def _is_quota_or_rate_limit(e: Exception) -> bool:
    s = str(e).lower()
    return (
        "429" in s
        or "resource_exhausted" in s
        or "quota exceeded" in s
        or "rate limit" in s
        or "ratelimiterror" in s
        or "rate_limit_exceeded" in s
        or "too many requests" in s
    )


def _run_once(df: pd.DataFrame, prompt: str, model: str, api_key: str) -> ExecutorResult:
    # Gemini: GOOGLE_API_KEY
    # Groq: GROQ_API_KEY
    if model.startswith("gemini/"):
        os.environ["GOOGLE_API_KEY"] = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if model.startswith("groq/"):
        os.environ["GROQ_API_KEY"] = api_key or os.environ.get("GROQ_API_KEY", "")

    llm = LiteLLM(model=model, api_key=api_key)

    # PandasAI SmartDataframe
    sdf = SmartDataframe(
        df,
        config={
            "llm": llm,
            "save_charts": True,
            "save_charts_path": "exports/charts",
            "open_charts": False,
            "enable_cache": False,
            "verbose": False,
        },
    )

    resp = sdf.chat(prompt)

    last_code = None
    try:
        last_code = getattr(sdf, "last_code_executed", None)  
    except Exception:
        last_code = None

    return ExecutorResult(response=resp, model_used=model, last_code_executed=last_code)


def run_pandasai(
    *,
    gemini_api_key: str,
    groq_api_key: str,
    primary_model: str,
    fallback_model: str,
    df: pd.DataFrame,
    pandasai_prompt: str,
) -> ExecutorResult:
    """
    Primary: Gemini (recommended)
    Fallback: Groq (if quota/rate limit)
    If both fail: return graceful error payload (not raising).
    """
    # Primary
    try:
        primary_key = gemini_api_key if primary_model.startswith("gemini/") else groq_api_key
        if not primary_key:
            raise ValueError(f"Missing API key for primary model: {primary_model}")
        return _run_once(df, pandasai_prompt, primary_model, primary_key)
    except Exception as e:
        if not _is_quota_or_rate_limit(e):
            return ExecutorResult(
                response={"type": "error", "value": f"{e.__class__.__name__}: {e}"},
                model_used=primary_model,
                last_code_executed=None,
            )

    # Fallback
    try:
        fallback_key = groq_api_key if fallback_model.startswith("groq/") else gemini_api_key
        if not fallback_key:
            raise ValueError(f"Missing API key for fallback model: {fallback_model}")
        return _run_once(df, pandasai_prompt, fallback_model, fallback_key)
    except Exception as e2:
        return ExecutorResult(
            response={"type": "error", "value": f"{e2.__class__.__name__}: {e2}"},
            model_used=fallback_model,
            last_code_executed=None,
        )