from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandasai import SmartDataframe
from pandasai_litellm.litellm import LiteLLM


@dataclass
class ExecutorResult:
    response: Any
    model_used: str
    last_code_executed: str | None = None


CHARTS_DIR = "exports/charts"


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
        or "tpm" in s  # tokens per minute
    )


def _retry_after_seconds(msg: str) -> float | None:
    """
    Groq/LiteLLM often returns: 'Please try again in 1.76s'
    """
    low = (msg or "").lower()
    m = re.search(r"try again in\s*([0-9.]+)s", low)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _ensure_charts_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        # If we cannot create it, PandasAI may still work but chart saving might fail.
        pass


def _run_once(
    df: pd.DataFrame,
    prompt: str,
    model: str,
    api_key: str,
    *,
    max_retries: int = 2,
) -> ExecutorResult:
    """
    Runs PandasAI once, with small retry on quota/rate-limit.
    """
    # ensure charts dir exists
    _ensure_charts_dir(CHARTS_DIR)

    # set env vars for LiteLLM providers (pandasai_litellm sometimes expects these)
    if model.startswith("gemini/"):
        os.environ["GOOGLE_API_KEY"] = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if model.startswith("groq/"):
        os.environ["GROQ_API_KEY"] = api_key or os.environ.get("GROQ_API_KEY", "")

    llm = LiteLLM(model=model, api_key=api_key)

    sdf = SmartDataframe(
        df,
        config={
            "llm": llm,
            "save_charts": True,
            "save_charts_path": CHARTS_DIR,
            "open_charts": False,
            "enable_cache": False,
            "verbose": False,
        },
    )

    last_err: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            resp = sdf.chat(prompt)

            last_code = None
            try:
                last_code = getattr(sdf, "last_code_executed", None)
            except Exception:
                last_code = None

            return ExecutorResult(
                response=resp,
                model_used=model,
                last_code_executed=last_code,
            )

        except Exception as e:
            last_err = e

            # retry only if quota/rate limit
            if _is_quota_or_rate_limit(e) and attempt < max_retries:
                wait = _retry_after_seconds(str(e))
                # fallback: exponential-ish backoff
                if wait is None:
                    wait = 1.5 * (attempt + 1)
                # add a tiny buffer so we don't hit the same window
                wait = wait + 0.75
                
                wait = min(wait, 25.0)
                
                time.sleep(wait)
                continue

            break

    # If we got here, all retries failed
    return ExecutorResult(
        response={"type": "error", "value": f"{last_err.__class__.__name__}: {last_err}"},
        model_used=model,
        last_code_executed=None,
    )


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

    # ---- Primary ----
    primary_key = gemini_api_key if primary_model.startswith("gemini/") else groq_api_key
    if not primary_key:
        return ExecutorResult(
            response={"type": "error", "value": f"Missing API key for primary model: {primary_model}"},
            model_used=primary_model,
            last_code_executed=None,
        )

    primary_res = _run_once(df, pandasai_prompt, primary_model, primary_key, max_retries=3)

    # If primary succeeded, return it
    if not (isinstance(primary_res.response, dict) and primary_res.response.get("type") == "error"):
        return primary_res

    # If primary failed but NOT due to rate-limit/quota, don't hide it by switching models
    if not _is_quota_or_rate_limit(Exception(str(primary_res.response.get("value", "")))):
        return primary_res

    # ---- Fallback ----
    fallback_key = groq_api_key if fallback_model.startswith("groq/") else gemini_api_key
    if not fallback_key:
        return ExecutorResult(
            response={"type": "error", "value": f"Missing API key for fallback model: {fallback_model}"},
            model_used=fallback_model,
            last_code_executed=None,
        )

    return _run_once(df, pandasai_prompt, fallback_model, fallback_key, max_retries=3)