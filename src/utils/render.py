from __future__ import annotations

from typing import Any
import os
import re

import pandas as pd

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")
_IMG_REGEX = re.compile(r'([A-Za-z0-9_\-./\\:]+?\.(?:png|jpg|jpeg|webp|gif))', re.IGNORECASE)


def _extract_image_path_from_text(text: str) -> str | None:
    """
    Extract first image path-like substring from any text.
    Handles:
      - "exports\\charts\\x.png"
      - "here: exports/charts/x.png"
      - quoted paths
    """
    if not isinstance(text, str):
        return None

    m = _IMG_REGEX.search(text)
    if not m:
        return None

    p = m.group(1).strip().strip('"').strip("'")
    p_norm = p.replace("/", os.sep).replace("\\", os.sep)

    candidates = [p_norm, os.path.join(os.getcwd(), p_norm)]
    for c in candidates:
        if os.path.exists(c):
            return c

    return p_norm


def response_to_display_payload(resp: Any) -> dict[str, Any]:
    """
    Normalize PandasAI output into a UI-friendly payload:
      kind: error | metric | table | chart | text
      data: depends on kind
    """
    if isinstance(resp, dict) and resp.get("type") == "error":
        return {"kind": "error", "data": resp.get("value", "Unknown error")}

    if isinstance(resp, dict):
        plot = resp.get("plot")
        if isinstance(plot, str):
            p = _extract_image_path_from_text(plot)
            if p:
                return {"kind": "chart", "data": p}

        if resp.get("type") == "plot" and isinstance(resp.get("value"), str):
            p = _extract_image_path_from_text(resp["value"])
            if p:
                return {"kind": "chart", "data": p}

        if "value" in resp:
            val = resp["value"]

            if isinstance(val, pd.DataFrame):
                return {"kind": "table", "data": val}

            if isinstance(val, (int, float)):
                return {"kind": "metric", "label": "Result", "data": val}

            if isinstance(val, str):
                p = _extract_image_path_from_text(val)
                if p:
                    return {"kind": "chart", "data": p}
                return {"kind": "text", "data": val}

            p = _extract_image_path_from_text(str(val))
            if p:
                return {"kind": "chart", "data": p}

            return {"kind": "text", "data": str(val)}

        p = _extract_image_path_from_text(str(resp))
        if p:
            return {"kind": "chart", "data": p}
        return {"kind": "text", "data": str(resp)}

    if isinstance(resp, str):
        p = _extract_image_path_from_text(resp)
        if p:
            return {"kind": "chart", "data": p}
        return {"kind": "text", "data": resp}

    if isinstance(resp, pd.DataFrame):
        return {"kind": "table", "data": resp}

    if isinstance(resp, (int, float)):
        return {"kind": "metric", "label": "Result", "data": resp}

    s = str(resp)
    p = _extract_image_path_from_text(s)
    if p:
        return {"kind": "chart", "data": p}

    return {"kind": "text", "data": s}


def maybe_make_plotly_chart(df: pd.DataFrame, chart_spec: dict[str, Any] | None):
    """
    Optional fallback chart builder (only use if you still want it).
    """
    if not chart_spec:
        return None

    ctype = chart_spec.get("type", "none")
    if ctype == "none":
        return None

    try:
        import plotly.express as px
    except Exception:
        return None

    x = chart_spec.get("x") or chart_spec.get("names")
    y = chart_spec.get("y") or chart_spec.get("values")

    if ctype == "scatter" and x in df.columns and y in df.columns:
        return px.scatter(df, x=x, y=y, title=chart_spec.get("title") or "Scatter")
    if ctype == "hist" and x in df.columns:
        return px.histogram(df, x=x, title=chart_spec.get("title") or "Histogram")
    if ctype == "bar" and x in df.columns and y in df.columns:
        return px.bar(df, x=x, y=y, title=chart_spec.get("title") or "Bar")
    if ctype == "line" and x in df.columns and y in df.columns:
        return px.line(df, x=x, y=y, title=chart_spec.get("title") or "Line")
    if ctype == "pie" and x in df.columns and y in df.columns:
        return px.pie(df, names=x, values=y, title=chart_spec.get("title") or "Pie")

    return None