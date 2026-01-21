from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from google import genai
from google.genai import types
from typing_extensions import TypedDict

ChartType = Literal["none", "line", "bar", "scatter", "pie", "hist", "map"]


class ChartSpec(TypedDict, total=False):
    type: ChartType
    x: str
    y: str
    color: str
    values: str
    names: str
    title: str


class PlanPayload(TypedDict):
    plan: list[str]
    requires_chart: bool
    chart: ChartSpec
    pandasai_prompt: str


@dataclass
class PlannerResult:
    plan: list[str]
    requires_chart: bool
    chart: dict[str, Any]
    pandasai_prompt: str
    raw_json: dict[str, Any]


SYSTEM_INSTRUCTIONS = """You are the Planner agent in a multi-agent data analysis app.

You receive:
- User question
- A dataframe schema (columns + inferred dtypes) and a small sample preview
- The last few chat turns

Your job:
1) Produce a short, step-by-step execution plan
2) Decide if a chart is required (trend/comparison/distribution/correlation)
3) Produce a single PandasAI prompt (one string) that the Executor agent will run.
   - The prompt must be executable using a dataframe called `df` (PandasAI will provide it).
   - If a chart is needed, explicitly instruct PandasAI to plot it.

Return ONLY valid JSON in the provided schema.
Be concise and product-first.
"""


def _is_quota_error(e: Exception) -> bool:
    s = str(e).lower()
    return (
        "429" in s
        or "resource_exhausted" in s
        or "quota exceeded" in s
        or "rate limit" in s
        or "too many requests" in s
    )


def _df_schema_summary(df: pd.DataFrame, max_cols: int = 80) -> str:
    cols = df.columns.tolist()[:max_cols]
    dtypes = {c: str(df[c].dtype) for c in cols}
    return json.dumps({"columns": cols, "dtypes": dtypes}, ensure_ascii=False)


def _df_preview(df: pd.DataFrame, n: int = 5) -> str:
    preview = df.head(n).to_dict(orient="records")
    return json.dumps(preview, ensure_ascii=False)


def _history_to_text(history: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines).strip()


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", " ", str(s).lower()).strip()


def _find_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    keys = [k.lower() for k in keywords]
    for c in df.columns.tolist():
        cn = _norm(c)
        if any(k in cn for k in keys):
            return c
    return None


def _extract_top_n(q: str, default: int = 10) -> int:
    m = re.search(r"\btop\s+(\d+)\b", q)
    if m:
        try:
            return max(1, int(m.group(1)))
        except Exception:
            return default
    return default


def _extract_year_range(q: str) -> tuple[int | None, int | None]:
    m = re.search(r"\b(20\d{2})\s*(?:–|-|to)\s*(20\d{2})\b", q)
    if m:
        return int(m.group(1)), int(m.group(2))
    years = re.findall(r"\b(20\d{2})\b", q)
    if years:
        ys = sorted({int(y) for y in years})
        if len(ys) == 1:
            return ys[0], ys[0]
        return ys[0], ys[-1]
    return None, None


def _is_followup_same(q: str) -> bool:
    ql = q.lower()
    return any(
        phrase in ql
        for phrase in [
            "do the same",
            "same as before",
            "same as previous",
            "repeat that",
            "repeat the same",
            "do that",
            "now do the same",
            "now do that",
            "do the same thing",
        ]
    )


def _get_last_anchor_user_question(history: list[dict[str, str]]) -> str | None:
    """
    Find the most recent user question that is NOT a 'do the same' style follow-up.
    """
    for msg in reversed(history[:-1]): 
        if msg.get("role") != "user":
            continue
        txt = (msg.get("content") or "").strip()
        if not txt:
            continue
        if _is_followup_same(txt):
            continue
        return txt
    return None


def _extract_segment_filter(df: pd.DataFrame, q: str) -> tuple[str | None, str | None]:
    """
    Returns (segment_column, segment_value) if user mentions segment value.
    """
    seg_col = _find_col(df, ["segment"])
    if not seg_col:
        return None, None

    ql = q.lower()

    # Try common values
    candidates = []
    for cand in ["consumer", "corporate", "home office", "home-office", "homeoffice"]:
        if cand in ql:
            candidates.append(cand)

    if not candidates:
        m = re.search(r"only\s+for\s+(.+?)\s+segment", ql)
        if m:
            candidates.append(m.group(1).strip())

    if not candidates:
        return seg_col, None

    # (case-insensitive)
    try:
        uniques = [str(x) for x in df[seg_col].dropna().unique().tolist()[:50]]
    except Exception:
        uniques = []

    for c in candidates:
        for u in uniques:
            if _norm(u) == _norm(c) or _norm(c) in _norm(u) or _norm(u) in _norm(c):
                return seg_col, u

    return seg_col, candidates[0].title()


def fallback_plan(user_question: str, df: pd.DataFrame, history: list[dict[str, str]] | None = None) -> PlannerResult:
    """
    Offline planner: schema-aware + history-aware (for "do the same" follow-ups).
    """
    history = history or []

    q_raw = (user_question or "").strip()
    q = q_raw.lower()

    # Schema signals
    profit_col = _find_col(df, ["profit"])
    sales_col = _find_col(df, ["sales", "revenue"])
    discount_col = _find_col(df, ["discount"])
    date_col = _find_col(df, ["order date", "ship date", "date"])
    segment_col = _find_col(df, ["segment"])
    state_col = _find_col(df, ["state"])
    region_col = _find_col(df, ["region"])
    category_col = _find_col(df, ["category"])
    subcat_col = _find_col(df, ["sub category", "subcategory", "sub-category"])
    shipmode_col = _find_col(df, ["ship mode", "shipmode"])
    order_id_col = _find_col(df, ["order id", "orderid"])
    customer_col = _find_col(df, ["customer name", "customer", "customer id", "cust"])
    lat_col = _find_col(df, ["lat", "latitude"])
    lon_col = _find_col(df, ["lon", "lng", "longitude"])

    anchor = None
    seg_filter_col, seg_filter_val = _extract_segment_filter(df, q_raw)

    if _is_followup_same(q_raw):
        anchor = _get_last_anchor_user_question(history)
        if anchor:
            q = (anchor + " " + q_raw).lower()
            q_raw = (
                f"Repeat the previous analysis: {anchor}\n"
                f"Follow-up instruction: {user_question}\n"
            )
            if seg_filter_col and seg_filter_val:
                q_raw += f"IMPORTANT: Filter df to `{seg_filter_col}` == '{seg_filter_val}' before doing the same analysis.\n"

    # Infer chart type 
    chart_type: ChartType = "none"
    wants_chart = any(k in q for k in ["chart", "plot", "graph", "visualize", "visualise", "draw"])

    if any(k in q for k in ["map", "location", "locations", "geo"]):
        chart_type = "map"
    elif any(k in q for k in ["scatter", "correlation", "relationship"]):
        chart_type = "scatter"
    elif "pie" in q:
        chart_type = "pie"
    elif any(k in q for k in ["hist", "histogram", "distribution"]):
        chart_type = "hist"
    elif any(k in q for k in ["trend", "over time", "time series", "per month", "per year", "line chart", "line"]):
        chart_type = "line"
    elif any(k in q for k in ["count plot", "countplot", "most orders", "number of orders", "count"]):
        chart_type = "bar"
    elif any(k in q for k in ["bar chart", "bar", "top ", "compare", "comparison", "rank", "highest", "lowest"]):
        chart_type = "bar"

    if chart_type == "line" and not date_col:
        chart_type = "bar" if (category_col or segment_col or state_col or region_col) else "none"

    requires_chart = (chart_type != "none") or wants_chart

    # Determine group + metric
    group_col: str | None = None
    if "state" in q and state_col:
        group_col = state_col
    elif "region" in q and region_col:
        group_col = region_col
    elif "segment" in q and segment_col:
        group_col = segment_col
    elif ("sub-category" in q or "subcategory" in q or "sub category" in q) and subcat_col:
        group_col = subcat_col
    elif "ship" in q and shipmode_col:
        group_col = shipmode_col
    elif "category" in q and category_col:
        group_col = category_col
    elif "customer" in q and customer_col:
        group_col = customer_col
    else:
        group_col = segment_col or state_col or region_col or category_col or customer_col

    metric_col: str | None = None
    if "profit" in q and profit_col:
        metric_col = profit_col
    elif ("sales" in q or "revenue" in q) and sales_col:
        metric_col = sales_col
    else:
        metric_col = profit_col or sales_col

    top_n = _extract_top_n(q, default=10)
    y1, y2 = _extract_year_range(q)

    wants_count = any(k in q for k in ["count plot", "countplot", "most orders", "number of orders"]) or (
        "count" in q and "discount" not in q and "profit" not in q and "sales" not in q
    )

    # Build plan
    plan: list[str] = []
    if chart_type == "scatter":
        plan = [
            f"Filter rows if needed (e.g., Segment) using schema column `{segment_col}`.",
            f"Use x=`{discount_col}` and y=`{profit_col or metric_col}` based on schema.",
            "Drop rows with missing x/y.",
            "Compute correlation (Pearson).",
            "Render a scatter plot (x vs y).",
        ]
    elif chart_type == "line":
        plan = [
            f"Parse `{date_col}` as datetime.",
            "Filter to the requested year range if present.",
            f"Aggregate SUM of `{metric_col}` over time.",
            "Return the time series table sorted by time.",
            "Render a line chart (time on x, metric on y).",
        ]
    elif chart_type == "bar":
        if wants_count:
            plan = [
                f"Filter rows if needed (e.g., Segment).",
                f"Group by `{group_col}`.",
                f"Count orders using `{order_id_col}` if available, otherwise count rows.",
                f"Sort descending and show top {top_n} if requested.",
                "Render a bar chart.",
            ]
        else:
            plan = [
                f"Filter rows if needed (e.g., Segment).",
                f"Group by `{group_col}`.",
                f"Aggregate SUM of `{metric_col}`.",
                f"Sort descending and show top {top_n} if requested.",
                "Render a bar chart.",
            ]
    elif chart_type == "pie":
        plan = [
            f"Group by `{group_col}`.",
            f"Aggregate SUM of `{metric_col}`.",
            "Render a pie chart (names=group, values=sum).",
        ]
    elif chart_type == "hist":
        plan = [
            f"Select numeric column `{metric_col or discount_col}`.",
            "Drop missing values.",
            "Render a histogram + summary stats.",
        ]
    elif chart_type == "map":
        plan = [
            "Check schema for latitude/longitude columns.",
            "If lat/lon exist: plot points on a map (or lon-vs-lat scatter if map unavailable).",
            "If missing: explain limitation and fallback to state/region aggregation chart.",
        ]
    else:
        plan = [
            "Identify relevant columns from the schema based on the question.",
            "Apply filtering/aggregation needed to compute the answer.",
            "Return a concise result with clear ordering.",
        ]

    # Build PandasAI prompt 
    pandasai_prompt = q_raw

    if seg_filter_col and seg_filter_val:
        pandasai_prompt += f"\nFilter df to `{seg_filter_col}` == '{seg_filter_val}' first."

    if chart_type == "scatter":
        x = discount_col or "discount"
        y = profit_col or metric_col or "profit"
        pandasai_prompt += (
            f"\nCompute Pearson correlation between `{x}` and `{y}`, then plot a scatter chart "
            f"(x=`{x}`, y=`{y}`). Return the correlation value and show the chart."
        )
    elif chart_type == "line":
        dcol = date_col or "date"
        mcol = metric_col or "profit"
        yr = ""
        if y1 is not None and y2 is not None:
            yr = f" Filter to years {y1}-{y2}."
        pandasai_prompt += (
            f"\nParse `{dcol}` as datetime.{yr} Group by year (or month) and sum `{mcol}`."
            f" Plot a line chart with x=`{dcol}` (grouped) and y=sum `{mcol}`."
        )
    elif chart_type == "bar":
        g = group_col or "category"
        if wants_count:
            pandasai_prompt += (
                f"\nGroup by `{g}` and compute order count (use `{order_id_col}` distinct if available, else count rows)."
                f" Sort desc, show top {top_n} if relevant, and plot a bar chart."
            )
        else:
            mcol = metric_col or "profit"
            pandasai_prompt += (
                f"\nGroup by `{g}` and sum `{mcol}`. Sort desc, show top {top_n} if relevant, and plot a bar chart."
            )
    elif chart_type == "pie":
        g = group_col or "region"
        mcol = metric_col or "sales"
        pandasai_prompt += f"\nGroup by `{g}` and sum `{mcol}` then plot a pie chart (names=`{g}`, values=sum)."
    elif chart_type == "hist":
        x = metric_col or discount_col or "sales"
        pandasai_prompt += f"\nPlot a histogram for `{x}` (drop missing values) and show summary stats."
    elif chart_type == "map":
        if lat_col and lon_col:
            pandasai_prompt += f"\nPlot locations using latitude `{lat_col}` and longitude `{lon_col}`."
        else:
            pandasai_prompt += (
                "\nDataset has no lat/lon. Explain limitation briefly. "
                "Fallback: aggregate by state/region and plot a bar chart."
            )

    chart_spec: dict[str, Any] = {"type": chart_type}
    if chart_type == "scatter":
        chart_spec.update({"x": discount_col or "", "y": profit_col or metric_col or "", "title": "Correlation"})
    elif chart_type == "line":
        chart_spec.update({"x": date_col or "", "y": metric_col or "", "title": "Trend over time"})
    elif chart_type == "bar":
        chart_spec.update({"x": group_col or "", "y": metric_col or "", "title": "Comparison"})
    elif chart_type == "pie":
        chart_spec.update({"names": group_col or "", "values": metric_col or "", "title": "Distribution"})
    elif chart_type == "map":
        chart_spec.update({"x": lon_col or "", "y": lat_col or "", "title": "Locations"})

    raw = {
        "plan": plan,
        "requires_chart": bool(requires_chart),
        "chart": chart_spec,
        "pandasai_prompt": pandasai_prompt.strip(),
        "note": "Planner offline schema-aware heuristic (Gemini unavailable).",
        "debug": {
            "followup_anchor": anchor,
            "schema_cols_sample": df.columns.tolist()[:12],
            "picked": {
                "date": date_col,
                "group": group_col,
                "metric": metric_col,
                "discount": discount_col,
                "profit": profit_col,
                "segment": segment_col,
                "segment_filter_value": seg_filter_val,
            },
        },
    }

    return PlannerResult(
        plan=plan,
        requires_chart=bool(requires_chart),
        chart=raw["chart"],
        pandasai_prompt=raw["pandasai_prompt"],
        raw_json=raw,
    )


def plan_question(
    *,
    api_key: str,
    model: str,
    user_question: str,
    df: pd.DataFrame,
    history: list[dict[str, str]],
) -> PlannerResult:
    # Offline planner (If no API Key or Quota error)
    if not api_key:
        return fallback_plan(user_question, df, history)

    client = genai.Client(api_key=api_key)

    schema = _df_schema_summary(df)
    preview = _df_preview(df)
    hist_text = _history_to_text(history)

    prompt = f"""
USER_QUESTION:
{user_question}

DATAFRAME_SCHEMA_JSON:
{schema}

DATAFRAME_PREVIEW_JSON:
{preview}

CHAT_HISTORY:
{hist_text if hist_text else "(none)"}
""".strip()

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTIONS,
        response_mime_type="application/json",
        response_schema=PlanPayload,
        temperature=0.2,
    )

    try:
        result = client.models.generate_content(model=model, contents=prompt, config=config)
        text = (getattr(result, "text", None) or "").strip()
        if not text:
            text = str(result).strip()

        payload: dict[str, Any] = json.loads(text)

        plan = payload.get("plan") or []
        requires_chart = bool(payload.get("requires_chart", False))
        chart = payload.get("chart") or {"type": "none"}
        pandasai_prompt = (payload.get("pandasai_prompt") or user_question).strip()

        return PlannerResult(
            plan=plan,
            requires_chart=requires_chart,
            chart=chart,
            pandasai_prompt=pandasai_prompt,
            raw_json=payload,
        )

    except Exception as e:
        if _is_quota_error(e):
            return fallback_plan(user_question, df, history)
        raise