from __future__ import annotations

import inspect
import os
import re
import traceback
from typing import Any

import pandas as pd
import streamlit as st

from src.agents.executor import run_pandasai
from src.agents.planner import plan_question
from src.config import get_config
from src.utils.data import load_uploaded_file
from src.utils.memory import append_message, trim_history
from src.utils.render import response_to_display_payload

st.set_page_config(page_title="Intelligent Data Room", page_icon="📊", layout="wide")


def _stretch_kwargs(fn) -> dict[str, Any]:
    """
    Streamlit is deprecating use_container_width.
    This helper keeps compatibility across versions:
      - if width exists -> width='stretch'
      - else -> use_container_width=True
    """
    try:
        params = inspect.signature(fn).parameters
    except Exception:
        params = {}
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


@st.cache_data(show_spinner=False)
def _profile_df(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }


def _init_state():
    if "df" not in st.session_state:
        st.session_state.df = None
    if "filename" not in st.session_state:
        st.session_state.filename = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_plan" not in st.session_state:
        st.session_state.last_plan = None


def _render_chat(history: list[dict[str, str]]):
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def _pretty_error(e: Exception) -> tuple[str, str, float | None]:
    msg = str(e) or e.__class__.__name__
    low = msg.lower()

    retry_after = None
    m = re.search(r"try again in\s*([0-9.]+)s", low)
    if m:
        try:
            retry_after = float(m.group(1))
        except Exception:
            retry_after = None

    if (
        "ratelimiterror" in low
        or "rate limit" in low
        or "rate_limit_exceeded" in low
        or "resource_exhausted" in low
        or "quota exceeded" in low
        or "too many requests" in low
        or "429" in low
    ):
        user_msg = (
            "⚠️ The LLM is currently rate-limited / quota-limited.\n\n"
            "Try:\n"
            "- Click **Retry** after a few seconds\n"
            "- Shorten the question (be more specific)\n"
            "- Reduce complexity (e.g., limit the year range / top N)\n"
        )
        if retry_after is not None:
            user_msg += f"\nSuggested retry in ~{retry_after:.1f} seconds."
        return user_msg, msg, retry_after

    if "nocodefounderror" in low or "no code found" in low:
        user_msg = (
            "⚠️ The model failed to generate analysis code for this query.\n\n"
            "Try:\n"
            "- Mention the exact columns (e.g., `profit`, `discount`, `order_date`)\n"
            "- Make the question more specific (example: 'group by state, sum profit, top 5')\n"
        )
        return user_msg, msg, retry_after

    if "codec can't decode" in low or "unicode" in low:
        user_msg = (
            "⚠️ The file appears to use an unsupported encoding (not UTF-8).\n\n"
            "Try:\n"
            "- Re-save the CSV as **UTF-8** (Excel: Save As → CSV UTF-8)\n"
            "- Or upload the XLSX version instead\n"
        )
        return user_msg, msg, retry_after

    user_msg = (
        "⚠️ Something went wrong while processing your request.\n\n"
        "Try:\n"
        "- Make sure the file is valid (CSV/XLSX)\n"
        "- Ask a simpler question\n"
        "- Click **Retry**\n"
    )
    return user_msg, msg, retry_after


def _show_elegant_error(e: Exception, *, context_label: str = "Error"):
    user_msg, tech_msg, retry_after = _pretty_error(e)

    st.error(f"**{context_label}**\n\n{user_msg}")

    c1, c2, c3 = st.columns([1.2, 1.3, 1.5])
    with c1:
        if st.button("🔁 Retry", type="primary", **_stretch_kwargs(st.button)):
            st.rerun()
    with c2:
        if retry_after is not None:
            st.info(f"Retry window: ~{retry_after:.1f}s")
    with c3:
        if st.button("🧹 Clear chat context", **_stretch_kwargs(st.button)):
            st.session_state.history = []
            st.session_state.last_plan = None
            st.rerun()

    with st.expander("Technical details (for debugging)"):
        st.code(tech_msg)
        st.code(traceback.format_exc())


def _render_chart_inline(path: str):
    """
    Render the chart inline from a file path (bytes),
    so it doesn't show up as plain text.
    """
    try:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                img_bytes = f.read()
            st.image(img_bytes, **_stretch_kwargs(st.image))
        else:
            st.warning("A chart path was returned, but the file was not found on disk.")
            with st.expander("Chart path"):
                st.code(str(path))
    except Exception as e:
        st.warning("Failed to render the chart inline.")
        with st.expander("Chart rendering error"):
            st.code(str(e))
            st.code(str(path))


def main():
    _init_state()

    try:
        cfg = get_config()
    except Exception as e:
        _show_elegant_error(e, context_label="Config error")
        st.stop()

    st.title("Intelligent Data Room")
    st.caption("Upload data → ask questions → get answers + charts (Planner + Executor agents).")

    with st.sidebar:
        st.subheader("1) Upload data")
        uploaded = st.file_uploader(
            "CSV/XLSX (max 10MB)",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=False,
        )

        colA, colB = st.columns(2)
        with colA:
            if st.button("Clear chat", **_stretch_kwargs(st.button)):
                st.session_state.history = []
                st.session_state.last_plan = None
        with colB:
            if st.button("Reset data", **_stretch_kwargs(st.button)):
                st.session_state.df = None
                st.session_state.filename = None
                st.session_state.history = []
                st.session_state.last_plan = None

        st.divider()
        st.subheader("2) App settings")
        st.write(f"Memory window: last **{cfg.chat_memory_k}** messages")
        st.write(f"Planner model: `{cfg.planner_model}`")
        st.write(f"PandasAI primary: `{cfg.pandasai_litellm_model}`")
        st.write(f"PandasAI fallback: `{cfg.pandasai_litellm_fallback_model}`")
        st.write("Fallback triggers automatically on 429/quota/rate limit.")

    # Load dataframe
    if uploaded is not None:
        try:
            loaded = load_uploaded_file(uploaded, max_mb=cfg.max_upload_mb)
            st.session_state.df = loaded.df
            st.session_state.filename = loaded.filename
        except Exception as e:
            _show_elegant_error(e, context_label="Upload error")
            st.stop()

    df: pd.DataFrame | None = st.session_state.df
    if df is None:
        st.info("Upload a CSV/XLSX to start.")
        st.stop()

    # Preview and profile
    left, right = st.columns([1.2, 0.8], gap="large")

    with left:
        st.subheader(f"Dataset Preview: {st.session_state.filename}")

        n_rows = len(df)
        max_preview = min(500, n_rows)

        if max_preview <= 30:
            preview_rows = max_preview
        else:
            step = 10 if (max_preview - 30) >= 10 else 1
            preview_rows = st.slider(
                "Preview rows",
                min_value=30,
                max_value=max_preview,
                value=30,
                step=step,
            )

        st.dataframe(df.head(preview_rows), **_stretch_kwargs(st.dataframe))
        st.caption(f"Showing {min(preview_rows, n_rows)} of {n_rows} rows")

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download full dataset (CSV)",
            data=csv_bytes,
            file_name=f"{(st.session_state.filename or 'dataset').rsplit('.', 1)[0]}_full.csv",
            mime="text/csv",
            **_stretch_kwargs(st.download_button),
        )

    with right:
        st.subheader("Quick Profile")
        prof = _profile_df(df)
        st.metric("Rows", prof["rows"])
        st.metric("Columns", prof["cols"])
        st.metric("Duplicate rows", prof["duplicate_rows"])
        with st.expander("Schema (columns & dtypes)"):
            st.json(prof["dtypes"])

    st.divider()

    # Chat
    _render_chat(st.session_state.history)

    user_q = st.chat_input("Ask something about your data… (e.g., 'Top 5 states by sales + bar chart')")
    if not user_q:
        return

    st.session_state.history = append_message(st.session_state.history, "user", user_q)
    st.session_state.history = trim_history(st.session_state.history, cfg.chat_memory_k)

    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        # Planner
        try:
            with st.spinner("Planner is thinking…"):
                planner_res = plan_question(
                    api_key=cfg.gemini_api_key,
                    model=cfg.planner_model,
                    user_question=user_q,
                    df=df,
                    history=st.session_state.history[-cfg.chat_memory_k :],
                )
        except Exception as e:
            _show_elegant_error(e, context_label="Planner error")
            st.stop()

        st.session_state.last_plan = planner_res.raw_json

        st.markdown("### ✅ Execution Plan")
        for i, step in enumerate(planner_res.plan, start=1):
            st.markdown(f"**{i}.** {step}")

        note = (planner_res.raw_json or {}).get("note")
        if note:
            st.caption(f"Planner note: {note}")

        with st.expander("Planner output (raw JSON)"):
            st.json(planner_res.raw_json)

        # Executor
        try:
            with st.spinner("Executor is running PandasAI…"):
                exec_res = run_pandasai(
                    gemini_api_key=cfg.gemini_api_key,
                    groq_api_key=cfg.groq_api_key,
                    primary_model=cfg.pandasai_litellm_model,
                    fallback_model=cfg.pandasai_litellm_fallback_model,
                    df=df,
                    pandasai_prompt=planner_res.pandasai_prompt,
                )
        except Exception as e:
            _show_elegant_error(e, context_label="Executor error")
            st.stop()

        st.caption(f"Executor model used: `{exec_res.model_used}`")

        if exec_res.last_code_executed:
            with st.expander("PandasAI generated code"):
                st.code(exec_res.last_code_executed, language="python")

        display = response_to_display_payload(exec_res.response)

        # Render
        if display["kind"] == "error":
            _show_elegant_error(Exception(str(display["data"])), context_label="Executor result error")
            assistant_text = f"⚠️ Error: {display['data']}"
        elif display["kind"] == "metric":
            st.metric(display.get("label", "Result"), display["data"])
            assistant_text = f"{display.get('label','Result')}: {display['data']}"
        elif display["kind"] == "table":
            st.dataframe(display["data"], **_stretch_kwargs(st.dataframe))
            assistant_text = "Displayed a table result."
        elif display["kind"] == "chart":
            _render_chart_inline(str(display["data"]))
            assistant_text = "Rendered a chart."
        else:
            st.markdown(display["data"])
            assistant_text = str(display["data"])

        if planner_res.requires_chart and display["kind"] != "chart" and display["kind"] != "error":
            st.info("Chart requested, but the Executor did not return a chart. Try rephrasing (e.g., 'show bar chart') or retry.")

        st.session_state.history = append_message(st.session_state.history, "assistant", assistant_text)
        st.session_state.history = trim_history(st.session_state.history, cfg.chat_memory_k)


if __name__ == "__main__":
    main()