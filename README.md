# Intelligent Data Room (Streamlit + PandasAI + Gemini)

A lightweight web app to upload a CSV/XLSX file (≤ 10MB) and **talk to your data** using a **Multi-Agent workflow** that separates **Thinking** and **Doing**:

- **Agent 1 — Planner (Thinking):** reads the user question + dataframe schema + last 3–5 chat messages, then outputs a step-by-step **Execution Plan** (JSON).
- **Agent 2 — Executor (Doing):** uses **PandasAI** with **Gemini** (via LiteLLM) to execute the plan, generate Python code, and return results (tables/metrics/charts).

## Features
- ✅ Upload CSV / XLSX (max 10MB)
- ✅ Multi-agent flow (Planner → Executor)
- ✅ Automatic visualization for trend / comparison / correlation requests
- ✅ Context retention (last 3–5 messages) for follow-up questions
- ✅ Clean chat UX + readable plans + chart rendering in-app
- ✅ Graceful error handling (rate limit, quota, encoding issues)

---

## Tech Stack
- **Frontend/UI:** Streamlit
- **AI Execution:** PandasAI + LiteLLM
- **Planner LLM:** Google Gemini (recommended)
- **Executor LLM:** Google Gemini primary, Groq fallback (optional)

---

## Requirements
- Python **3.8–3.11** (PandasAI 3.x requires Python < 3.12)
- A **Gemini API Key**
- (Optional) A **Groq API Key** for fallback

---

## Setup (Local)
```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Fill in GEMINI_API_KEY (and GROQ_API_KEY if you want fallback)