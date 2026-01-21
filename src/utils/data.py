from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st


@dataclass
class LoadedData:
    df: pd.DataFrame
    filename: str


def _file_size_mb(uploaded_file) -> float:
    try:
        return float(uploaded_file.size) / (1024 * 1024)
    except Exception:
        return 0.0


def load_uploaded_file(uploaded_file, max_mb: int = 10) -> LoadedData:
    size_mb = _file_size_mb(uploaded_file)
    if size_mb > max_mb:
        raise ValueError(f"File too large: {size_mb:.2f}MB. Max allowed is {max_mb}MB.")

    name = getattr(uploaded_file, "name", "dataset")
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""

    if ext in ["xlsx", "xls"]:
        df = pd.read_excel(uploaded_file)
        return LoadedData(df=df, filename=name)

    if ext == "csv":
        raw = uploaded_file.getvalue()
        for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
            try:
                df = pd.read_csv(pd.io.common.BytesIO(raw), encoding=enc)
                return LoadedData(df=df, filename=name)
            except Exception:
                continue
        raise ValueError("Failed to read CSV. Try saving as UTF-8 or upload XLSX.")

    raise ValueError("Unsupported file type. Please upload CSV/XLSX.")