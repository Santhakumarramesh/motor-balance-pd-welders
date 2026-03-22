"""Data I/O, column normalization, and H&Y parsing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RANDOM_SEED = 42

FEATURES = ("BBS", "Mini", "FES")

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_path() -> Path:
    return repo_root() / "data" / "PD_WELDERS RAW Long Data-2.xlsx"


def parse_hy(val: Any) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).lower().replace("stage", "").strip()
    if s in ROMAN:
        return int(ROMAN[s])
    try:
        return int(float(re.findall(r"[\d.]+", s)[0]))
    except (IndexError, ValueError):
        return np.nan


def _norm_mini_col(columns: pd.Index) -> str | None:
    for c in columns:
        cl = str(c).lower().replace("_", "-").replace(" ", "-")
        if "mini" in cl and "best" in cl:
            return c
    return None


def load_pd_dataframe(xl_path: Path | str, sheet: str | int = "PD") -> pd.DataFrame:
    path = Path(xl_path)
    raw = pd.ExcelFile(path).parse(sheet)
    mini_col = _norm_mini_col(raw.columns) or "MINI-BEST"
    records = []
    for _, r in raw.iterrows():
        records.append(
            {
                "ID": r.get("Participant ID", ""),
                "Age": pd.to_numeric(r.get("Age (in years)"), errors="coerce"),
                "BBS": pd.to_numeric(r.get("BBS"), errors="coerce"),
                "Mini": pd.to_numeric(r.get(mini_col), errors="coerce"),
                "FES": pd.to_numeric(r.get("FES"), errors="coerce"),
                "HY": parse_hy(r.get("Current Stage of PD (Hoehn & Yahr)")),
                "Disease_Duration": pd.to_numeric(
                    str(r.get("Disease Duration (years/months)", "")).split("/")[0],
                    errors="coerce",
                ),
            }
        )
    df = pd.DataFrame(records)
    df["HY_bin"] = df["HY"].apply(
        lambda v: 0
        if pd.notna(v) and v <= 2
        else (1 if pd.notna(v) and v >= 3 else np.nan)
    )
    df["HY_bin_label"] = df["HY_bin"].map({0: "Early (I-II)", 1: "Late (III-IV)"})
    return df


def parse_fall_wd(v: Any) -> float:
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in ("no", "nil", "none", "0", ""):
        return 0.0
    if s in ("yes", "y"):
        return 1.0
    return np.nan


def encode_exposure(v: Any) -> float:
    if pd.isna(v):
        return np.nan
    s = str(v).lower()
    if "never" in s or "none" in s:
        return 0.0
    if "occasional" in s or "less" in s:
        return 1.0
    if "regular" in s or "sometimes" in s or "moderate" in s:
        return 2.0
    if "frequent" in s or "always" in s or "high" in s:
        return 3.0
    return np.nan


def encode_ppe(v: Any) -> float:
    if pd.isna(v):
        return np.nan
    s = str(v).lower()
    if "never" in s:
        return 0.0
    if "sometime" in s or "occasional" in s:
        return 1.0
    if "always" in s or "regular" in s:
        return 2.0
    return np.nan


def load_wd_dataframe(xl_path: Path | str) -> pd.DataFrame:
    path = Path(xl_path)
    raw = pd.ExcelFile(path).parse("WD")
    mini_col = _norm_mini_col(raw.columns) or "MINI-BEST SCORE"

    def col_find(pred):
        for c in raw.columns:
            if pred(str(c).lower()):
                return c
        return None

    fume_c = col_find(lambda s: "fume" in s)
    vib_c = col_find(lambda s: "vibrat" in s)
    noise_c = col_find(lambda s: "noise" in s and "exposure" in s)
    resp_c = col_find(
        lambda s: ("respirat" in s and "ppe" in s)
        or ("respiratory" in s and "protect" in s)
    )

    records = []
    for _, r in raw.iterrows():
        records.append(
            {
                "ID": r.get("Participant's Name", ""),
                "Age": pd.to_numeric(r.get("Age"), errors="coerce"),
                "BBS": pd.to_numeric(r.get("BBS"), errors="coerce"),
                "Mini": pd.to_numeric(r.get(mini_col), errors="coerce"),
                "FES": pd.to_numeric(r.get("FES"), errors="coerce"),
                "WeldYrs": pd.to_numeric(r.get("Total Years in Welding"), errors="coerce"),
                "HrsPerDay": pd.to_numeric(r.get("Work Hours per Day"), errors="coerce"),
                "FallHist": parse_fall_wd(r.get("History of Fall")),
                "FumeExp": encode_exposure(r.get(fume_c) if fume_c else np.nan),
                "VibExp": encode_exposure(r.get(vib_c) if vib_c else np.nan),
                "NoiseExp": encode_exposure(r.get(noise_c) if noise_c else np.nan),
                "RespPPE": encode_ppe(r.get(resp_c) if resp_c else np.nan),
            }
        )
    return pd.DataFrame(records)


def w_stage(yrs: float) -> float:
    if pd.isna(yrs):
        return np.nan
    if yrs < 10:
        return 1.0
    if yrs < 20:
        return 2.0
    if yrs < 30:
        return 3.0
    if yrs < 40:
        return 4.0
    return 5.0


def find_id_column(raw: pd.DataFrame) -> str | None:
    for c in raw.columns:
        cl = str(c).lower()
        if any(x in cl for x in ("participant", "subject", "name", "id")):
            return c
    return None


def normalize_balance_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Map heterogeneous Excel columns to BBS, Mini, FES and an ID column."""
    bbs_c = next((c for c in raw.columns if str(c).strip().lower() == "bbs"), None)
    if bbs_c is None:
        bbs_c = next((c for c in raw.columns if "bbs" in str(c).lower()), None)
    mini_c = _norm_mini_col(raw.columns)
    fes_c = next((c for c in raw.columns if str(c).strip().lower() == "fes"), None)
    if fes_c is None:
        fes_c = next(
            (
                c
                for c in raw.columns
                if "fes" in str(c).lower() and "mini" not in str(c).lower()
            ),
            None,
        )
    id_c = find_id_column(raw)
    missing = []
    if bbs_c is None:
        missing.append("BBS")
    if mini_c is None:
        missing.append("Mini-BEST (column containing 'mini' and 'best')")
    if fes_c is None:
        missing.append("FES")
    if missing:
        raise ValueError(
            "Could not resolve columns: "
            + ", ".join(missing)
            + f". Found columns: {list(raw.columns)}"
        )
    out = pd.DataFrame(
        {
            "ID": raw[id_c].astype(str) if id_c else [f"row{i}" for i in range(len(raw))],
            "BBS": pd.to_numeric(raw[bbs_c], errors="coerce"),
            "Mini": pd.to_numeric(raw[mini_c], errors="coerce"),
            "FES": pd.to_numeric(raw[fes_c], errors="coerce"),
        }
    )
    return out


def read_excel_sheet(path: Path | str, sheet: str | int | None) -> pd.DataFrame:
    """Parse one sheet; if sheet is None, prefer WD then PD then first sheet."""
    path = Path(path)
    xl = pd.ExcelFile(path)
    if sheet is not None:
        return xl.parse(sheet)
    for name in ("WD", "wd", "Welders"):
        if name in xl.sheet_names:
            return xl.parse(name)
    for name in ("PD",):
        if name in xl.sheet_names:
            return xl.parse(name)
    return xl.parse(xl.sheet_names[0])


def validate_ranges(df: pd.DataFrame, label: str = "") -> list[str]:
    warnings = []
    for _, row in df.iterrows():
        if pd.notna(row.get("BBS")) and (row["BBS"] < 0 or row["BBS"] > 56):
            warnings.append(f"{label} BBS out of 0–56: {row.get('ID', '')}")
        if pd.notna(row.get("Mini")) and (row["Mini"] < 0 or row["Mini"] > 28):
            warnings.append(f"{label} Mini-BEST out of 0–28: {row.get('ID', '')}")
    return warnings[:20]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
