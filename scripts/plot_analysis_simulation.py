#!/usr/bin/env python3
"""
plot_sections_per_metric.py

Create ONE plot per metric section in a "SIMULATION ANALYSIS REPORT" CSV.
A section starts with a line like: === METRIC NAME ===
The script parses each section and builds the most informative plot it can
from the two text columns contained in the CSV.

What it handles:
- Standard "metric | value" tables -> horizontal bar chart
- Category distributions like "obstacle_count | total_sims" -> bar chart
- Any two-column table with numeric values in the 2nd column -> bar chart
- "CORRELATIONS" sections -> a single figure listing the variable pairs
- It auto-strips %, thousands separators, and converts to numbers where possible

Usage:
  - Set CSV_PATH below, or
  - python3 plot_sections_per_metric.py --csv /path/to/report.csv

All figures are saved into a "plots" folder beside the CSV.
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= USER CONFIG =========
CSV_PATH = "/home/matteo/Simulation_rmp/Run_1000_1_290925/simulation_analysis_report_20251006_094651.csv"
DPI = 150
# ===============================

SECTION_RE = re.compile(r"^\s*===\s*(.+?)\s*===\s*$")

def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def out_dir(csv_path: Path) -> Path:
    try:
        p = csv_path.parent / "plots"
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        p = Path.cwd() / "plots"
        p.mkdir(parents=True, exist_ok=True)
    return p

def clean_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in s)

def savefig(outdir: Path, name: str) -> Path:
    p = outdir / f"{name}.png"
    plt.tight_layout()
    plt.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close()
    return p

def coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def read_sections(csv_path: Path) -> dict[str, pd.DataFrame]:
    # Read as strings to preserve report formatting
    df = pd.read_csv(csv_path, dtype=str)
    c0, c1 = df.columns[:2]
    rows = df[[c0, c1]].fillna("")
    sections: dict[str, pd.DataFrame] = {}
    current = None
    buf = []
    for _, (a, b) in rows.iterrows():
        a = str(a)
        b = str(b)
        m = SECTION_RE.match(a)
        if m:
            # flush previous
            if current and buf:
                sections[current] = pd.DataFrame(buf, columns=["col0", "col1"])
                buf = []
            current = m.group(1).strip()
            continue
        if current is None:
            continue
        buf.append([a, b])
    if current and buf:
        sections[current] = pd.DataFrame(buf, columns=["col0", "col1"])
    return sections

def normalize_table(tbl: pd.DataFrame) -> pd.DataFrame:
    """Drop blank lines, remove common header rows, and return cleaned table."""
    t = tbl.copy()
    # drop fully blank
    t = t[~((t["col0"].str.strip()=="") & (t["col1"].str.strip()==""))]
    # remove obvious headers
    col0_low = t["col0"].str.lower().str.strip()
    col1_low = t["col1"].str.lower().str.strip()
    header_like = (
        ((col0_low=="metric") & (col1_low.isin(["value","count","total","total_sims"])))
        | (col0_low.str.contains(r"variable\s*_?1", regex=True))
        | (col1_low.str.contains(r"variable\s*_?2", regex=True))
        | ((col0_low.str.contains("obstacle")) & (col1_low.str.contains("total")))
    )
    t = t[~header_like]
    return t

def plot_bar_for_two_column(section_name: str, tbl: pd.DataFrame, outdir: Path, tag: str) -> Path | None:
    t = normalize_table(tbl)
    if t.empty:
        return None
    names = t["col0"].astype(str).str.strip()
    vals = coerce_numeric(t["col1"])
    data = pd.DataFrame({"name": names, "value": vals}).dropna()
    if data.empty:
        return None
    # de-duplicate by name, keep first
    data = data.groupby("name", as_index=False)["value"].first()
    # choose orientation based on count
    if len(data) <= 15:
        # horizontal bars for readability
        plt.figure(figsize=(9, max(3.5, 0.35*len(data))))
        order = data.sort_values("value", ascending=True)
        plt.barh(order["name"], order["value"])
        plt.xlabel("value")
        plt.title(section_name)
    else:
        plt.figure(figsize=(max(9, 0.25*len(data)), 5))
        plt.bar(data["name"], data["value"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("value")
        plt.title(section_name)
    fname = f"{clean_filename(section_name)}_{tag}"
    return savefig(outdir, fname)

def plot_correlations(section_name: str, tbl: pd.DataFrame, outdir: Path, tag: str) -> Path | None:
    # List variable_1 / variable_2 pairs in one figure.
    t = normalize_table(tbl)
    if t.empty:
        return None
    pairs = t[~t["col0"].str.strip().eq("") & ~t["col1"].str.strip().eq("")][["col0","col1"]]
    if pairs.empty:
        return None
    lines = [f"{a}  —  {b}" for a,b in pairs.itertuples(index=False)]
    plt.figure(figsize=(12, max(2.5, 0.35*len(lines))))
    plt.axis("off")
    plt.title(section_name, pad=10)
    y = 0.95
    for line in lines:
        plt.text(0.02, y, line, va="top", ha="left", fontsize=10, family="monospace")
        y -= 0.06
        if y < 0.05:
            # add another column if long
            y = 0.95
    fname = f"{clean_filename(section_name)}_{tag}"
    return savefig(outdir, fname)

def main():
    ap = argparse.ArgumentParser(description="One-plot-per-metric from SIMULATION ANALYSIS REPORT CSV")
    ap.add_argument("--csv", type=str, default=None, help="Path to CSV")
    args = ap.parse_args()

    csv_path = Path(args.csv or CSV_PATH)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return 1

    outdir = out_dir(csv_path)
    tag = ts()

    print(f"Using CSV: {csv_path}")
    print(f"Saving plots to: {outdir}")

    sections = read_sections(csv_path)
    if not sections:
        print("[WARN] No sections detected (no '=== NAME ===' markers found).")
        return 2

    saved = []
    for name, tbl in sections.items():
        lname = name.lower()
        if "correlation" in lname:
            p = plot_correlations(name, tbl, outdir, tag)
        else:
            p = plot_bar_for_two_column(name, tbl, outdir, tag)
        if p:
            saved.append(p)

    print(f"\n✓ Finished. Generated {len(saved)} plots:")
    for p in saved:
        print(" -", p)

if __name__ == "__main__":
    main()