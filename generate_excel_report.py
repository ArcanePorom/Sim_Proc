from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import configparser
from datetime import datetime
import sys

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "reports"
# Default; can be overridden via Sim_Proc/config.ini [report].summary_top_n
SUMMARY_TOP_N = "100"  # Allowed: "10", "50", "100", "500", "All"

# Read summary_top_n from config if available
def _load_summary_top_n_from_config() -> str:
    cfg_path = ROOT / "config.ini"
    try:
        if cfg_path.exists():
            cp = configparser.ConfigParser()
            cp.read(cfg_path)
            if cp.has_section("report") and cp.has_option("report", "summary_top_n"):
                val = cp.get("report", "summary_top_n").strip()
                if val:
                    return val
    except Exception:
        pass
    return SUMMARY_TOP_N

# Initialize from config
SUMMARY_TOP_N = _load_summary_top_n_from_config()

# Minimal dependencies: pandas with openpyxl (writer) is fine

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column cases and types
    # Ensure common numeric fields are numeric
    num_cols = [
        "dps","priority_dps","range_pct_dps","targets","crit","haste","vers","mastery",
        "min_dps","max_dps","StdDev"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Replace infinities with NaN to avoid Excel writer issues
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def kpis(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["dps", "priority_dps", "range_pct_dps"]
    data = {
        "metric": [],
        "count": [],
        "mean": [],
        "median": [],
        "min": [],
        "max": [],
        "std": [],
    }
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        data["metric"].append(c)
        data["count"].append(int(s.count()))
        data["mean"].append(float(s.mean()) if not s.empty else None)
        data["median"].append(float(s.median()) if not s.empty else None)
        data["min"].append(float(s.min()) if not s.empty else None)
        data["max"].append(float(s.max()) if not s.empty else None)
        data["std"].append(float(s.std(ddof=0)) if not s.empty else None)
    return pd.DataFrame(data)


def pivots(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    piv = {}
    # DPS by build and targets
    if set(["builds", "targets", "dps"]).issubset(df.columns):
        piv["dps_by_build_targets"] = pd.pivot_table(
            df, index=["builds","targets"], values="dps", aggfunc=["mean","max","min","count"]
        )
    # Priority DPS by trinket combos (top 20)
    if set(["trinket_1","trinket_2","priority_dps"]).issubset(df.columns):
        pt = pd.pivot_table(
            df, index=["trinket_1","trinket_2"], values="priority_dps",
            aggfunc=["mean","max","count"]
        ).sort_values(("mean","priority_dps"), ascending=False)
        piv["priority_by_trinkets_top20"] = pt.head(20)
    # DPS by race/spec (if more races/specs appear later it scales)
    if set(["race","spec","dps"]).issubset(df.columns):
        piv["dps_by_race_spec"] = pd.pivot_table(
            df, index=["race","spec"], values="dps", aggfunc=["mean","max","count"]
        )
    # Talent presence impact: for each binary talent column, compute mean delta vs not taken
    talent_cols = [c for c in df.columns if c not in {
        "file","builds","build_id","race","class","spec","talent_str","trinket_1","trinket_2",
        "targets","crit","haste","vers","mastery","dps","priority_dps","range_pct_dps",
        "min_dps","max_dps","StdDev"
    } and df[c].dropna().isin([0,1,2]).all()]
    talent_rows = []
    for c in talent_cols:
        s = df[c].fillna(0)
        # treat >0 as taken
        taken = df.loc[s > 0]
        notaken = df.loc[s == 0]
        if not taken.empty and not notaken.empty and "dps" in df.columns:
            delta = float(taken["dps"].mean() - notaken["dps"].mean())
            talent_rows.append({"talent": c, "samples_taken": int(len(taken)), "samples_not": int(len(notaken)), "mean_dps_delta": delta})
    if talent_rows:
        piv["talent_mean_dps_delta"] = pd.DataFrame(talent_rows).sort_values("mean_dps_delta", ascending=False)
    return piv


def _detect_talent_columns(df: pd.DataFrame) -> list[str]:
    known = {
        "file","builds","build_id","race","class","spec","talent_str","trinket_1","trinket_2",
        "targets","crit","haste","vers","mastery","dps","priority_dps","range_pct_dps",
        "min_dps","max_dps","StdDev"
    }
    talent_cols: list[str] = []
    for c in df.columns:
        if c in known:
            continue
        s = df[c].dropna()
        if not len(s):
            continue
        unique_vals = set(pd.to_numeric(s, errors="coerce").dropna().unique().tolist())
        if unique_vals.issubset({0,1,2}):
            talent_cols.append(c)
    return talent_cols


def _build_top_table(df: pd.DataFrame, top_n: int | None) -> tuple[pd.DataFrame, list[int], list[str]]:
    # Collect unique targets and mean metrics for each combo
    targets_list = sorted([int(t) for t in pd.to_numeric(df["targets"], errors="coerce").dropna().unique().tolist()])
    desired_keys = ["talent_str", "race", "trinket_1", "trinket_2", "crit", "haste", "vers", "mastery"]
    # Exclude grouping keys that are entirely NaN (otherwise pandas groupby drops all rows)
    group_keys = [k for k in desired_keys if k in df.columns and df[k].notna().any()]

    agg = (
        df.dropna(subset=["targets"]) 
          .groupby(group_keys + ["targets"], as_index=False)[["dps", "priority_dps"]]
          .mean()
    )

    # Pivot to wide format: dps_{t}, priority_dps_{t}
    wide = agg.pivot_table(index=group_keys, columns="targets", values=["dps", "priority_dps"], aggfunc="mean")
    wide.columns = [f"{m}_{int(c)}" for m, c in wide.columns.to_flat_index()]
    wide = wide.reset_index()

    metric_cols = [c for c in wide.columns if c.startswith("dps_") or c.startswith("priority_dps_")]
    col_max = {c: float(wide[c].max()) if c in wide.columns else 0.0 for c in metric_cols}

    def row_weighted_avg(row: pd.Series) -> float:
        ratios = []
        for c in metric_cols:
            mv = float(row.get(c, np.nan))
            mx = col_max.get(c, 0.0)
            if np.isfinite(mv) and mx and mx > 0:
                ratios.append(mv / mx)
        if not ratios:
            return np.nan
        return float(np.mean(ratios))

    wide["weighted_avg"] = wide.apply(row_weighted_avg, axis=1)
    wide = wide.dropna(subset=["weighted_avg"]).sort_values("weighted_avg", ascending=False)
    if top_n is None:
        top = wide.reset_index(drop=True)
    else:
        top = wide.head(top_n).reset_index(drop=True)
    return top, targets_list, metric_cols

def _write_summary_sheet(df: pd.DataFrame, xl: pd.ExcelWriter, source_csv_name: str):
    # Resolve Top N selection
    sel = (SUMMARY_TOP_N or "100").strip()
    top_n: int | None
    if sel.lower() == "all":
        top_n = None
    else:
        try:
            top_n = int(sel)
        except Exception:
            top_n = 100
    top, targets_list, metric_cols = _build_top_table(df, top_n)
    # Debug: print size for diagnostics
    try:
        print(f"[Summary] Top builds rows: {len(top)} (targets: {targets_list})")
    except Exception:
        pass
    talent_cols = _detect_talent_columns(df)

    desired_keys = ["talent_str", "race", "trinket_1", "trinket_2", "crit", "haste", "vers", "mastery"]
    group_keys = [k for k in desired_keys if k in df.columns and df[k].notna().any()]

    def canonical_talents(sub: pd.DataFrame) -> dict[str, int]:
        res: dict[str, int] = {}
        for c in talent_cols:
            s = (pd.to_numeric(sub[c], errors="coerce").fillna(0) > 0).astype(int)
            res[c] = int((s.mean() >= 0.5)) if len(s) else 0
        return res

    talents_by_key: dict[tuple, dict[str, int]] = {}
    for key, sub in df.groupby(group_keys):
        talents_by_key[key] = canonical_talents(sub)

    best_tal = {c: 0 for c in talent_cols}
    if len(top) > 0:
        best_key = tuple(top.loc[0, k] for k in group_keys)
        best_tal = talents_by_key.get(best_key, best_tal)

    wb = xl.book
    ws = wb.add_worksheet("Summary")

    fmt_bold = wb.add_format({"bold": True})
    fmt_header = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
    fmt_green = wb.add_format({"font_color": "#00B050"})
    fmt_red = wb.add_format({"font_color": "#C00000"})
    # Number formats
    fmt_int0 = wb.add_format({"num_format": "#,##0"})
    fmt_pct1 = wb.add_format({"num_format": "0.0%"})

    # Header info
    ws.write(0, 0, "Simulation Report", fmt_bold)  # A1
    ws.write(1, 0, f"Source CSV: {source_csv_name}")  # A2
    ws.write(2, 0, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  # A3
    # Data quality note for missing optional columns
    optional_cols = ["trinket_1", "trinket_2"]
    missing_optional = [c for c in optional_cols if c not in df.columns]
    if missing_optional:
        ws.write(3, 0, f"Missing optional columns: {', '.join(missing_optional)}")  # A4 (before title)

    title = "All builds" if top_n is None else f"Top {top_n} builds"
    ws.write(4, 0, title, fmt_bold)  # A5

    # Table headers at row A6
    # Only include trinket columns if present to avoid confusing empty columns
    base_headers = ["talent_str"] \
        + (["race"] if "race" in df.columns else []) \
        + (["trinket_1"] if "trinket_1" in df.columns else []) \
        + (["trinket_2"] if "trinket_2" in df.columns else []) \
        + ["changed_talents", "crit", "haste", "vers", "mastery"]
    # Display-friendly headers for target metrics
    dyn_headers: list[str] = []
    for t in targets_list:
        dyn_headers.append(f"{t} Target DPS")
        dyn_headers.append(f"{t} Target Priority")
    headers = base_headers + dyn_headers + ["Weighted Avg"]

    header_row = 5
    for col_idx, h in enumerate(headers):
        ws.write(header_row, col_idx, h, fmt_header)

    def write_changed_talents(r: int, c: int, current_key: tuple):
        cur_tal = talents_by_key.get(current_key, {cc: 0 for cc in talent_cols})
        diffs = [name for name in talent_cols if cur_tal.get(name, 0) != best_tal.get(name, 0)]
        if not diffs:
            ws.write(r, c, "")
            return
        rich_parts: list = []
        for i, name in enumerate(diffs):
            if i > 0:
                rich_parts.append(", ")
            fmt = fmt_green if int(cur_tal.get(name, 0) > 0) == 1 else fmt_red
            rich_parts.extend([fmt, name])
        ws.write_rich_string(r, c, *rich_parts)

    start_row = 6
    if len(top) == 0:
        # Provide a visible note if no rows were generated
        ws.write(start_row, 0, "No summary rows generated (insufficient data after grouping or all metrics NaN).")
    else:
        for i in range(len(top)):
            row = top.iloc[i]
            current_key = tuple(row.get(k, "") for k in group_keys)
            col = 0
            for h in base_headers:
                if h == "changed_talents":
                    write_changed_talents(start_row + i, col, current_key)
                elif h in ("crit","haste","vers","mastery"):
                    val = row.get(h, "")
                    ws.write(start_row + i, col, float(val) if pd.notna(val) else "")
                else:
                    ws.write(start_row + i, col, row.get(h, ""))
                col += 1
            for t in targets_list:
                ws.write(start_row + i, col, float(row.get(f"dps_{t}", np.nan)) if pd.notna(row.get(f"dps_{t}", np.nan)) else ""); col += 1
                ws.write(start_row + i, col, float(row.get(f"priority_dps_{t}", np.nan)) if pd.notna(row.get(f"priority_dps_{t}", np.nan)) else ""); col += 1
            ws.write(start_row + i, col, float(row.get("weighted_avg", np.nan)) if pd.notna(row.get("weighted_avg", np.nan)) else ""); col += 1

    # Conditional color scale per metric column
    dps_start_col = len(base_headers)  # first metric col is after all base headers
    total_cols = len(headers)
    last_row = start_row + len(top) - 1 if len(top) > 0 else start_row
    color_scale = {
        "type": "3_color_scale",
        "min_color": "#FF0000",
        "mid_color": "#FFD966",
        "max_color": "#00B050",
    }
    for c in range(dps_start_col, total_cols):
        ws.conditional_format(start_row, c, last_row, c, color_scale)

    # Column widths and formats
    # Set column widths dynamically based on header meaning
    for col_idx, h in enumerate(base_headers):
        if h == "talent_str":
            ws.set_column(col_idx, col_idx, 48)
        elif h == "race":
            ws.set_column(col_idx, col_idx, 14)
        elif h in ("trinket_1","trinket_2"):
            ws.set_column(col_idx, col_idx, 20)
        elif h == "changed_talents":
            ws.set_column(col_idx, col_idx, 50)
        elif h in ("crit","haste","vers","mastery"):
            ws.set_column(col_idx, col_idx, 10, fmt_int0)
        else:
            ws.set_column(col_idx, col_idx, 16)
    # Metrics (DPS/Prio) no decimals
    for col_idx in range(dps_start_col, total_cols - 1):
        ws.set_column(col_idx, col_idx, 16, fmt_int0)
    # Weighted Avg as percentage with 1 decimal
    ws.set_column(total_cols - 1, total_cols - 1, 14, fmt_pct1)

    # Enable AutoFilter for the table
    ws.autofilter(header_row, 0, last_row, total_cols - 1)


def _excel_col_letter(idx0: int) -> str:
    # Convert 0-based column index to Excel column letters
    s = ""
    n = idx0 + 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _build_pivot_dimension(df: pd.DataFrame, targets: list[int], dim: str, talent_cols: list[str]) -> pd.DataFrame:
    # Return DataFrame with columns: label, dps_<t>, priority_dps_<t>, weighted_avg
    metrics_cols = []
    for t in targets:
        metrics_cols.extend([f"dps_{t}", f"priority_dps_{t}"])

    def finalize(part: pd.DataFrame) -> pd.DataFrame:
        # Ensure all metric columns exist even for empty parts so downstream selection works
        part = part.copy()
        for c in metrics_cols:
            if c not in part.columns:
                part[c] = np.nan
        # Compute weighted_avg per-part based on per-column max within part
        col_max = {c: float(part[c].max()) if c in part.columns else 0.0 for c in metrics_cols}
        def wavgr(row: pd.Series) -> float:
            ratios = []
            for c in metrics_cols:
                if c in row and pd.notna(row[c]) and col_max.get(c, 0) > 0:
                    ratios.append(float(row[c]) / col_max[c])
            return float(np.mean(ratios)) if ratios else np.nan
        part["weighted_avg"] = part.apply(wavgr, axis=1)
        return part

    if dim == "talent_str":
        group_keys = ["talent_str", "targets"]
        agg = df.groupby(group_keys, as_index=False)[["dps","priority_dps"]].mean()
        wide = agg.pivot_table(index=["talent_str"], columns="targets", values=["dps","priority_dps"], aggfunc="mean")
        wide.columns = [f"{m}_{int(c)}" for m, c in wide.columns.to_flat_index()]
        wide = wide.reset_index().rename(columns={"talent_str":"label"})
        for c in metrics_cols:
            if c not in wide.columns:
                wide[c] = np.nan
        return finalize(wide[["label"] + metrics_cols])

    if dim == "race":
        group_keys = ["race", "targets"]
        agg = df.groupby(group_keys, as_index=False)[["dps","priority_dps"]].mean()
        wide = agg.pivot_table(index=["race"], columns="targets", values=["dps","priority_dps"], aggfunc="mean")
        wide.columns = [f"{m}_{int(c)}" for m, c in wide.columns.to_flat_index()]
        wide = wide.reset_index().rename(columns={"race":"label"})
        for c in metrics_cols:
            if c not in wide.columns:
                wide[c] = np.nan
        return finalize(wide[["label"] + metrics_cols])

    if dim == "trinket":
        # Combine trinket_1 and trinket_2 into a single label space; each row contributes to both if present
        frames: list[pd.DataFrame] = []
        if "trinket_1" in df.columns:
            sub1 = df[["trinket_1", "targets", "dps", "priority_dps"]].copy()
            sub1 = sub1.rename(columns={"trinket_1": "label"})
            frames.append(sub1)
        if "trinket_2" in df.columns:
            sub2 = df[["trinket_2", "targets", "dps", "priority_dps"]].copy()
            sub2 = sub2.rename(columns={"trinket_2": "label"})
            frames.append(sub2)
        if not frames:
            return finalize(pd.DataFrame({"label": []}))
        comb = pd.concat(frames, ignore_index=True)
        # Drop empty labels
        comb = comb[pd.notna(comb["label"]) & (comb["label"].astype(str).str.len() > 0)]
        if comb.empty:
            return finalize(pd.DataFrame({"label": []}))
        agg = comb.groupby(["label", "targets"], as_index=False)[["dps", "priority_dps"]].mean()
        wide = agg.pivot_table(index=["label"], columns="targets", values=["dps","priority_dps"], aggfunc="mean")
        wide.columns = [f"{m}_{int(c)}" for m, c in wide.columns.to_flat_index()]
        wide = wide.reset_index()
        for c in metrics_cols:
            if c not in wide.columns:
                wide[c] = np.nan
        return finalize(wide[["label"] + metrics_cols])

    if dim == "stat_profile":
        # Create stat label from crit/haste/vers/mastery
        df_sp = df.copy()
        def fmt_stat(v):
            try:
                f = float(v)
                return str(int(f)) if abs(f - int(f)) < 1e-6 else str(f)
            except Exception:
                return ""
        df_sp["label"] = df_sp.apply(lambda r: f"{fmt_stat(r.get('crit',''))}/{fmt_stat(r.get('haste',''))}/{fmt_stat(r.get('vers',''))}/{fmt_stat(r.get('mastery',''))}", axis=1)
        group_keys = ["label", "targets"]
        agg = df_sp.groupby(group_keys, as_index=False)[["dps","priority_dps"]].mean()
        wide = agg.pivot_table(index=["label"], columns="targets", values=["dps","priority_dps"], aggfunc="mean")
        wide.columns = [f"{m}_{int(c)}" for m, c in wide.columns.to_flat_index()]
        wide = wide.reset_index()
        for c in metrics_cols:
            if c not in wide.columns:
                wide[c] = np.nan
        return finalize(wide[["label"] + metrics_cols])

    if dim == "trinket_pair":
        df_tp = df.copy()
        df_tp["label"] = df_tp.apply(lambda r: f"{r.get('trinket_1','')} + {r.get('trinket_2','')}", axis=1)
        group_keys = ["label", "targets"]
        agg = df_tp.groupby(group_keys, as_index=False)[["dps","priority_dps"]].mean()
        wide = agg.pivot_table(index=["label"], columns="targets", values=["dps","priority_dps"], aggfunc="mean")
        wide.columns = [f"{m}_{int(c)}" for m, c in wide.columns.to_flat_index()]
        wide = wide.reset_index()
        for c in metrics_cols:
            if c not in wide.columns:
                wide[c] = np.nan
        return finalize(wide[["label"] + metrics_cols])

    if dim == "individual talents":
        rows = []
        for tal in talent_cols:
            mask = pd.to_numeric(df[tal], errors="coerce").fillna(0) > 0
            if not mask.any():
                continue
            sub = df.loc[mask]
            agg = sub.groupby(["targets"], as_index=False)[["dps","priority_dps"]].mean()
            # Turn into single row with columns
            row = {"label": tal}
            for _, r in agg.iterrows():
                t = int(r["targets"]) if pd.notna(r["targets"]) else None
                if t is not None:
                    row[f"dps_{t}"] = float(r["dps"]) if pd.notna(r["dps"]) else np.nan
                    row[f"priority_dps_{t}"] = float(r["priority_dps"]) if pd.notna(r["priority_dps"]) else np.nan
            rows.append(row)
        part = pd.DataFrame(rows)
        if part.empty:
            part = pd.DataFrame({"label": []})
        for c in metrics_cols:
            if c not in part.columns:
                part[c] = np.nan
        return finalize(part[["label"] + metrics_cols])

    # Fallback empty
    return finalize(pd.DataFrame({"label": []}))


def _write_pivots_sheet(df: pd.DataFrame, xl: pd.ExcelWriter):
    wb = xl.book
    ws_data = wb.add_worksheet("pivot_data")
    ws = wb.add_worksheet("Pivots")

    # Build dataset for all dimensions
    targets = sorted([int(t) for t in pd.to_numeric(df["targets"], errors="coerce").dropna().unique().tolist()])
    talent_cols = _detect_talent_columns(df)
    dims = ["talent_str", "race", "trinket", "stat_profile", "trinket_pair", "individual talents"]

    dps_headers = [f"dps_{t}" for t in targets]
    prio_headers = [f"priority_dps_{t}" for t in targets]
    both_headers = []
    # Display-friendly headers
    dps_disp_headers = [f"{t} Target DPS" for t in targets]
    prio_disp_headers = [f"{t} Target Priority" for t in targets]
    both_disp_headers: list[str] = []
    for t in targets:
        both_headers.extend([f"dps_{t}", f"priority_dps_{t}"])
        both_disp_headers.extend([f"{t} Target DPS", f"{t} Target Priority"])

    # Write blocks per dimension and metric view, define named ranges
    cur_row = 0
    max_rows = 0
    for dim in dims:
        base = _build_pivot_dimension(df, targets, dim, talent_cols)
        # Sort by weighted_avg desc for nicer default ordering
        if "weighted_avg" in base.columns:
            base = base.sort_values("weighted_avg", ascending=False).reset_index(drop=True)

        base_headers = ["label"] + both_headers + ["weighted_avg"]
        base_disp_headers = ["label"] + both_disp_headers + ["weighted_avg"]
        dps_only_cols = ["label"] + dps_headers + ["weighted_avg"]
        dps_only_disp = ["label"] + dps_disp_headers + ["weighted_avg"]
        prio_only_cols = ["label"] + prio_headers + ["weighted_avg"]
        prio_only_disp = ["label"] + prio_disp_headers + ["weighted_avg"]

        # BOTH block
        hdr_row = cur_row
        # Write display headers while keeping data selection by internal column names
        ws_data.write_row(hdr_row, 0, base_disp_headers)
        data_rows = base[base_headers]
        for r_i, (_, r) in enumerate(data_rows.iterrows(), start=1):
            ws_data.write_row(hdr_row + r_i, 0, r.tolist())
        rows_count = len(data_rows)
        max_rows = max(max_rows, rows_count)
        last_col_letter = _excel_col_letter(len(base_headers) - 1)
        # Define names
        name_base = f"pivot_{dim.replace(' ','_')}_both"
        wb.define_name(name_base + "_headers", f"=pivot_data!$A${hdr_row+1}:${last_col_letter}${hdr_row+1}")
        # Ensure at least one blank data row exists when empty to keep the named range valid
        if rows_count == 0:
            ws_data.write_row(hdr_row + 1, 0, [None] * len(base_headers))
        rows_count_for_range = max(1, rows_count)
        wb.define_name(name_base + "_data", f"=pivot_data!$A${hdr_row+2}:${last_col_letter}${hdr_row+1+rows_count_for_range}")
        cur_row = hdr_row + rows_count + 2

        # DPS block
        hdr_row = cur_row
        ws_data.write_row(hdr_row, 0, dps_only_disp)
        data_rows = base[dps_only_cols]
        for r_i, (_, r) in enumerate(data_rows.iterrows(), start=1):
            ws_data.write_row(hdr_row + r_i, 0, r.tolist())
        rows_count = len(data_rows)
        max_rows = max(max_rows, rows_count)
        last_col_letter = _excel_col_letter(len(dps_only_cols) - 1)
        name_base = f"pivot_{dim.replace(' ','_')}_dps"
        wb.define_name(name_base + "_headers", f"=pivot_data!$A${hdr_row+1}:${last_col_letter}${hdr_row+1}")
        if rows_count == 0:
            ws_data.write_row(hdr_row + 1, 0, [None] * len(dps_only_cols))
        rows_count_for_range = max(1, rows_count)
        wb.define_name(name_base + "_data", f"=pivot_data!$A${hdr_row+2}:${last_col_letter}${hdr_row+1+rows_count_for_range}")
        cur_row = hdr_row + rows_count + 2

        # PRIORITY block
        hdr_row = cur_row
        ws_data.write_row(hdr_row, 0, prio_only_disp)
        data_rows = base[prio_only_cols]
        for r_i, (_, r) in enumerate(data_rows.iterrows(), start=1):
            ws_data.write_row(hdr_row + r_i, 0, r.tolist())
        rows_count = len(data_rows)
        max_rows = max(max_rows, rows_count)
        last_col_letter = _excel_col_letter(len(prio_only_cols) - 1)
        name_base = f"pivot_{dim.replace(' ','_')}_priority_dps"
        wb.define_name(name_base + "_headers", f"=pivot_data!$A${hdr_row+1}:${last_col_letter}${hdr_row+1}")
        if rows_count == 0:
            ws_data.write_row(hdr_row + 1, 0, [None] * len(prio_only_cols))
        rows_count_for_range = max(1, rows_count)
        wb.define_name(name_base + "_data", f"=pivot_data!$A${hdr_row+2}:${last_col_letter}${hdr_row+1+rows_count_for_range}")
        cur_row = hdr_row + rows_count + 2

    ws_data.hide()

    # Pivots UI
    fmt_bold = wb.add_format({"bold": True})
    fmt_header = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
    ws.write(0, 0, "Dimension:", fmt_bold)
    ws.data_validation(0, 1, 0, 1, {
        'validate': 'list',
        'source': dims
    })
    ws.write(0, 3, "Metric:", fmt_bold)
    ws.data_validation(0, 4, 0, 4, {
        'validate': 'list',
        'source': ["both", "dps", "priority_dps"]
    })
    # Default selections
    ws.write(0, 1, "talent_str")
    ws.write(0, 4, "both")
    ws.write(1, 0, "Select a dimension and metric set to view the pivot table.")

    # Headers row at A5
    header_row = 4
    # Add Index header in column A
    ws.write(header_row, 0, "Index", fmt_header)
    # Label header in column B
    ws.write_formula(header_row, 1, '=IF(LEN($B$1)=0,"(select dimension)",$B$1)', fmt_header)
    # Build selected header name
    sel_hdr_name = '"pivot_" & SUBSTITUTE($B$1," ","_") & "_" & IF(LEN($E$1)=0,"both",$E$1) & "_headers"'
    # Fill headers across max columns (index + label + metrics + weighted)
    max_cols = 2 + len(both_headers) + 1
    for c in range(2, max_cols):
        # Dynamic headers start at C5 now. Skip the 'label' header in named headers (position 1)
        ws.write_formula(header_row, c, f'=IFERROR(INDEX(INDIRECT({sel_hdr_name}),1,COLUMN()-COLUMN($C$5)+2),"")', fmt_header)

    # Data grid starting A6
    start_row = 5
    sel_data_name = '"pivot_" & SUBSTITUTE($B$1," ","_") & "_" & IF(LEN($E$1)=0,"both",$E$1) & "_data"'
    # Fill a grid of max_rows x max_cols with INDEX/INDIRECT
    for r in range(max_rows):
        for c in range(max_cols):
            if c == 0:
                # Hard index values 1..max_rows (anchors rows for manual sorting)
                ws.write(start_row + r, c, r + 1)
                continue
            elif c == 1:
                # label column anchored to index in col A
                excel_this_row = start_row + r + 1
                formula = f'=IF($A{excel_this_row}<=ROWS(INDIRECT({sel_data_name})),INDEX(INDIRECT({sel_data_name}),$A{excel_this_row},1),"")'
            else:
                # metrics anchored to index in col A; map sheet col C.. to named range col 2..
                excel_this_row = start_row + r + 1
                formula = f'=IF($A{excel_this_row}<=ROWS(INDIRECT({sel_data_name})),IF(COLUMN()-COLUMN($C$5)+2<=COLUMNS(INDIRECT({sel_data_name})),INDEX(INDIRECT({sel_data_name}),$A{excel_this_row},COLUMN()-COLUMN($C$5)+2),""),"")'
            ws.write_formula(start_row + r, c, formula)

    # Conditional formatting on numeric columns (C..last)
    color_scale = {
        'type': '3_color_scale',
        'min_color': '#FF0000',
        'mid_color': '#FFD966',
        'max_color': '#00B050',
    }
    # Apply color scales per column (by-column)
    for c in range(2, max_cols):
        ws.conditional_format(start_row, c, start_row + max_rows, c, color_scale)
    # Column widths
    ws.set_column(0, 0, 6)   # Index
    ws.set_column(1, 1, 40)  # Label
    for c in range(2, max_cols):
        ws.set_column(c, c, 16)

    # Hide Index for empty rows via conditional formatting (when Label is blank)
    fmt_hide = wb.add_format({'font_color': '#FFFFFF'})
    ws.conditional_format(start_row, 0, start_row + max_rows, 0, {
        'type': 'formula',
        'criteria': f'=$B${start_row+1}=""',
        'format': fmt_hide,
    })

    # Enable AutoFilter for the pivots grid (includes Index and Label)
    ws.autofilter(header_row, 0, start_row + max_rows - 1, max_cols - 1)


def _build_single_talent_table(df: pd.DataFrame, targets: list[int], talent_cols: list[str]) -> pd.DataFrame:
    rows = []
    for tal in talent_cols:
        s = pd.to_numeric(df[tal], errors="coerce").fillna(0)
        taken_mask = s > 0
        not_mask = ~taken_mask
        taken = df.loc[taken_mask]
        not_taken = df.loc[not_mask]
        samples_taken = int(taken.shape[0])
        samples_not = int(not_taken.shape[0])
        if samples_taken == 0 and samples_not == 0:
            continue
        taken_rate = float(samples_taken / max(1, samples_taken + samples_not))
        mean_dps_taken = float(taken["dps"].mean()) if samples_taken else np.nan
        mean_dps_not = float(not_taken["dps"].mean()) if samples_not else np.nan
        mean_dps_delta = float(mean_dps_taken - mean_dps_not) if (pd.notna(mean_dps_taken) and pd.notna(mean_dps_not)) else np.nan
        row = {
            "talent": tal,
            "taken_rate": taken_rate,
            "samples_taken": samples_taken,
            "samples_not": samples_not,
            "mean_dps_delta": mean_dps_delta,
        }
        # Per-target means when taken
        if samples_taken:
            agg = (
                taken.dropna(subset=["targets"])
                     .groupby(["targets"], as_index=False)[["dps","priority_dps"]]
                     .mean()
            )
            for _, r in agg.iterrows():
                t = int(r["targets"]) if pd.notna(r["targets"]) else None
                if t is not None:
                    row[f"dps_{t}"] = float(r["dps"]) if pd.notna(r["dps"]) else np.nan
                    row[f"priority_dps_{t}"] = float(r["priority_dps"]) if pd.notna(r["priority_dps"]) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame({
            "talent": [],
            "taken_rate": [],
            "samples_taken": [],
            "samples_not": [],
            "mean_dps_delta": [],
        })
    # Ensure all metric columns exist
    metric_cols: list[str] = []
    for t in targets:
        metric_cols.extend([f"dps_{t}", f"priority_dps_{t}"])
    for c in metric_cols:
        if c not in out.columns:
            out[c] = np.nan
    # Weighted avg across available per-target metrics
    col_max = {c: float(out[c].max()) if c in out.columns else 0.0 for c in metric_cols}
    def wavgr(row: pd.Series) -> float:
        ratios = []
        for c in metric_cols:
            if c in row and pd.notna(row[c]) and col_max.get(c, 0) > 0:
                ratios.append(float(row[c]) / col_max[c])
        return float(np.mean(ratios)) if ratios else np.nan
    out["weighted_avg"] = out.apply(wavgr, axis=1)
    # Sort by mean_dps_delta desc as primary, then weighted_avg
    out = out.sort_values(["mean_dps_delta","weighted_avg"], ascending=[False, False], na_position="last").reset_index(drop=True)
    return out


def _build_talent_synergy_table(df: pd.DataFrame, targets: list[int], talent_cols: list[str], top_n: int = 20, min_co: int = 10) -> pd.DataFrame:
    # Limit to top talents by taken frequency
    freq = []
    total_rows = df.shape[0]
    for tal in talent_cols:
        taken_count = int((pd.to_numeric(df[tal], errors="coerce").fillna(0) > 0).sum())
        freq.append((tal, taken_count))
    freq.sort(key=lambda x: x[1], reverse=True)
    base_talents = [t for t, _ in freq[:top_n]]

    rows = []
    # Helper: mean dps for a mask
    def mean_dps_for(mask: pd.Series) -> float:
        sub = df.loc[mask]
        return float(sub["dps"].mean()) if not sub.empty else np.nan

    for i in range(len(base_talents)):
        t1 = base_talents[i]
        s1 = pd.to_numeric(df[t1], errors="coerce").fillna(0)
        t1_taken = s1 > 0
        # More available baseline: mean when t1 is taken (regardless of others)
        mean_t1_any = mean_dps_for(t1_taken)
        for j in range(i+1, len(base_talents)):
            t2 = base_talents[j]
            s2 = pd.to_numeric(df[t2], errors="coerce").fillna(0)
            t2_taken = s2 > 0
            both = t1_taken & t2_taken
            co_count = int(both.sum())
            if co_count < min_co:
                continue
            # per-target means when both taken
            sub = df.loc[both]
            row = {"pair": f"{t1} + {t2}", "co_samples": co_count}
            agg = (
                sub.dropna(subset=["targets"])
                   .groupby(["targets"], as_index=False)[["dps","priority_dps"]]
                   .mean()
            )
            for _, r in agg.iterrows():
                t = int(r["targets"]) if pd.notna(r["targets"]) else None
                if t is not None:
                    row[f"dps_{t}"] = float(r["dps"]) if pd.notna(r["dps"]) else np.nan
                    row[f"priority_dps_{t}"] = float(r["priority_dps"]) if pd.notna(r["priority_dps"]) else np.nan
            mean_both = float(sub["dps"].mean()) if not sub.empty else np.nan
            # Baselines: mean when t2 is taken (regardless), mean when neither of pair is taken, and overall mean
            mean_t2_any = mean_dps_for(t2_taken)
            none_pair = (~t1_taken) & (~t2_taken)
            mean_none_pair = mean_dps_for(none_pair)
            overall_mean = float(df["dps"].mean()) if "dps" in df.columns and not df["dps"].dropna().empty else np.nan
            # Use the best available baseline to represent non-synergistic expectation
            candidates = [mean_t1_any, mean_t2_any, mean_none_pair, overall_mean]
            candidates = [x for x in candidates if pd.notna(x)]
            baseline = float(np.max(candidates)) if candidates else np.nan
            synergy_score = float(mean_both - baseline) if (pd.notna(mean_both) and pd.notna(baseline)) else np.nan
            row["synergy_score"] = synergy_score
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame({"pair": [], "co_samples": [], "synergy_score": []})
    # Ensure metric columns exist
    metric_cols: list[str] = []
    for t in targets:
        metric_cols.extend([f"dps_{t}", f"priority_dps_{t}"])
    for c in metric_cols:
        if c not in out.columns:
            out[c] = np.nan
    # Weighted avg across available metrics
    col_max = {c: float(out[c].max()) if c in out.columns else 0.0 for c in metric_cols}
    def wavgr(row: pd.Series) -> float:
        ratios = []
        for c in metric_cols:
            if c in row and pd.notna(row[c]) and col_max.get(c, 0) > 0:
                ratios.append(float(row[c]) / col_max[c])
        return float(np.mean(ratios)) if ratios else np.nan
    out["weighted_avg"] = out.apply(wavgr, axis=1)
    # Sort by weighted_avg primarily, then synergy_score
    out = out.sort_values(["weighted_avg","synergy_score"], ascending=[False, False], na_position="last").head(100).reset_index(drop=True)
    return out


def _write_talents_sheet(df: pd.DataFrame, xl: pd.ExcelWriter):
    wb = xl.book
    ws = wb.add_worksheet("Talents")
    fmt_bold = wb.add_format({"bold": True})
    fmt_header = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})

    targets = sorted([int(t) for t in pd.to_numeric(df["targets"], errors="coerce").dropna().unique().tolist()])
    talent_cols = _detect_talent_columns(df)
    if not talent_cols:
        ws.write(0, 0, "No talent columns detected.", fmt_bold)
        return

    # Section 1: Single talent analytics
    ws.write(0, 0, "Single Talents", fmt_bold)
    single = _build_single_talent_table(df, targets, talent_cols)
    # Drop talents with no comparison group (always taken or never taken)
    if "samples_not" in single.columns:
        single = single[single["samples_not"] > 0]
    if "samples_taken" in single.columns:
        single = single[single["samples_taken"] > 0]
    single = single.reset_index(drop=True)
    # Sort single talents by Mean Delta (mean_dps_delta) descending
    if "mean_dps_delta" in single.columns:
        single = single.sort_values("mean_dps_delta", ascending=False, na_position="last").reset_index(drop=True)
    # Headers
    # Display headers: rename per request and drop samples_not
    single_base_headers = ["talent", "Pick Rate", "Sample Size", "Mean Delta"]
    single_metric_headers = []
    for t in targets:
        single_metric_headers.extend([f"{t} Target DPS", f"{t} Target Priority"])
    # Display header is "Weighted Avg" but we keep underlying column name weighted_avg for values
    single_headers = single_base_headers + single_metric_headers + ["Weighted Avg"]
    # Write headers
    for c, h in enumerate(single_headers):
        ws.write(2, c, h, fmt_header)
    # Write rows
    for i in range(len(single)):
        r = 3 + i
        ws.write(r, 0, single.loc[i, "talent"])  # talent
        ws.write(r, 1, float(single.loc[i, "taken_rate"]) if pd.notna(single.loc[i, "taken_rate"]) else "")  # Pick Rate
        ws.write(r, 2, int(single.loc[i, "samples_taken"]) if pd.notna(single.loc[i, "samples_taken"]) else "")  # Sample Size
        ws.write(r, 3, float(single.loc[i, "mean_dps_delta"]) if pd.notna(single.loc[i, "mean_dps_delta"]) else "")  # Mean Delta
        col = 4
        for t in targets:
            ws.write(r, col, float(single.loc[i, f"dps_{t}"]) if f"dps_{t}" in single.columns and pd.notna(single.loc[i, f"dps_{t}"]) else ""); col += 1
            ws.write(r, col, float(single.loc[i, f"priority_dps_{t}"]) if f"priority_dps_{t}" in single.columns and pd.notna(single.loc[i, f"priority_dps_{t}"]) else ""); col += 1
        ws.write(r, col, float(single.loc[i, "weighted_avg"]) if pd.notna(single.loc[i, "weighted_avg"]) else "")

    # Formatting for single section
    color_scale = {
        'type': '3_color_scale',
        'min_color': '#FF0000',
        'mid_color': '#FFD966',
        'max_color': '#00B050',
    }
    last_row_single = (3 + len(single) - 1) if len(single) > 0 else 3
    # Per-column color scale for numeric metrics (skip first column talent)
    for c in range(1, len(single_headers)):
        ws.conditional_format(3, c, last_row_single, c, color_scale)
    # Column widths
    ws.set_column(0, 0, 36)  # talent
    ws.set_column(1, 1, 12)  # Pick Rate
    ws.set_column(2, 2, 14)  # Sample Size
    ws.set_column(3, 3, 16)  # Mean Delta
    # Metrics
    total_cols_single = len(single_headers)
    for c in range(4, total_cols_single):
        ws.set_column(c, c, 16)
    # Enable AutoFilter for Single Talents table (Excel supports one AutoFilter per sheet)
    ws.autofilter(2, 0, last_row_single, total_cols_single - 1)

    # Section 2: Talent synergies
    start_synergy_row = (len(single) + 6)
    ws.write(start_synergy_row, 0, "Talent Synergies (pairs)", fmt_bold)
    synergy = _build_talent_synergy_table(df, targets, talent_cols)
    # Sorted within builder by Weighted Avg primarily
    # Headers
    syn_base_headers = ["pair", "Sample Size", "Synergy Score"]
    syn_metric_headers = []
    for t in targets:
        syn_metric_headers.extend([f"{t} Target DPS", f"{t} Target Priority"])
    syn_headers = syn_base_headers + syn_metric_headers + ["Weighted Avg"]
    # Write headers
    for c, h in enumerate(syn_headers):
        ws.write(start_synergy_row + 2, c, h, fmt_header)
    # Write rows
    for i in range(len(synergy)):
        r = start_synergy_row + 3 + i
        ws.write(r, 0, synergy.loc[i, "pair"])  # label
        ws.write(r, 1, int(synergy.loc[i, "co_samples"]) if pd.notna(synergy.loc[i, "co_samples"]) else "")
        ws.write(r, 2, float(synergy.loc[i, "synergy_score"]) if pd.notna(synergy.loc[i, "synergy_score"]) else "")
        col = 3
        for t in targets:
            ws.write(r, col, float(synergy.loc[i, f"dps_{t}"]) if f"dps_{t}" in synergy.columns and pd.notna(synergy.loc[i, f"dps_{t}"]) else ""); col += 1
            ws.write(r, col, float(synergy.loc[i, f"priority_dps_{t}"]) if f"priority_dps_{t}" in synergy.columns and pd.notna(synergy.loc[i, f"priority_dps_{t}"]) else ""); col += 1
        ws.write(r, col, float(synergy.loc[i, "weighted_avg"]) if pd.notna(synergy.loc[i, "weighted_avg"]) else "")

    # Formatting for synergy section
    last_row_syn = start_synergy_row + 2 + max(1, len(synergy))
    for c in range(1, len(syn_headers)):
        ws.conditional_format(start_synergy_row + 3, c, last_row_syn, c, color_scale)
    # Column widths
    ws.set_column(0, 0, 42)  # pair label
    ws.set_column(1, 1, 12)  # co_samples
    ws.set_column(2, 2, 16)  # synergy score
    total_cols_syn = len(syn_headers)
    for c in range(3, total_cols_syn):
        ws.set_column(c, c, 16)
    # Note: Only one sheet-level AutoFilter is supported; leaving Synergies without a filter here.


def _write_stat_heatmap_sheet(df: pd.DataFrame, xl: pd.ExcelWriter):
    wb = xl.book
    ws = wb.add_worksheet("Stat Heatmap")

    fmt_bold = wb.add_format({"bold": True})
    fmt_header = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
    fmt_int0 = wb.add_format({"num_format": "#,##0"})

    # Helper: coerce to numeric
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")
    # Helper: header-friendly display of numbers
    def fmt_val(v):
        try:
            f = float(v)
            return int(f) if abs(f - int(f)) < 1e-6 else f
        except Exception:
            return v

    # Determine which stats we can plot
    all_stats = ["crit", "haste", "vers", "mastery"]
    present = [s for s in all_stats if s in df.columns]
    if "dps" not in df.columns or len(present) < 2:
        ws.write(0, 0, "Stat columns not found (need dps and at least two of crit/haste/vers/mastery).", fmt_bold)
        return

    ws.write(0, 0, "Stat Profile Heatmaps — Pairwise (Mean DPS)", fmt_bold)
    ws.write(1, 0, "Each heatmap shows mean DPS for the pair; other stats are averaged. Empty cells = no samples.")

    # Function to render one heatmap at a given anchor row
    def render_heatmap(anchor_row: int, row_stat: str, col_stat: str) -> int:
        # Prepare data subset
        dsub = df[[row_stat, col_stat, "dps"]].copy()
        dsub[row_stat] = to_num(dsub[row_stat])
        dsub[col_stat] = to_num(dsub[col_stat])
        dsub["dps"] = to_num(dsub["dps"])
        dsub = dsub.dropna(subset=[row_stat, col_stat, "dps"])  # keep valid rows
        if dsub.empty:
            ws.write(anchor_row, 0, f"{row_stat.title()} vs {col_stat.title()} — no data.")
            return anchor_row + 2

        col_vals = sorted(dsub[col_stat].unique().tolist())
        row_vals = sorted(dsub[row_stat].unique().tolist())

        grouped = (
            dsub.groupby([row_stat, col_stat], as_index=False)[["dps"]].mean()
        )
        grid = grouped.pivot_table(index=row_stat, columns=col_stat, values="dps", aggfunc="mean")
        grid = grid.reindex(index=row_vals, columns=col_vals)

        # Title
        ws.write(anchor_row, 0, f"{row_stat.title()} \\ {col_stat.title()} (Mean DPS)", fmt_bold)

        # Headers
        ws.write(anchor_row + 2, 0, f"{row_stat.title()} \\ {col_stat.title()}", fmt_header)
        for j, h in enumerate(col_vals, start=1):
            ws.write(anchor_row + 2, j, fmt_val(h), fmt_header)
        # Rows
        for i, rv in enumerate(row_vals, start=anchor_row + 3):
            ws.write(i, 0, fmt_val(rv), fmt_header)
            for j, cv in enumerate(col_vals, start=1):
                val = grid.at[rv, cv] if (rv in grid.index and cv in grid.columns) else np.nan
                if pd.notna(val):
                    ws.write(i, j, float(val), fmt_int0)
                else:
                    ws.write(i, j, "")

        # Conditional formatting for this block
        if col_vals and row_vals:
            last_row = (anchor_row + 2) + len(row_vals)
            last_col = len(col_vals)
            color_scale = {
                'type': '3_color_scale',
                'min_color': '#FF0000',
                'mid_color': '#FFD966',
                'max_color': '#00B050',
            }
            ws.conditional_format(anchor_row + 3, 1, last_row, last_col, color_scale)

        # Column widths for this block
        ws.set_column(0, 0, 14)
        for c in range(1, len(col_vals) + 1):
            ws.set_column(c, c, 10)

        # Return next anchor row with spacing
        return (anchor_row + 3 + len(row_vals) + 3)

    # Render all unique pairs among present stats, stacked vertically
    cur = 3
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            row_stat = present[i]
            col_stat = present[j]
            cur = render_heatmap(cur, row_stat, col_stat)

def _write_ranked_chart_sheet(df: pd.DataFrame, xl: pd.ExcelWriter):
    wb = xl.book
    ws = wb.add_worksheet("Ranked Chart")

    # Formats
    fmt_bold = wb.add_format({"bold": True})
    fmt_header = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
    fmt_pct1 = wb.add_format({"num_format": "0.0%"})

    # Selector
    dims = ["talent_str", "race", "trinket", "stat_profile", "trinket_pair", "individual talents"]
    sources = ["Summary"] + dims
    ws.write(0, 0, "Source:", fmt_bold)  # A1
    ws.data_validation(0, 1, 0, 1, {'validate': 'list', 'source': sources})
    ws.write(0, 1, "Summary")

    # Visible table headers (single table)
    ws.write(4, 0, "Label", fmt_header)
    ws.write(4, 1, "Weighted Avg", fmt_header)

    # Build data blocks: Summary and each pivot dimension (top 20 by weighted_avg)
    # Helper to create a compact label for summary builds
    def _label_from_row(row: pd.Series) -> str:
        parts: list[str] = []
        r = row.get('race', '')
        if pd.notna(r) and str(r) not in ('nan', ''):
            parts.append(str(r))
        t1 = row.get('trinket_1', '')
        t2 = row.get('trinket_2', '')
        if (pd.notna(t1) and str(t1) != 'nan') or (pd.notna(t2) and str(t2) != 'nan'):
            parts.append(f"{'' if pd.isna(t1) else t1} + {'' if pd.isna(t2) else t2}".strip())
        ts = row.get('talent_str', '')
        if isinstance(ts, str) and ts:
            s = ts.strip()
            parts.append(s if len(s) <= 60 else (s[:57] + "…"))
        label = " | ".join([p for p in parts if p and str(p) != 'nan'])
        return label if label else str(row.name)

    # Resolve helpers lazily to avoid static analysis undefined warnings
    _build_top_table_fn = globals().get("_build_top_table")
    _detect_talent_columns_fn = globals().get("_detect_talent_columns")
    _build_pivot_dimension_fn = globals().get("_build_pivot_dimension")

    # Summary list
    try:
        if callable(_build_top_table_fn):
            top_all, _, _ = _build_top_table_fn(df, top_n=None)
        else:
            top_all = pd.DataFrame(columns=["weighted_avg"])  # fallback
    except Exception:
        top_all = pd.DataFrame(columns=["weighted_avg"])  # fallback
    sum_pairs: list[tuple[str, float]] = []
    if not top_all.empty and "weighted_avg" in top_all.columns:
        for _, r in top_all.sort_values("weighted_avg", ascending=False).head(20).iterrows():
            lbl = _label_from_row(r)
            v = r.get("weighted_avg", np.nan)
            sum_pairs.append((lbl, float(v) if pd.notna(v) else np.nan))

    # Pivot lists
    targets = sorted([int(t) for t in pd.to_numeric(df.get("targets", pd.Series([])), errors="coerce").dropna().unique().tolist()])
    talent_cols = _detect_talent_columns_fn(df) if callable(_detect_talent_columns_fn) else []
    pivot_blocks: dict[str, list[tuple[str, float]]] = {}
    for d in dims:
        if callable(_build_pivot_dimension_fn):
            part = _build_pivot_dimension_fn(df, targets, d, talent_cols)
        else:
            part = pd.DataFrame({"label": [], "weighted_avg": []})
        pairs: list[tuple[str, float]] = []
        if not part.empty and "label" in part.columns and "weighted_avg" in part.columns:
            for _, r in part.sort_values("weighted_avg", ascending=False).head(20).iterrows():
                lbl = r.get("label", "")
                v = r.get("weighted_avg", np.nan)
                pairs.append((str(lbl), float(v) if pd.notna(v) else np.nan))
        pivot_blocks[d] = pairs

    # Write hidden data blocks starting at G5 (two columns per source)
    start_col = 6  # G
    max_rows = 20
    area_label_refs = []  # e.g., 'Ranked Chart'!$G$6:$G$25
    area_value_refs = []

    def _col_letter(idx0: int) -> str:
        s = ""
        n = idx0 + 1
        while n:
            n, r = divmod(n - 1, 26)
            s = chr(65 + r) + s
        return s

    # Helper to write a block of (label, value) pairs
    def write_block(col0: int, header: str, pairs: list[tuple[str, float]]):
        lbl_col = col0
        val_col = col0 + 1
        ws.write(4, lbl_col, f"{header} Labels", fmt_header)
        ws.write(4, val_col, f"{header} Values", fmt_header)
        for i in range(max_rows):
            r = 5 + i
            if i < len(pairs):
                lbl, v = pairs[i]
                ws.write(r, lbl_col, lbl)
                ws.write(r, val_col, v if pd.notna(v) else "", fmt_pct1 if pd.notna(v) else None)
            else:
                ws.write(r, lbl_col, "")
                ws.write(r, val_col, "")
        # return area refs
        L = _col_letter(lbl_col)
        V = _col_letter(val_col)
        return (f"'Ranked Chart'!${L}$6:${L}$25", f"'Ranked Chart'!${V}$6:${V}$25")

    # Summary block
    lbl_ref, val_ref = write_block(start_col, "Summary", sum_pairs)
    area_label_refs.append(lbl_ref)
    area_value_refs.append(val_ref)

    # Pivot blocks
    cur_col = start_col + 2
    for d in dims:
        lbl_ref, val_ref = write_block(cur_col, d, pivot_blocks.get(d, []))
        area_label_refs.append(lbl_ref)
        area_value_refs.append(val_ref)
        cur_col += 2

    # Single table driven by INDEX with multi-area references (no INDIRECT/OFFSET)
    # Selector list in a constant to map to area_num
    selector_const = "{" + ",".join([f'"{s}"' for s in sources]) + "}"
    # Multi-area references for labels and values
    labels_multi = "(" + ",".join(area_label_refs) + ")"
    values_multi = "(" + ",".join(area_value_refs) + ")"

    for i in range(max_rows):
        r = 5 + i
        row_count_expr = f"ROWS($A$6:A{r+1})"
        # Values: if <=0 return "" so we can compute min/max easily; blanks are skipped visually
        val_inner = f"INDEX({values_multi},{row_count_expr},1,MATCH($B$1,{selector_const},0))"
        f_val = f"=IFERROR(IF({val_inner}>0,{val_inner},\"\"),\"\")"
        # Labels: hide when the paired value cell is blank
        lbl_inner = f"INDEX({labels_multi},{row_count_expr},1,MATCH($B$1,{selector_const},0))"
        f_lbl = f"=IFERROR(IF($B{r+1}=\"\",\"\",{lbl_inner}),\"\")"
        ws.write_formula(r, 0, f_lbl)
        ws.write_formula(r, 1, f_val, fmt_pct1)

    # Helpers for dynamic lower/upper bucket and stacked baseline rendering
    # Place helpers in hidden columns C..F
    ws.write(4, 2, "Num", fmt_header)  # C5 numeric mirror of B
    for i in range(max_rows):
        rr = 5 + i
        ws.write_formula(rr, 2, f'=IF($B{rr+1}="","",$B{rr+1})')
    ws.write(0, 2, "Max", fmt_bold)  # C1
    ws.write_formula(0, 3, "=MAX($C$6:$C$25)")  # D1
    ws.write(1, 2, "Min", fmt_bold)  # C2
    ws.write_formula(1, 3, "=MIN($C$6:$C$25)")  # D2
    ws.write(2, 2, "Lower", fmt_bold)  # C3
    ws.write_formula(2, 3, "=IF($D$2>0.95,0.95,IF($D$2>0.9,0.9,IF($D$2>0.8,0.8,IF($D$2>0.6,0.6,IF($D$2>0.4,0.4,0.2)))))")  # D3 lower bucket
    ws.write(3, 2, "Upper", fmt_bold)  # C4
    ws.write_formula(3, 3, "=IF($D$1<=0.4,0.4,IF($D$1<=0.6,0.6,IF($D$1<=0.8,0.8,IF($D$1<=0.9,0.9,IF($D$1<=0.95,0.95,1)))))")  # D4 upper bucket
    ws.write(4, 4, "Relative", fmt_header)  # E5
    for i in range(max_rows):
        rr = 5 + i
        ws.write_formula(rr, 4, f'=IF($A{rr+1}="","",IF($D$4>$D$3,MAX(0,MIN($C{rr+1},$D$4)-$D$3)/($D$4-$D$3),""))')

    # Show current scale label
    ws.write_formula(2, 0, '="Scale: "&TEXT($D$3,"0%")&" – "&TEXT($D$4,"0%")')

    # Chart: single horizontal bar series on relative values to start at left boundary
    chart = wb.add_chart({'type': 'bar'})
    categories = "='Ranked Chart'!$A$6:$A$25"
    series_rel = "='Ranked Chart'!$E$6:$E$25"
    chart.add_series({'name': 'Weighted Avg', 'categories': categories, 'values': series_rel})
    chart.set_title({'name': 'Ranked Builds'})
    chart.set_x_axis({'name': '', 'num_format': ';;;', 'major_gridlines': {'visible': False}})  # hide axis numbers
    chart.set_y_axis({'name': '', 'reverse': True})
    chart.set_legend({'none': True})
    # Larger chart size
    ws.insert_chart(2, 6, chart, {'x_scale': 2.0, 'y_scale': 2.0})

    # Column widths for visible table
    ws.set_column(0, 0, 50)
    ws.set_column(1, 1, 14, fmt_pct1)
    # Keep helper columns visible but narrow so Excel always plots data reliably
    ws.set_column(2, 2, 8)   # C Num
    ws.set_column(3, 3, 8)   # D Bounds
    ws.set_column(4, 4, 10)  # E Relative
    for c in range(start_col, cur_col):
        ws.set_column(c, c, 2)  # G.. helper blocks for sources

def _write_stat_map_sheet(df: pd.DataFrame, xl: pd.ExcelWriter):
    wb = xl.book
    ws = wb.add_worksheet("Stat Map")

    fmt_bold = wb.add_format({"bold": True})
    fmt_header = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
    fmt_int0 = wb.add_format({"num_format": "#,##0"})
    fmt_num4 = wb.add_format({"num_format": "0.0000"})

    stats_all = ["crit", "haste", "vers", "mastery"]
    present = [s for s in stats_all if s in df.columns]
    if "dps" not in df.columns or len(present) < 2:
        ws.write(0, 0, "Needs dps and at least two of crit/haste/vers/mastery.", fmt_bold)
        return

    # Aggregate to unique stat profiles
    dsub = df.copy()
    for s in present + ["dps"]:
        dsub[s] = pd.to_numeric(dsub[s], errors="coerce")
    dsub = dsub.dropna(subset=present + ["dps"]).copy()
    grp = dsub.groupby(present, as_index=False).agg(dps_mean=("dps","mean"), n=("dps","count"))
    if grp.empty:
        ws.write(0, 0, "No stat profiles to plot.", fmt_bold)
        return

    # If exactly 2 stats present, just use them directly as axes
    use_pca = len(present) >= 3
    if use_pca:
        X = grp[present].to_numpy(dtype=float)
        # Standardize
        mu = np.nanmean(X, axis=0)
        sigma = np.nanstd(X, axis=0)
        sigma[sigma == 0] = 1.0
        Xs = (X - mu) / sigma
        # PCA via SVD
        try:
            U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
            comps = Vt[:2].T  # shape: (p,2)
            T = Xs.dot(comps)  # scores: (n,2)
            xs = T[:, 0]
            ys = T[:, 1] if T.shape[1] > 1 else np.zeros_like(xs)
            x_label = "PC1"
            y_label = "PC2"
        except Exception:
            # Fallback: first two raw stats
            xs = grp[present[0]].to_numpy(dtype=float)
            ys = grp[present[1]].to_numpy(dtype=float)
            x_label = present[0]
            y_label = present[1]
            use_pca = False
    else:
        xs = grp[present[0]].to_numpy(dtype=float)
        ys = grp[present[1]].to_numpy(dtype=float)
        x_label = present[0]
        y_label = present[1]

    dps = grp["dps_mean"].to_numpy(dtype=float)
    n = grp["n"].to_numpy(dtype=float)

    # Write data table for chart
    ws.write(0, 0, "Stat Map — Unified 2D View", fmt_bold)
    note = "Projected from [" + ", ".join(present) + "] using PCA; color = DPS; size = samples."
    if not use_pca:
        note = f"{present[0]} vs {present[1]} — color = DPS; size = samples."
    ws.write(1, 0, note)

    headers = [x_label, y_label, "DPS", "Samples"] + present
    for c, h in enumerate(headers):
        ws.write(3, c, h, fmt_header)
    for i in range(len(grp)):
        r = 4 + i
        # round display to 4 decimals
        ws.write(r, 0, float(np.round(xs[i], 4)), fmt_num4)
        ws.write(r, 1, float(np.round(ys[i], 4)), fmt_num4)
        ws.write(r, 2, float(dps[i]), fmt_int0)
        ws.write(r, 3, int(n[i]) if np.isfinite(n[i]) else "")
        for j, s in enumerate(present):
            ws.write(r, 4 + j, float(grp.iloc[i][s]))

    # Compute colors per point: red -> yellow -> green by DPS
    d_min = float(np.nanmin(dps)) if len(dps) else 0.0
    d_max = float(np.nanmax(dps)) if len(dps) else 1.0
    if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max <= d_min:
        d_min, d_max = 0.0, 1.0

    def lerp(a, b, t):
        return a + (b - a) * t
    def color_for(val):
        t = 0.0 if d_max == d_min else max(0.0, min(1.0, (val - d_min) / (d_max - d_min)))
        # RGB for red, yellow, green
        r1, g1, b1 = 255, 0, 0        # red
        r2, g2, b2 = 255, 217, 102    # yellow
        r3, g3, b3 = 0, 176, 80       # green
        if t <= 0.5:
            tt = t * 2
            r = int(round(lerp(r1, r2, tt)))
            g = int(round(lerp(g1, g2, tt)))
            b = int(round(lerp(b1, b2, tt)))
        else:
            tt = (t - 0.5) * 2
            r = int(round(lerp(r2, r3, tt)))
            g = int(round(lerp(g2, g3, tt)))
            b = int(round(lerp(b2, b3, tt)))
        return f"#{r:02X}{g:02X}{b:02X}"

    # Marker sizes by log(samples)
    n_safe = np.where(n > 0, n, 1)
    n_scaled = 4 + 4 * (np.log10(n_safe) - np.nanmin(np.log10(n_safe))) / max(1e-6, (np.nanmax(np.log10(n_safe)) - np.nanmin(np.log10(n_safe))))
    n_scaled = np.clip(n_scaled, 4, 10)

    # Build scatter chart (markers only to show per-point colors)
    chart = wb.add_chart({'type': 'scatter', 'subtype': 'markers'})
    # Add each point as its own series to ensure per-point colors in all Excel versions
    for i in range(len(grp)):
        r = 4 + i
        chart.add_series({
            'name': '',
            'categories': ['Stat Map', r, 0, r, 0],  # x
            'values':     ['Stat Map', r, 1, r, 1],  # y
            'marker': {
                'type': 'circle',
                'size': int(n_scaled[i]) if np.isfinite(n_scaled[i]) else 6,
                'fill': {'color': color_for(dps[i])},
                'border': {'none': True},
            },
            'line': {'none': True},
        })
    chart.set_title({'name': 'Stat Map'})
    chart.set_x_axis({'name': x_label, 'num_format': '0.0000'})
    chart.set_y_axis({'name': y_label, 'num_format': '0.0000'})
    chart.set_legend({'none': True})
    # Place chart to the right of the data table
    ws.insert_chart(3, len(headers) + 2, chart, {'x_scale': 1.6, 'y_scale': 2.0})

    # Removed the in-sheet color legend for a cleaner layout

def _write_stat_radar_sheet(df: pd.DataFrame, xl: pd.ExcelWriter, top_n: int = 5):
    wb = xl.book
    ws = wb.add_worksheet("Stat Radar")

    fmt_bold = wb.add_format({"bold": True})
    fmt_header = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
    fmt_int0 = wb.add_format({"num_format": "#,##0"})
    fmt_num0 = wb.add_format({"num_format": "0"})
    fmt_pct0 = wb.add_format({"num_format": "0%"})

    stats_all = ["crit", "haste", "vers", "mastery"]
    present = [s for s in stats_all if s in df.columns]
    if len(present) < 2 or "dps" not in df.columns:
        ws.write(0, 0, "Needs dps and at least two stats among crit/haste/vers/mastery.", fmt_bold)
        return

    # Prepare aggregated profiles
    dsub = df.copy()
    for s in present + ["dps"]:
        dsub[s] = pd.to_numeric(dsub[s], errors="coerce")
    dsub = dsub.dropna(subset=present + ["dps"]).copy()
    if dsub.empty:
        ws.write(0, 0, "No stat profiles to chart.", fmt_bold)
        return
    grp = dsub.groupby(present, as_index=False).agg(dps_mean=("dps","mean"), n=("dps","count"))
    if grp.empty:
        ws.write(0, 0, "No stat profiles to chart.", fmt_bold)
        return

    # Normalize each axis 0..1 across the dataset for fairness
    mins = {s: float(grp[s].min()) for s in present}
    maxs = {s: float(grp[s].max()) for s in present}
    def norm(v, s):
        lo, hi = mins[s], maxs[s]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.0
        return float((v - lo) / (hi - lo))
    for s in present:
        grp[f"{s}_norm"] = grp[s].apply(lambda x, ss=s: norm(x, ss))

    # Pick top N profiles by mean DPS
    grp = grp.sort_values("dps_mean", ascending=False).head(top_n).reset_index(drop=True)

    # Write header and note
    ws.write(0, 0, "Stat Radar — Top Profiles (Normalized per-axis)", fmt_bold)
    ws.write(1, 0, "Each axis is scaled 0–1 using min/max across all profiles; series = top profiles by mean DPS.")

    # Data table layout for radar: row 4 headers, rows 5.. series values
    header_row = 3
    ws.write(header_row, 0, "Series", fmt_header)
    for j, s in enumerate(present, start=1):
        ws.write(header_row, j, s.title(), fmt_header)
    # Write series rows
    for i in range(len(grp)):
        r = header_row + 1 + i
        # Series label includes raw stats and DPS
        raw_parts = "/".join([str(int(grp.loc[i, s])) if pd.notna(grp.loc[i, s]) else "" for s in present])
        label = f"{raw_parts} (DPS {int(grp.loc[i, 'dps_mean'])})"
        ws.write(r, 0, label)
        for j, s in enumerate(present, start=1):
            ws.write(r, j, float(grp.loc[i, f"{s}_norm"]))

    # Create radar chart (filled for readability)
    subtype = 'filled'
    try:
        chart = wb.add_chart({'type': 'radar', 'subtype': subtype})
    except Exception:
        chart = wb.add_chart({'type': 'radar'})
    categories = ['Stat Radar', header_row, 1, header_row, len(present)]

    colors = [
        '#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5',
        '#70AD47', '#264478', '#9E480E', '#636363', '#997300'
    ]
    for i in range(len(grp)):
        r = header_row + 1 + i
        series = {
            'name': ['Stat Radar', r, 0],
            'categories': categories,
            'values': ['Stat Radar', r, 1, r, len(present)],
            'line': {'color': colors[i % len(colors)]},
        }
        if subtype == 'filled':
            series['fill'] = {'color': colors[i % len(colors)], 'transparency': 70}
            series['marker'] = {'type': 'circle', 'size': 4, 'border': {'none': True}, 'fill': {'color': colors[i % len(colors)]}}
        chart.add_series(series)

    chart.set_title({'name': 'Stat Radar'})
    chart.set_y_axis({'min': 0, 'max': 1, 'major_unit': 0.2, 'num_format': '0%'})
    chart.set_legend({'position': 'right'})

    # Insert chart to the right of the table
    ws.insert_chart(3, len(present) + 3, chart, {'x_scale': 1.6, 'y_scale': 1.8})

    # Formatting
    ws.set_column(0, 0, 36)
    for j in range(1, len(present) + 1):
        ws.set_column(j, j, 10, fmt_pct0)


def write_excel(df: pd.DataFrame, out_xlsx: Path, source_csv_name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter", engine_kwargs={"options": {"nan_inf_to_errors": True}}) as xl:
            # Data sheet
            df.to_excel(xl, sheet_name="Data", index=False)
            # Enable AutoFilter on Data sheet
            try:
                ws_data = xl.sheets.get("Data")
                if ws_data is not None:
                    last_row_data = df.shape[0]
                    last_col_data = max(0, df.shape[1] - 1)
                    ws_data.autofilter(0, 0, last_row_data, last_col_data)
            except Exception:
                pass
            # Custom formatted Summary sheet
            _write_summary_sheet(df, xl, source_csv_name)
            # Pivots sheet with dropdown-controlled dimension
            _write_pivots_sheet(df, xl)
            # Talents sheet: single talent analytics and synergies
            _write_talents_sheet(df, xl)
            # Ranked Chart sheet: horizontal bar chart of best builds with optional Pivots source
            _write_ranked_chart_sheet(df, xl)
            # Stat heatmap (Haste vs Mastery)
            _write_stat_heatmap_sheet(df, xl)
            # Unified stat map (PCA scatter colored by DPS)
            _write_stat_map_sheet(df, xl)
            # Independent stat radar (spider chart)
            _write_stat_radar_sheet(df, xl)
    except PermissionError:
        # Fallback to timestamped filename if the file is currently open/locked
        stem = out_xlsx.stem
        ts_name = OUT_DIR / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        with pd.ExcelWriter(ts_name, engine="xlsxwriter", engine_kwargs={"options": {"nan_inf_to_errors": True}}) as xl:
            df.to_excel(xl, sheet_name="Data", index=False)
            try:
                ws_data = xl.sheets.get("Data")
                if ws_data is not None:
                    last_row_data = df.shape[0]
                    last_col_data = max(0, df.shape[1] - 1)
                    ws_data.autofilter(0, 0, last_row_data, last_col_data)
            except Exception:
                pass
            _write_summary_sheet(df, xl, source_csv_name)
            _write_pivots_sheet(df, xl)
            _write_talents_sheet(df, xl)
            _write_ranked_chart_sheet(df, xl)
            _write_stat_heatmap_sheet(df, xl)
            _write_stat_map_sheet(df, xl)
            _write_stat_radar_sheet(df, xl)
        print(f"Target locked. Wrote fallback Excel: {ts_name}")
        return


def _find_root_csvs(root: Path) -> list[Path]:
    # Non-recursive: only CSVs in the root folder next to this script
    files = list(root.glob("*.csv"))
    # Also include .CSV (case-insensitive safety on some platforms)
    files += [p for p in root.glob("*.CSV") if p not in files]
    return sorted(files)


def main(argv: list[str] | None = None) -> int:
    csv_files = _find_root_csvs(ROOT)
    if not csv_files:
        print(f"No CSV files found in {ROOT}")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    success = 0
    for csv_path in csv_files:
        try:
            df = load_data(csv_path)
        except FileNotFoundError:
            print(f"CSV not found (skipping): {csv_path}")
            continue
        except Exception as e:
            print(f"Failed to load {csv_path.name}: {e}")
            continue
        out_xlsx = OUT_DIR / f"{csv_path.stem}.xlsx"
        try:
            write_excel(df, out_xlsx, csv_path.name)
            print(f"Wrote Excel report: {out_xlsx}")
            success += 1
        except Exception as e:
            print(f"Failed to write Excel for {csv_path.name}: {e}")
    if success == 0:
        return 1
    print(f"Done. Generated {success} Excel report(s) in {OUT_DIR}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
