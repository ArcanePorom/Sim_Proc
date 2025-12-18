#!/usr/bin/env python3
from __future__ import annotations
import re
import sys
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "sim_proc_combined_results.csv"
OUT_DIR = ROOT / "reports"


def detect_talent_columns(df: pd.DataFrame) -> list[str]:
    known = {
        "file","builds","build_id","race","class","spec","talent_str","trinket_1","trinket_2",
        "targets","crit","haste","vers","mastery","dps","priority_dps","range_pct_dps",
        "min_dps","max_dps","StdDev"
    }
    cols: list[str] = []
    for c in df.columns:
        if c in known:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if not len(s):
            continue
        uniq = set(s.unique().tolist())
        if uniq.issubset({0,1,2}):
            cols.append(c)
    return cols


def strip_chunk_suffix(name: str) -> str:
    try:
        return re.sub(r'(?:_\d+)+$', '', name)
    except Exception:
        return name


def extract_actor(build_id: str | float | int | None, regex: str | None) -> str:
    s = str(build_id) if build_id is not None else ""
    if regex:
        try:
            m = re.search(regex, s)
            if m:
                return (m.group(1) if m.groups() else m.group(0)).strip()
        except re.error:
            pass
    # Heuristics around "Explore" tokens in build names
    m = re.search(r"(?i)explore[:\- ]*([A-Za-z0-9_\- ]{1,40})", s)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?i)(explore[^_\s]*)", s)
    if m:
        return m.group(1).strip()
    # Fallback: keep base id without trailing numeric chunks (preserve names like 'missiles_more')
    base = strip_chunk_suffix(s)
    return base or "default"


def make_build_key(df: pd.DataFrame) -> list[str]:
    # Use the primary identifiers that define a build config apart from actor
    desired = ["talent_str", "race", "trinket_1", "trinket_2", "crit", "haste", "vers", "mastery", "targets"]
    keys = [c for c in desired if c in df.columns]
    return keys


def compute_build_deltas(df: pd.DataFrame, actor_col: str, baseline_actor: str | None) -> pd.DataFrame:
    keys = make_build_key(df)
    metrics = [c for c in ["dps", "priority_dps"] if c in df.columns]
    if not metrics:
        return pd.DataFrame()
    agg = (
        df.groupby(keys + [actor_col], as_index=False)[metrics]
          .mean()
    )
    # Pivot per metric, then merge
    outs: list[pd.DataFrame] = []
    for m in metrics:
        wide = agg.pivot_table(index=keys, columns=actor_col, values=m, aggfunc="mean").reset_index()
        # Add deltas vs baseline if available
        if baseline_actor and baseline_actor in wide.columns:
            base = wide[baseline_actor]
            for c in wide.columns:
                if c in keys or c == baseline_actor:
                    continue
                wide[f"{m}_delta_vs_{baseline_actor}:{c}"] = wide[c] - base
                wide[f"{m}_pct_vs_{baseline_actor}:{c}"] = np.where(np.isfinite(base) & (base != 0), (wide[c] / base) - 1.0, np.nan)
        outs.append(wide)
    # Merge on keys
    out = outs[0]
    for w in outs[1:]:
        out = out.merge(w, on=keys, how="outer")
    return out


def compute_talent_effects_by_actor(df: pd.DataFrame, actor_col: str, targets: int | None) -> pd.DataFrame:
    # For each actor/build, compute mean DPS when the talent is taken, and the sample size (times picked).
    if targets is not None and "targets" in df.columns:
        df = df.loc[pd.to_numeric(df["targets"], errors="coerce") == targets]
    tcols = detect_talent_columns(df)
    if not tcols:
        return pd.DataFrame()
    actors = sorted([str(a) for a in df[actor_col].dropna().unique().tolist()])
    rows = []
    for tal in tcols:
        row = {"talent": tal}
        for actor in actors:
            sub = df.loc[df[actor_col] == actor]
            s = pd.to_numeric(sub[tal], errors="coerce").fillna(0)
            taken = sub.loc[s > 0]
            mean_taken = float(taken["dps"].mean()) if "dps" in taken.columns and not taken.empty else np.nan
            samples_taken = int(taken.shape[0])
            row[f"{actor}_avg_dps"] = mean_taken
            row[f"{actor}_samples"] = samples_taken
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Add pairwise delta and delta% columns for all actor combinations, relative to the alphabetically-first actor in the pair
    pair_delta_cols = []
    for i in range(len(actors)):
        for j in range(i + 1, len(actors)):
            ai, aj = actors[i], actors[j]
            col_ai = f"{ai}_avg_dps"
            col_aj = f"{aj}_avg_dps"
            delta_col = f"delta_{ai}_minus_{aj}"
            pct_col = f"delta_pct_{ai}_vs_{aj}"  # percent relative to {ai}
            out[delta_col] = out[col_ai] - out[col_aj]
            with np.errstate(divide='ignore', invalid='ignore'):
                out[pct_col] = np.where(np.isfinite(out[col_ai]) & (out[col_ai] != 0), (out[col_ai] - out[col_aj]) / out[col_ai], np.nan)
            pair_delta_cols.append(delta_col)
    # Sort rows by max absolute pairwise delta for readability
    if pair_delta_cols:
        sort_key = out[pair_delta_cols].abs().max(axis=1)
        out = out.assign(_sort=sort_key).sort_values("_sort", ascending=False, na_position="last").drop(columns=["_sort"]).reset_index(drop=True)
    return out


def compute_trinket_effects_by_actor(df: pd.DataFrame, actor_col: str, targets: int | None) -> pd.DataFrame:
    if targets is not None and "targets" in df.columns:
        df = df.loc[pd.to_numeric(df["targets"], errors="coerce") == targets]
    frames: list[pd.DataFrame] = []
    if "trinket_1" in df.columns:
        t1 = df[[actor_col, "trinket_1", "dps"]].rename(columns={"trinket_1": "trinket"})
        frames.append(t1)
    if "trinket_2" in df.columns:
        t2 = df[[actor_col, "trinket_2", "dps"]].rename(columns={"trinket_2": "trinket"})
        frames.append(t2)
    if not frames:
        return pd.DataFrame()
    comb = pd.concat(frames, ignore_index=True)
    comb = comb[pd.notna(comb["trinket"]) & (comb["trinket"].astype(str).str.len() > 0)]
    agg = comb.groupby(["trinket", actor_col], as_index=False)[["dps"]].mean()
    wide = agg.pivot_table(index=["trinket"], columns=actor_col, values="dps", aggfunc="mean").reset_index()
    return wide.sort_values("trinket").reset_index(drop=True)


def compute_stat_profiles_by_actor(df: pd.DataFrame, actor_col: str, targets: int | None) -> pd.DataFrame:
    if targets is not None and "targets" in df.columns:
        df = df.loc[pd.to_numeric(df["targets"], errors="coerce") == targets]
    for c in ["crit","haste","vers","mastery","dps"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if not set(["crit","haste","vers","mastery","dps"]).issubset(set(df.columns)):
        return pd.DataFrame()
    df = df.dropna(subset=["crit","haste","vers","mastery","dps"]).copy()
    df["stat_label"] = df.apply(lambda r: f"{int(r['crit'])}/{int(r['haste'])}/{int(r['vers'])}/{int(r['mastery'])}", axis=1)
    agg = df.groupby(["stat_label", actor_col], as_index=False)[["dps"]].mean()
    wide = agg.pivot_table(index=["stat_label"], columns=actor_col, values="dps", aggfunc="mean").reset_index()
    return wide


def write_outputs(out_dir: Path, build_deltas: pd.DataFrame, talent_effects: pd.DataFrame, trinket_effects: pd.DataFrame, stat_profiles: pd.DataFrame, out_xlsx: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    def safe_to_csv(df: pd.DataFrame, path: Path):
        try:
            df.to_csv(path, index=False)
        except PermissionError:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            alt = path.with_stem(f"{path.stem}_{ts}")
            df.to_csv(alt, index=False)
    if not build_deltas.empty:
        safe_to_csv(build_deltas, out_dir / "explore_build_deltas.csv")
    if not talent_effects.empty:
        safe_to_csv(talent_effects, out_dir / "explore_talent_effects.csv")
    if not trinket_effects.empty:
        safe_to_csv(trinket_effects, out_dir / "explore_trinket_effects.csv")
    if not stat_profiles.empty:
        safe_to_csv(stat_profiles, out_dir / "explore_stat_profiles.csv")
    if out_xlsx:
        xlsx_path = out_dir / "explore_actors_analysis.xlsx"

        def write_excel_with_formats(xl: pd.ExcelWriter):
            wb = xl.book
            # Build deltas via to_excel
            if not build_deltas.empty:
                build_deltas.to_excel(xl, sheet_name="build_deltas", index=False)
            # Talent effects: write via XlsxWriter for precise formats
            if not talent_effects.empty:
                tf = talent_effects.copy()
                pct_cols = [c for c in tf.columns if isinstance(c, str) and c.startswith("delta_pct_")]
                for c in pct_cols:
                    tf[c] = pd.to_numeric(tf[c], errors="coerce")
                ws = wb.add_worksheet("talent_effects")
                fmt_header = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
                fmt_pct = wb.add_format({"num_format": "0.0%"})
                fmt_int0 = wb.add_format({"num_format": "#,##0"})
                fmt_float0 = wb.add_format({"num_format": "#,##0"})
                # Headers
                for j, name in enumerate(tf.columns):
                    ws.write(0, j, str(name), fmt_header)
                # Rows
                for i in range(len(tf)):
                    for j, name in enumerate(tf.columns):
                        val = tf.iloc[i, j]
                        if pd.isna(val):
                            ws.write(i+1, j, "")
                        else:
                            if isinstance(name, str) and name.startswith("delta_pct_"):
                                try:
                                    ws.write_number(i+1, j, float(val), fmt_pct)
                                except Exception:
                                    ws.write(i+1, j, val)
                            elif isinstance(val, (int, np.integer)):
                                ws.write_number(i+1, j, int(val), fmt_int0)
                            elif isinstance(val, (float, np.floating)):
                                ws.write_number(i+1, j, float(val), fmt_float0)
                            else:
                                ws.write(i+1, j, str(val))
                # Column widths
                for j, name in enumerate(tf.columns):
                    if isinstance(name, str) and name.startswith("delta_pct_"):
                        ws.set_column(j, j, 12)
                    elif isinstance(name, str) and name.endswith("_avg_dps"):
                        ws.set_column(j, j, 18)
                    else:
                        ws.set_column(j, j, 14)
                # AutoFilter
                ws.autofilter(0, 0, len(tf), len(tf.columns) - 1)
            # Other sheets via to_excel
            if not trinket_effects.empty:
                trinket_effects.to_excel(xl, sheet_name="trinket_effects", index=False)
            if not stat_profiles.empty:
                stat_profiles.to_excel(xl, sheet_name="stat_profiles", index=False)

        try:
            with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xl:
                write_excel_with_formats(xl)
        except PermissionError:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            alt = xlsx_path.with_stem(f"{xlsx_path.stem}_{ts}")
            with pd.ExcelWriter(alt, engine="xlsxwriter") as xl:
                write_excel_with_formats(xl)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Analyze Explore-actor effects on builds, talents, trinkets, and stats.")
    ap.add_argument("--csv", default=str(INPUT_CSV), help="Path to the combined results CSV (default: sim_proc_combined_results.csv)")
    ap.add_argument("--actor-from", choices=["builds","build_id","file"], default="builds", help="Which column to extract actor labels from (default: builds)")
    ap.add_argument("--actor-regex", default=None, help="Regex to extract actor label; group(1) is used when available")
    ap.add_argument("--baseline-actor", default=None, help="Optional baseline actor name for delta columns in build deltas")
    ap.add_argument("--targets", type=int, default=None, help="Filter to a specific targets count (e.g., 1)")
    ap.add_argument("--xlsx", action="store_true", help="Also write a multi-sheet Excel file of results")
    args = ap.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1
    df = pd.read_csv(csv_path)
    if args.targets is not None and "targets" in df.columns:
        df["targets"] = pd.to_numeric(df["targets"], errors="coerce")
        df = df.loc[df["targets"] == args.targets]
    # Actor extraction
    base_col = args.actor_from
    if base_col not in df.columns:
        print(f"Column '{base_col}' not found in CSV; cannot extract actors.")
        return 1
    if base_col == "builds":
        # Use builds verbatim for actor labels
        df["actor"] = df[base_col].astype(str)
    else:
        df["actor"] = df[base_col].apply(lambda s: extract_actor(s, args.actor_regex))
    # Small report
    dist = df["actor"].value_counts().to_dict()
    print("Actor distribution:", json.dumps(dist, indent=2))

    build_deltas = compute_build_deltas(df, actor_col="actor", baseline_actor=args.baseline_actor)
    talent_effects = compute_talent_effects_by_actor(df, actor_col="actor", targets=args.targets)
    trinket_effects = compute_trinket_effects_by_actor(df, actor_col="actor", targets=args.targets)
    stat_profiles = compute_stat_profiles_by_actor(df, actor_col="actor", targets=args.targets)

    write_outputs(OUT_DIR, build_deltas, talent_effects, trinket_effects, stat_profiles, out_xlsx=args.xlsx)
    print(f"Wrote analysis outputs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
