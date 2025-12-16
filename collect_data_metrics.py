def parse_section(div, source_file, gear_stats, target_count, min_dps=None, median_dps=None, max_dps=None, trinket_1=None, trinket_2=None, talent_names=None):
    # Extract StdDev from DPS statistics table in this player section
    stddev = None
    dps_stats_table = None
    for tbl in div.find_all("table"):
        header = tbl.find("th")
        if header and "Damage Per Second" in header.get_text():
            dps_stats_table = tbl
            break
    if dps_stats_table:
        for tr in dps_stats_table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) == 2:
                label = tds[0].get_text(" ", strip=True)
                value = tds[1].get_text(" ", strip=True)
                if "Standard Deviation" in label:
                    stddev = to_float(value)
                    break
    # Extract range_pct_dps from the 'DPS Range' cell in the table with DPS headers
    range_pct_dps = None
    for tbl in div.find_all("table"):
        headers = [th.get_text(" ", strip=True) for th in tbl.find_all("th")]
        if any("DPS Range" in h for h in headers):
            # Find the index of the DPS Range column
            try:
                idx = headers.index("DPS Range")
            except ValueError:
                continue
            # Find the first data row
            for tr in tbl.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) > idx:
                    cell = tds[idx].get_text(" ", strip=True)
                    # Extract the percentage after the '/' and before the '%'
                    m = re.search(r"/\s*([0-9.]+)%", cell)
                    if m:
                        range_pct_dps = to_float(m.group(1))
                        break
            if range_pct_dps is not None:
                break

    # Extract min_dps, max_dps, and dps (mean) from the 'Damage Per Second' table in the 'DPS' dropdown
    min_dps = max_dps = dps = None
    dps_table = None
    for a in div.find_all("a", class_="toggle-details"):
        if "DPS" in a.text:
            tr_details = a.find_parent("tr").find_next_sibling("tr", class_="details")
            if tr_details:
                details_table = tr_details.find("table", class_="details")
                if details_table:
                    dps_table = details_table
            break
    if dps_table:
        for tr in dps_table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) == 2:
                label = tds[0].get_text(" ", strip=True)
                value = tds[1].get_text(" ", strip=True)
                llabel = label.strip().lower()
                if llabel == "minimum":
                    min_dps = to_float(value)
                elif llabel == "maximum":
                    max_dps = to_float(value)
                elif "mean" in llabel:
                    dps = to_float(value)

    # Extract priority_dps from the table with header containing 'Priority Target Damage Per Second'
    priority_dps = None
    for tbl in div.find_all("table"):
        header = tbl.find("th")
        if header and "Priority Target Damage Per Second" in header.get_text():
            for tr in tbl.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) == 2:
                    label = tds[0].get_text(" ", strip=True)
                    value = tds[1].get_text(" ", strip=True)
                    if label.strip().lower() == "mean":
                        priority_dps = to_float(value)
                        break
            break
    # Robust trinket extraction from text block after gear table inside this player section
    trinket_1 = trinket_2 = None
    found_trinkets = False
    for p in div.find_all("p"):
        html = p.decode_contents()
        m1 = re.search(r"trinket1=([a-zA-Z0-9_\-']+)", html)
        m2 = re.search(r"trinket2=([a-zA-Z0-9_\-']+)", html)
        if m1:
            trinket_1 = m1.group(1)
            found_trinkets = True
        if m2:
            trinket_2 = m2.group(1)
            found_trinkets = True
        if found_trinkets:
            break
    h2 = div.select_one("h2.toggle")
    txt = h2.get_text(" ", strip=True) if h2 else ""
    build_re = re.compile(r"(.+?)\s*:\s*([\d,]+)\s*dps", re.IGNORECASE)
    m = build_re.search(txt)
    if not m:
        print(f"[DEBUG] Regex did not match for: '{txt}'")
        return None
    build_id = m.group(1).strip()
    # Truncate trailing _digits for 'builds' field
    builds = re.sub(r'(?:_\d+)+$', '', build_id)
    if any(x in build_id for x in ("Dummy_Enemy", "Fluffy_Pillow")):
        return None
    # (dps and priority_dps are now extracted above from the correct tables)
    race = cls = spec = None
    params_ul = div.find("ul", class_="params")
    if params_ul:
        for li in params_ul.find_all("li"):
            txt = li.get_text(" ", strip=True)
            if txt.startswith("Race:"):
                race = txt.replace("Race:", "").strip()
            elif txt.startswith("Class:"):
                cls = txt.replace("Class:", "").strip()
            elif txt.startswith("Spec:"):
                spec = txt.replace("Spec:", "").strip()
    talent_str = None
    for p in div.find_all("p"):
        html = p.decode_contents()
        if "talents=" in html:
            t = re.search(r"talents=([^<]+)", html)
            if t:
                talent_str = t.group(1).strip()
            break
    # Ensure range_pct_dps is always defined
    if 'range_pct_dps' not in locals():
        range_pct_dps = None
    row = {
        "file":         source_file,
        "build_id":     build_id,
        "builds":       builds,
        "dps":          dps,
        "priority_dps": priority_dps,
        "race":         race,
        "class":        cls,
        "spec":         spec,
        "range_pct_dps": range_pct_dps,
        "talent_str":   talent_str,
        "targets":      target_count,
        "crit":         gear_stats.get("crit"),
        "haste":        gear_stats.get("haste"),
        "vers":         gear_stats.get("vers"),
        "mastery":      gear_stats.get("mastery"),
        "min_dps":      min_dps,
        "max_dps":      max_dps,
        "StdDev":       stddev,
        "trinket_1":    trinket_1,
        "trinket_2":    trinket_2
    }
    # Fill talent columns with number of points (0 if not found)
    if talent_names:
        talent_points = {}
        # Scrape raw talent names (without ids) and their ranks from HTML
        for talents_div in div.find_all("div", class_="player-section talents"):
            for table in talents_div.find_all("table", class_="sc"):
                for li in table.find_all("li"):
                    a = li.find("a")
                    if a:
                        name = a.get_text(strip=True)
                        m = re.search(r'\[(\d+)\]', li.get_text())
                        points = int(m.group(1)) if m else 1
                        talent_points[name] = points
        # Fallback if structure differs
        if not talent_points:
            for table in div.find_all("table", class_="sc"):
                for li in table.find_all("li"):
                    a = li.find("a")
                    if a:
                        name = a.get_text(strip=True)
                        m = re.search(r'\[(\d+)\]', li.get_text())
                        points = int(m.group(1)) if m else 1
                        talent_points[name] = points
        # Map headers 'Name (Id)' back to scraped plain 'Name'
        for tname in talent_names:
            plain = tname.split(' (')[0]
            row[tname] = talent_points.get(plain, 0)
    return row
#!/usr/bin/env python3
import re
import csv
import sys
from pathlib import Path
import configparser
from bs4 import BeautifulSoup
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from generate_talents import TalentJSON, tokenize

BASE_DIR   = Path(__file__).parent
OUTPUT_CSV = BASE_DIR / "sim_proc_combined_results.csv"
HTML_DIR   = BASE_DIR / "simc_data"
TALENTS_JSON = BASE_DIR / "talents.json"
_cfg = configparser.ConfigParser()
try:
    _cfg.read(BASE_DIR / "config.ini")
    _DEL_HTML = _cfg.getboolean("cleanup", "delete_html", fallback=True)
    _DEL_JSON = _cfg.getboolean("cleanup", "delete_json", fallback=True)
except Exception:
    _DEL_HTML = True
    _DEL_JSON = True
def load_all_spec_talent_headers():
    """Return ordered talent headers for ALL class/spec combinations.

    Format: 'Talent Name (EntryId)'. Ordering within each spec follows that spec's
    fullNodeOrder; final header list preserves first-seen order across specs while
    ensuring uniqueness by entry id. Hero subtree selection entries included.
    """
    try:
        with open(TALENTS_JSON, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    headers = []
    seen_entry_ids = set()
    for spec_block in data:
        if not isinstance(spec_block, dict):
            continue
        node_sources = ["classNodes", "specNodes", "heroNodes", "subTreeNodes"]
        nodes_by_id = {}
        for src in node_sources:
            for node in spec_block.get(src, []):
                if "id" in node:
                    nodes_by_id[node["id"]] = node
        node_order = spec_block.get("fullNodeOrder", [])
        for node_id in node_order:
            node = nodes_by_id.get(node_id)
            if not node:
                continue
            for entry in node.get("entries", []):
                entry_id = entry.get("id")
                name = entry.get("name")
                max_ranks = entry.get("maxRanks", 0)
                if entry_id is None or max_ranks <= 0:
                    continue
                if entry_id in seen_entry_ids:
                    continue
                seen_entry_ids.add(entry_id)
                headers.append(f"{name} ({entry_id})")
    return headers

# --- Talent string decoding helpers ---
_TALENT_JSON = None
def _get_talent_json():
    global _TALENT_JSON
    if _TALENT_JSON is None:
        try:
            _TALENT_JSON = TalentJSON.from_file(str(TALENTS_JSON))
        except Exception:
            _TALENT_JSON = None
    return _TALENT_JSON

def _get_spec_obj(class_name: str | None, spec_name: str | None):
    tj = _get_talent_json()
    if tj is None or not class_name or not spec_name:
        return None
    # Access via tokenized attributes from generate_talents
    try:
        class_attr = tokenize(class_name)
        spec_attr = tokenize(spec_name)
        class_helper = getattr(tj, class_attr, None)
        return getattr(class_helper, spec_attr, None) if class_helper else None
    except Exception:
        return None

def decode_talent_string_to_header_ranks(talent_str: str, headers: list[str], class_name: str | None, spec_name: str | None) -> dict[str, int]:
    spec = _get_spec_obj(class_name, spec_name)
    if spec is None or not talent_str:
        return {h:0 for h in headers}
    try:
        build = spec.parse_talent_string(talent_str)
    except Exception:
        return {h:0 for h in headers}
    id_to_rank = {choice.id: rank for choice, rank in build.items()}
    ranks: dict[str, int] = {}
    for h in headers:
        m = re.search(r'\((\d+)\)$', h)
        if m:
            eid = int(m.group(1))
            ranks[h] = id_to_rank.get(eid, 0)
        else:
            ranks[h] = 0
    return ranks

def to_float(s):
    if not s:
        return None
    s = s.replace(",", "").strip().rstrip("%")
    try:
        return float(re.match(r"^([0-9.]+)", s).group(1))
    except Exception:
        return None

def extract_gear_stats(stats_table):
    """
    Given a BeautifulSoup table element for the 'Stats' table,
    returns a dict with gear amounts for crit, haste, vers, mastery.
    """
    stats = {"crit": None, "haste": None, "vers": None, "mastery": None}
    label_map = {
        "Crit": "crit",
        "Haste": "haste",
        "Versatility": "vers",
        "Mastery": "mastery"
    }
    rows = stats_table.find_all("tr")
    # Skip the header row (first row)
    for row in rows[1:]:
        ths = row.find_all("th")
        tds = row.find_all("td")
        if not ths or not tds:
            continue
        label = ths[0].get_text(strip=True)
        if label in label_map:
            # Use the last <td> for Gear Amount
            value = tds[-1].get_text(strip=True).replace(',', '')
            try:
                stats[label_map[label]] = float(value) if value else None
            except Exception:
                stats[label_map[label]] = None
    return stats

def parse_file(fp, talent_names=None):
    try:
        soup = BeautifulSoup(fp.read_text(encoding="utf-8"), "html.parser")
    except Exception as e:
        print(f"⚠️ Error reading {fp}: {e}")
        return []
    target_count = sum(
        1 for p in soup.find_all("p")
        if "tank_dummy=" in p.decode_contents().lower() or "enemy=" in p.decode_contents().lower()
    )
    # Parse min/max/median DPS from the statistics table in the whole HTML
    min_dps = max_dps = median_dps = None
    for table in soup.find_all("table"):
        header = table.find("th")
        if header and "Damage Per Second" in header.get_text():
            for tr in table.find_all("tr"):
                th = tr.find("th")
                td = tr.find("td")
                if th and td:
                    label = th.get_text(" ", strip=True)
                    value = td.get_text(" ", strip=True)
                    if "Minimum" in label:
                        min_dps = to_float(value)
                    elif "Mean" in label:
                        median_dps = to_float(value)
                    elif "Maximum" in label:
                        max_dps = to_float(value)
            break

    # Parse trinket names from the gear table in the whole HTML
    trinket_1 = trinket_2 = None
    for table in soup.find_all("table"):
        first_row = table.find("tr")
        if first_row:
            headers = [th.get_text(" ", strip=True).lower() for th in first_row.find_all("th")]
            if "slot" in headers:
                slot_idx = headers.index("slot")
                item_idx = headers.index("item") if "item" in headers else slot_idx + 1
                for tr in table.find_all("tr")[1:]:
                    tds = tr.find_all("td")
                    if len(tds) > max(slot_idx, item_idx):
                        slot = tds[slot_idx].get_text(" ", strip=True).lower()
                        item_name = tds[item_idx].get_text(" ", strip=True)
                        if slot == "trinket1":
                            trinket_1 = item_name
                        elif slot == "trinket2":
                            trinket_2 = item_name
                break

    results = []
    for div in soup.select("div.player.section"):
        # Find the "Stats" table for this actor section
        stats_table = None
        for table in div.find_all("table"):
            # Look for a table with a header row containing "Gear Amount"
            header_row = table.find("tr")
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all("th")]
                if "Gear Amount" in headers:
                    stats_table = table
                    break
        if stats_table:
            gear_stats = extract_gear_stats(stats_table)
        else:
            gear_stats = {"crit": None, "haste": None, "vers": None, "mastery": None}
        parsed = parse_section(div, fp.stem, gear_stats, target_count, min_dps, median_dps, max_dps, trinket_1, trinket_2, talent_names)
        if parsed:
            results.append(parsed)
    return results

def strip_class_spec(specialization_value):
    """Given a specialization string like 'Arcane Mage' return ('Mage','Arcane')."""
    if not specialization_value:
        return (None, None)
    parts = specialization_value.split()
    if len(parts) >= 2:
        # Assume last is class, rest joined are spec
        cls = parts[-1]
        spec = " ".join(parts[:-1])
        return (cls, spec)
    return (None, specialization_value)

def prettify_race(race):
    if not race:
        return None
    return " ".join([p.capitalize() for p in race.split('_')])

def parse_json_file(fp, talent_names=None):
    """Parse a SimulationCraft JSON result file into the same row format used for HTML builds.
    Initial implementation: extracts baseline player plus profileset results.
    Talent point breakdown is not decoded from compressed talent strings (set to 0).
    Range % DPS approximated as (max-min)/mean*100.
    """
    rows = []
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️ Error reading JSON {fp}: {e}")
        return rows
    sim = data.get("sim", {})
    opts = sim.get("options", {})
    desired_targets = opts.get("desired_targets")
    # Recursively locate 'players' and 'profilesets' keys (their position can vary)
    def find_key(obj, target):
        if isinstance(obj, dict):
            if target in obj:
                return obj[target]
            for v in obj.values():
                found = find_key(v, target)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for v in obj:
                found = find_key(v, target)
                if found is not None:
                    return found
        return None
    players = find_key(data, "players") or []
    first_player = players[0] if players else {}
    # Baseline (player level) stats
    # Derive base build name from the main profile (e.g., "10t")
    base_build = None
    if isinstance(first_player, dict):
        base_build = first_player.get("name")
    if not base_build:
        # Fallback: infer from filename like "13_10t_001" -> "10t"
        m = re.search(r"_([A-Za-z0-9]+t)_", fp.stem)
        if m:
            base_build = m.group(1)
    baseline_talent_str = None
    if first_player:
        collected = first_player.get("collected_data", {})
        dps_obj = collected.get("dps", {})
        prio_obj = collected.get("prioritydps", {})
        stats_block = collected.get("buffed_stats", {}).get("stats", {})
        # Extract baseline gear trinket names safely
        base_gear = first_player.get("gear", {}) if isinstance(first_player, dict) else {}
        trinket_1_name = (base_gear.get("trinket1", {}) or {}).get("name")
        trinket_2_name = (base_gear.get("trinket2", {}) or {}).get("name")
        mean_dps = dps_obj.get("mean")
        min_dps = dps_obj.get("min")
        max_dps = dps_obj.get("max")
        stddev = dps_obj.get("std_dev") or dps_obj.get("stddev")
        priority_mean = prio_obj.get("mean")
        range_pct = None
        if mean_dps and min_dps is not None and max_dps is not None and mean_dps != 0:
            try:
                range_pct = (max_dps - min_dps) / mean_dps * 100.0
            except Exception:
                range_pct = None
        specialization_value = first_player.get("specialization")
        cls, spec = strip_class_spec(specialization_value)
        race = prettify_race(first_player.get("race"))
        talent_str = first_player.get("talents")
        baseline_talent_str = talent_str
        build_id = first_player.get("name")
        builds = base_build or (strip_chunk_suffix(build_id) if build_id else build_id)
        row = {
            "file": fp.stem,
            "build_id": build_id,
            "builds": builds,
            "dps": mean_dps,
            "priority_dps": priority_mean,
            "race": race,
            "class": cls,
            "spec": spec,
            "range_pct_dps": range_pct,
            "talent_str": talent_str,
            "targets": desired_targets,
            "crit": stats_block.get("crit_rating"),
            "haste": stats_block.get("haste_rating"),
            "vers": stats_block.get("versatility_rating"),
            "mastery": stats_block.get("mastery_rating"),
            "min_dps": min_dps,
            "max_dps": max_dps,
            "StdDev": stddev,
            "trinket_1": trinket_1_name,
            "trinket_2": trinket_2_name
        }
        if talent_names and talent_str:
            # Only decode when a talent string is explicitly present; no fallback/zero fill.
            row.update(decode_talent_string_to_header_ranks(talent_str, talent_names, cls, spec))
        rows.append(row)
        # Cache baseline stats block for profilesets reuse
        baseline_stats_block = stats_block

    # Copy blocks: emit a row per player if multiple players are present
    # Many JSON files with copies have multiple players each representing a copy profile
    for idx, pl in enumerate(players):
        try:
            name = pl.get("name")
            if not name:
                continue
            # Skip baseline if this is the same as first_player already emitted
            if idx == 0:
                # We already added baseline above
                continue
            collected = pl.get("collected_data", {})
            dps_obj = collected.get("dps", {})
            prio_obj = collected.get("prioritydps", {})
            stats_block = collected.get("buffed_stats", {}).get("stats", {})
            gear = pl.get("gear", {})
            trinket_1_name = (gear.get("trinket1", {}) or {}).get("name")
            trinket_2_name = (gear.get("trinket2", {}) or {}).get("name")
            mean_dps = dps_obj.get("mean")
            min_dps = dps_obj.get("min")
            max_dps = dps_obj.get("max")
            stddev = dps_obj.get("std_dev") or dps_obj.get("stddev")
            priority_mean = prio_obj.get("mean")
            range_pct = None
            if mean_dps and min_dps is not None and max_dps is not None and mean_dps != 0:
                try:
                    range_pct = (max_dps - min_dps) / mean_dps * 100.0
                except Exception:
                    range_pct = None
            specialization_value = pl.get("specialization")
            cls, spec = strip_class_spec(specialization_value)
            race = prettify_race(pl.get("race"))
            talent_str = pl.get("talents")  # No fallback; if absent we leave it missing.
            build_id = name
            builds = base_build or (strip_chunk_suffix(build_id) if build_id else build_id)
            row = {
                "file": fp.stem,
                "build_id": build_id,
                "builds": builds,
                "dps": mean_dps,
                "priority_dps": priority_mean,
                "race": race,
                "class": cls,
                "spec": spec,
                "range_pct_dps": range_pct,
                "talent_str": talent_str,
                "targets": desired_targets,
                "crit": stats_block.get("crit_rating"),
                "haste": stats_block.get("haste_rating"),
                "vers": stats_block.get("versatility_rating"),
                "mastery": stats_block.get("mastery_rating"),
                "min_dps": min_dps,
                "max_dps": max_dps,
                "StdDev": stddev,
                "trinket_1": trinket_1_name,
                "trinket_2": trinket_2_name
            }
            if talent_names and talent_str:
                row.update(decode_talent_string_to_header_ranks(talent_str, talent_names, cls, spec))
            rows.append(row)
        except Exception:
            continue
    # Profilesets results
    profilesets_root = find_key(data, "profilesets") or {}
    profilesets = []
    profilesets_metric = None
    if isinstance(profilesets_root, dict):
        results = profilesets_root.get("results")
        if isinstance(results, list):
            profilesets = results
        profilesets_metric = profilesets_root.get("metric")
    metric_lower = (profilesets_metric or "").lower()
    for result in profilesets:
        build_id = result.get("name")
        if not build_id:
            continue
        main_mean = result.get("mean")
        main_min = result.get("min")
        main_max = result.get("max")
        main_stddev = result.get("stddev") or result.get("std_dev")
        # Initialize values for both metrics
        dps_val = None
        prio_val = None
        dps_min = None
        dps_max = None
        dps_stddev = None
        # Assign based on which metric is the primary one
        if "priority" in metric_lower:
            prio_val = main_mean
        else:
            dps_val = main_mean
            dps_min = main_min
            dps_max = main_max
            dps_stddev = main_stddev
        # Look into additional_metrics to capture the other metric if provided
        for add in result.get("additional_metrics", []) or []:
            mname = (add.get("metric") or "").lower()
            if "priority" in mname:
                prio_val = add.get("mean", prio_val)
            elif "damage" in mname:
                dps_val = add.get("mean", dps_val)
                dps_min = add.get("min", dps_min)
                dps_max = add.get("max", dps_max)
                dps_stddev = add.get("stddev") or add.get("std_dev") or dps_stddev
        range_pct = None
        if dps_val and dps_min is not None and dps_max is not None and dps_val != 0:
            try:
                range_pct = (dps_max - dps_min) / dps_val * 100.0
            except Exception:
                range_pct = None
        specialization_value = first_player.get("specialization") if first_player else None
        cls, spec = strip_class_spec(specialization_value)
        race = prettify_race(first_player.get("race")) if first_player else None
        builds = base_build or strip_chunk_suffix(build_id)
        # Use cached baseline stats and gear for profileset rows
        stats_block = baseline_stats_block if 'baseline_stats_block' in locals() else {}
        base_gear = first_player.get("gear", {}) if first_player else {}
        trinket_1_name = (base_gear.get("trinket1", {}) or {}).get("name")
        trinket_2_name = (base_gear.get("trinket2", {}) or {}).get("name")
        # Extract talent string for profileset rows. In profileset result objects
        # the talent string is usually nested under the overrides block, not
        # at the top level, e.g. {"name":..., "overrides": {"stats": {...}, "talents": "C4D..."}}
        # Extract talent string from profileset result; do not fallback to baseline.
        overrides = result.get("overrides") if isinstance(result, dict) else None
        talent_str = result.get("talents") or (overrides.get("talents") if isinstance(overrides, dict) else None)
        row = {
            "file": fp.stem,
            "build_id": build_id,
            "builds": builds,
            "dps": dps_val,
            "priority_dps": prio_val,
            "race": race,
            "class": cls,
            "spec": spec,
            "range_pct_dps": range_pct,
            "talent_str": talent_str,
            "targets": desired_targets,
            "crit": stats_block.get("crit_rating"),
            "haste": stats_block.get("haste_rating"),
            "vers": stats_block.get("versatility_rating"),
            "mastery": stats_block.get("mastery_rating"),
            "min_dps": dps_min,
            "max_dps": dps_max,
            "StdDev": dps_stddev,
            "trinket_1": trinket_1_name,
            "trinket_2": trinket_2_name
        }
        if talent_names and talent_str:
            row.update(decode_talent_string_to_header_ranks(talent_str, talent_names, cls, spec))
        rows.append(row)
    return rows

def strip_chunk_suffix(build_id):
    # Removes trailing underscores followed by digits (multiple blocks)
    return re.sub(r'(?:_\d+)+$', '', build_id)


def clean_build_ids_in_csv(csv_path):
    pass

def get_max_workers():
    import os
    import sys
    from multiprocessing import cpu_count
    env_val = os.environ.get("SIMC_MAX_WORKERS")
    if env_val and env_val.isdigit() and int(env_val) > 0:
        return int(env_val)
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--max-workers"):
            if "=" in arg:
                val = arg.split("=", 1)[1]
            elif i+1 < len(sys.argv):
                val = sys.argv[i+1]
            else:
                continue
            if val.isdigit() and int(val) > 0:
                return int(val)
    cores = cpu_count()
    workers = max(1, int(cores * 0.9))
    return workers

def main():
    # Recursively include HTML and JSON files from simc_data and its subfolders
    html_files = list(HTML_DIR.rglob("*.html"))
    json_files = list(HTML_DIR.rglob("*.json"))
    if not html_files and not json_files:
        print(f"No HTML/JSON result files found under {HTML_DIR} (recursive)")
        return
    talent_headers = load_all_spec_talent_headers()
    fieldnames = [
        "file", "builds", "build_id", "dps", "priority_dps", "race", "class", "spec",
        "range_pct_dps", "talent_str", "targets",
        "crit", "haste", "vers", "mastery",
        "min_dps", "max_dps", "StdDev", "trinket_1", "trinket_2"
    ] + talent_headers
    max_workers = get_max_workers()
    print(f"Using {max_workers} worker(s) for parallel processing.")
    with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        processed_rows = 0
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_fp = {}
            for fp in html_files:
                future_to_fp[executor.submit(parse_file, fp, talent_headers)] = fp
            for fp in json_files:
                future_to_fp[executor.submit(parse_json_file, fp, talent_headers)] = fp
            for future in as_completed(future_to_fp):
                fp = future_to_fp[future]
                try:
                    rows = future.result()
                except Exception as exc:
                    print(f"Error processing {fp.name}: {exc}")
                    continue
                for r in rows:
                    writer.writerow(r)
                    processed_rows += 1
                print(f"Processed {fp.name}: {len(rows)} rows")

    print(f"Done! Wrote {processed_rows} rows to {OUTPUT_CSV}")

    # Remove only talent columns after trinket_2 where all values are identical
    try:
        import pandas as pd
        df = pd.read_csv(OUTPUT_CSV)
        # Final fixup: if targets == 1 and priority_dps is null, copy dps into priority_dps
        if "targets" in df.columns and "priority_dps" in df.columns and "dps" in df.columns:
            df["targets"] = pd.to_numeric(df["targets"], errors="coerce")
            df["priority_dps"] = pd.to_numeric(df["priority_dps"], errors="coerce")
            mask = (df["targets"] == 1) & (df["priority_dps"].isna())
            df.loc[mask, "priority_dps"] = pd.to_numeric(df["dps"], errors="coerce")
        trinket_idx = df.columns.get_loc('trinket_2')
        talent_cols = df.columns[trinket_idx+1:]
        # Only check and drop talent columns
        for col in talent_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        drop_cols = [col for col in talent_cols if df[col].nunique(dropna=True) == 1]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
        # Write back CSV after fixups
        df.to_csv(OUTPUT_CSV, index=False)
    except Exception as e:
        print(f"Error removing constant talent columns: {e}")

    # Cleanup controlled by config
    deleted_parent_dirs = set()
    html_count = 0
    json_count = 0
    if _DEL_HTML:
        try:
            for html_file in HTML_DIR.rglob("*.html"):
                try:
                    html_file.unlink()
                    deleted_parent_dirs.add(html_file.parent)
                    html_count += 1
                except Exception:
                    pass
            print(f"Deleted {html_count} .html files under simc_data (recursive).")
        except Exception as e:
            print(f"Error deleting .html files: {e}")
    else:
        print("Skipped deletion of .html files (config).")

    if _DEL_JSON:
        try:
            for json_file in HTML_DIR.rglob("*.json"):
                try:
                    json_file.unlink()
                    deleted_parent_dirs.add(json_file.parent)
                    json_count += 1
                except Exception:
                    pass
            print(f"Deleted {json_count} .json files under simc_data (recursive).")
        except Exception as e:
            print(f"Error deleting .json files: {e}")
    else:
        print("Skipped deletion of .json files (config).")

    # Attempt to remove just the immediate directories that contained deleted files
    # Only remove one level deep (i.e., children of HTML_DIR) and only if empty.
    if (html_count + json_count) > 0:
        try:
            for d in sorted(deleted_parent_dirs):
                try:
                    if d != HTML_DIR and d.parent == HTML_DIR:
                        d.rmdir()
                        print(f"Removed directory: {d}")
                except Exception:
                    # Ignore non-empty or permission issues; do not recurse higher
                    pass
        except Exception:
            pass
    else:
        print("Skipped directory removal (no files deleted).")
    # Remove simc_data and Temp if they have no remaining files
    try:
        remaining_simc = list(HTML_DIR.rglob('*'))
        if not remaining_simc:
            # Remove directory if empty (no files or subdirs)
            HTML_DIR.rmdir()
            print(f"Removed empty directory: {HTML_DIR}")
    except Exception:
        pass
    try:
        TEMP_DIR = BASE_DIR / "Temp"
        remaining_temp = list(TEMP_DIR.rglob('*'))
        if not remaining_temp:
            TEMP_DIR.rmdir()
            print(f"Removed empty directory: {TEMP_DIR}")
    except Exception:
        pass
#    try:
#        sim_proc_output = BASE_DIR / "sim_proc_output.txt"
#        if sim_proc_output.exists():
#            sim_proc_output.unlink()
#            print("Deleted sim_proc_output.txt.")
#    except Exception as e:
#        print(f"Error deleting sim_proc_output.txt: {e}")

if __name__ == "__main__":
    main()