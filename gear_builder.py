import itertools
import time
import re
import os
import configparser

def parse_gear_blocks(text):
    lines = [l.strip() for l in text.splitlines() if l.strip().startswith("#") and "=" in l]
    gear_by_slot = {}
    for line in lines:
        slot, item = line[1:].split("=", 1)
        slot = slot.strip()
        item = item.strip()
        base_slot = slot.replace("1", "").replace("2", "") if "finger" in slot or "trinket" in slot else slot
        gear_by_slot.setdefault(base_slot, []).append((slot, item))
    return gear_by_slot

def valid_combinations(gear_map):
    fixed_slots = {s: gear_map[s] for s in gear_map if s not in ["finger", "trinket"]}
    fixed_values = [[f"{slot}={item}" for slot, item in items] for items in fixed_slots.values()]
    finger_items = gear_map.get("finger", [])
    trinket_items = gear_map.get("trinket", [])

    finger_pairs = [
        f"{f1[0]}={f1[1]}\n{f2[0].replace('1', '2')}={f2[1]}"
        for f1 in finger_items for f2 in finger_items if f1 != f2
    ]
    trinket_pairs = [
        f"{t1[0]}={t1[1]}\n{t2[0].replace('1', '2')}={t2[1]}"
        for t1 in trinket_items for t2 in trinket_items if t1 != t2
    ]

    if not finger_pairs: finger_pairs = [""]
    if not trinket_pairs: trinket_pairs = [""]

    for fixed_combo in itertools.product(*fixed_values):
        for fingers in finger_pairs:
            for trinkets in trinket_pairs:
                lines = list(filter(None, fixed_combo))
                if fingers: lines.append(fingers)
                if trinkets: lines.append(trinkets)
                if any(line.startswith("main_hand=") for line in lines) and not any("off_hand=" in l for l in lines):
                    lines.append("off_hand=")
                yield "\n".join(lines)

def is_legal_gearblock(gear_block):
    slots = [line.split("=")[0] for line in gear_block.splitlines() if "=" in line]
    counts = {}
    for slot in slots:
        base = slot.replace("1", "").replace("2", "") if "finger" in slot or "trinket" in slot else slot
        counts[base] = counts.get(base, 0) + 1

    for slot, count in counts.items():
        if slot in ["finger", "trinket"]:
            if count != 2: return False
        else:
            if count > 1: return False
    return True

def parse_talent_file(talent_path):
    with open(talent_path, encoding="utf-8") as f:
        lines = [l.rstrip('\n') for l in f if l.strip()]
    builds = []
    current_build = []
    for line in lines:
        if line.startswith("copy="):
            if current_build:
                builds.append(tuple(current_build))
            current_build = [line]
        else:
            if current_build:
                current_build.append(line)
    if current_build:
        builds.append(tuple(current_build))
    return builds

def parse_profilesets_file(talent_path):
    with open(talent_path, encoding="utf-8") as f:
        lines = [l.rstrip('\n') for l in f if l.strip()]
    # Group by profileset name: profileset."name"+=...
    ps_re = re.compile(r'^profileset\."([^"]+)"\+=')
    groups = {}
    order = []
    for line in lines:
        m = ps_re.match(line)
        if not m:
            continue
        name = m.group(1)
        if name not in groups:
            groups[name] = []
            order.append(name)
        groups[name].append(line)
    return order, groups

def estimate_total(talent_file, gear_map):
    T = len(parse_talent_file(talent_file))
    fixed_combo_count = 1
    for slot in (s for s in gear_map if s not in ["finger", "trinket"]):
        fixed_combo_count *= max(len(gear_map[slot]), 1)
    f_len = len(gear_map.get("finger", []))
    t_len = len(gear_map.get("trinket", []))
    f_pairs = f_len * (f_len - 1) if f_len > 1 else 1
    t_pairs = t_len * (t_len - 1) if t_len > 1 else 1
    return T * fixed_combo_count * f_pairs * t_pairs

def expand_profilesets(talent_file, gear_file, output_file="sim_proc_output.txt", split_files=False):
    with open(gear_file, encoding="utf-8") as f:
        gear_map = parse_gear_blocks(f.read())
    # Parse both formats
    copy_builds = parse_talent_file(talent_file)
    ps_order, ps_groups = parse_profilesets_file(talent_file)
    if not copy_builds and not ps_order:
        print("No valid copy or profileset builds found.")
        return

    # Count total unique legal builds by actually generating and counting them
    print("\nCounting total unique legal builds (this may take a moment)...")
    copy_builds = parse_talent_file(talent_file)
    if not copy_builds and not ps_order:
        print("No valid copy or profileset builds found.")
        return
    seen = set()
    for gear_block in valid_combinations(gear_map):
        if not is_legal_gearblock(gear_block):
            continue
        # Copies signatures
        for build_lines in copy_builds:
            build_str = "\n".join(build_lines)
            sig = f"copy:{build_str}|{'|'.join(sorted(gear_block.splitlines()))}"
            seen.add(sig)
        # Profilesets signatures
        for name in ps_order:
            rows = ps_groups.get(name, [])
            sig = f"ps:{name}|{'|'.join(rows)}|{'|'.join(sorted(gear_block.splitlines()))}"
            seen.add(sig)
    total_unique_builds = len(seen)
    print(f"\nTotal unique legal builds: {total_unique_builds:,}")
    # Read threshold from config; default to 1000
    try:
        cfg = configparser.ConfigParser()
        cfg.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini"))
        threshold = cfg.getint("builder", "min_unique_builds_warning", fallback=1000)
    except Exception:
        threshold = 1000
    if total_unique_builds >= threshold:
        proceed = None
        while proceed not in {"y", "n"}:
            proceed = input(f"⚠️ Warning: Total unique legal builds is {total_unique_builds:,} (≥ {threshold}). Proceed? (y/n): ").strip().lower()
        if proceed == "n":
            print("Aborted by user.")
            return

    # Now actually write the output
    # 1) Write original copies unchanged
    file_ix = 0
    out = open(f"{output_file}.{file_ix}" if split_files else output_file, "w", encoding="utf-8")
    # 1) Write original copies unchanged
    for build_lines in copy_builds:
        out.write("\n".join(build_lines) + "\n\n")
    # 1b) Write original profilesets unchanged
    for name in ps_order:
        for row in ps_groups.get(name, []):
            out.write(row + "\n")
        out.write("\n")

    # 2) Append merged legal builds after originals
    seen.clear()
    start = time.time()
    processed = 0
    for gear_block in valid_combinations(gear_map):
        if not is_legal_gearblock(gear_block):
            continue
        # Append copies with unique suffix
        for build_lines in copy_builds:
            # Append a unique numeric suffix to the copy name for appended entries
            # Only modify the first line if it starts with 'copy='
            if build_lines and build_lines[0].startswith("copy="):
                base = build_lines[0].split("=", 1)[1]
                suffixed_first = f"copy={base}_{processed + 1}"
                build_lines_out = [suffixed_first] + list(build_lines[1:])
            else:
                build_lines_out = list(build_lines)
            build_str = "\n".join(build_lines_out)
            sig = f"copy:{build_lines[0]}|{'|'.join(sorted(gear_block.splitlines()))}"
            if sig in seen:
                continue
            seen.add(sig)
            out.write(f"{build_str}\n{gear_block}\n\n")
            processed += 1

        # Append profilesets with unique suffix on name
        for name in ps_order:
            rows = ps_groups.get(name, [])
            if not rows:
                continue
            suffixed = f"{name}_{processed + 1}"
            # Write all rows with suffixed name
            for row in rows:
                out.write(row.replace(f'profileset."{name}"+=', f'profileset."{suffixed}"+=') + "\n")
            # Then append gear block lines as cumulative profileset rows
            for gl in (gear_block.splitlines()):
                out.write(f'profileset."{suffixed}"+={gl}\n')
            out.write("\n")
            sig = f"ps:{name}|{'|'.join(rows)}|{'|'.join(sorted(gear_block.splitlines()))}"
            if sig in seen:
                continue
            seen.add(sig)
            processed += 1

            if processed % 1000 == 0:
                rate = processed / (time.time() - start)
                print(f"\rProcessed {processed:,} builds ({rate:.1f}/sec)", end="")
            if split_files and processed % 6399 == 0:
                out.close()
                file_ix += 1
                out = open(f"{output_file}.{file_ix}", "w", encoding="utf-8")
    out.close()
    print(f"\nDone! Generated {processed:,} unique legal builds (appended after originals).")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    expand_profilesets(
        talent_file=os.path.join(script_dir, "sim_proc_output.txt"),
        gear_file=os.path.join(script_dir, "sim_proc_gear_input.txt"),
        output_file=os.path.join(script_dir, "sim_proc_output.txt"),
        split_files=False
    )