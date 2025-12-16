#!/usr/bin/env python3
import os
import sys
import queue
import shutil
import subprocess
import time
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
import configparser

# ----- CONFIG -----
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR       = os.path.dirname(SCRIPT_DIR)
APL_FILE         = os.path.join(SCRIPT_DIR, "Primary_APL.simc")
PROFILES_FILE    = os.path.join(SCRIPT_DIR, "sim_proc_output.txt")
SIMC_PATH        = None
ITERATIONS       = "10000"
OUTPUT_FORMAT    = "json"  # or "html"
# MAX_WORKERS is configurable via env var, CLI arg, or defaults to min(4, cpu_count())

def get_max_workers():
    import os
    import sys
    from multiprocessing import cpu_count
    # 1. Check environment variable
    env_val = os.environ.get("SIMC_MAX_WORKERS")
    if env_val and env_val.isdigit() and int(env_val) > 0:
        return int(env_val)
    # 2. Check command-line argument
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
    # 3. Default: fraction of CPU cores from config (workers_core_fraction), else 0.3
    frac = 0.3
    try:
        cfg = configparser.ConfigParser()
        cfg.read(os.path.join(SCRIPT_DIR, "config.ini"))
        raw = cfg.get("sim", "workers_core_fraction", fallback="0.3").strip()
        if raw.endswith("%"):
            raw = raw[:-1]
        f = float(raw)
        # Support values like 30 -> 0.30
        if f > 1:
            f = f / 100.0
        # Clamp to sensible bounds
        if 0.01 <= f <= 1.0:
            frac = f
    except Exception:
        pass
    cores = cpu_count()
    workers = max(8, int(cores * frac))
    return workers

MAX_WORKERS      = get_max_workers()
BAR_LENGTH       = 40
PROFILES_PER_SIM = 100
ROOT_DIR         = os.getcwd()
OUTPUT_DIR       = os.path.join(SCRIPT_DIR, "simc_data")
TEMP_DIR         = os.path.join(SCRIPT_DIR, "Temp")

# ----- READ CONFIG.INI OVERRIDES -----
try:
    _cfg = configparser.ConfigParser()
    _cfg.read(os.path.join(SCRIPT_DIR, "config.ini"))
    # iterations
    ITERATIONS = _cfg.get("sim", "iterations", fallback=ITERATIONS)
    # output format (validated later)
    ofmt = _cfg.get("sim", "output_format", fallback=OUTPUT_FORMAT)
    if isinstance(ofmt, str) and ofmt.lower() in {"html", "json"}:
        OUTPUT_FORMAT = ofmt.lower()
    # profiles per sim chunk
    pps = _cfg.get("sim", "profiles_per_sim", fallback="").strip()
    if pps.isdigit() and int(pps) > 0:
        PROFILES_PER_SIM = int(pps)
except Exception:
    pass

# ----- JOB ID (two-digit) -----
def get_next_job_id(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    used = set()
    try:
        for name in os.listdir(base_dir):
            if len(name) == 2 and name.isdigit() and os.path.isdir(os.path.join(base_dir, name)):
                used.add(int(name))
    except FileNotFoundError:
        pass
    for i in range(100):
        if i not in used:
            return f"{i:02d}"
    # Fallback if all 00..99 exist: roll over
    return "00"

JOB_ID          = get_next_job_id(OUTPUT_DIR)
JOB_OUTPUT_DIR  = os.path.join(OUTPUT_DIR, JOB_ID)

# ----- PARSE APL FILE + # Explore -----
def get_actor_names(apl_text):
    names = []
    base_name = None
    found_explore = False

    for line in apl_text.splitlines():
        line = line.strip()
        if line.startswith("mage=") and base_name is None:
            base_name = line.split("=", 1)[1].strip()
        elif line.startswith("# Explore"):
            found_explore = True
        elif found_explore and line.startswith("# copy="):
            names.append(line.split("=", 1)[1].strip())

    if base_name:
        names.insert(0, base_name)

    return names

def replace_mage_line(apl_text, actor):
    new_lines = []
    replaced = False
    for line in apl_text.splitlines():
        if not replaced and line.strip().startswith("mage="):
            new_lines.append(f'mage="{actor}"')
            replaced = True
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

# ----- PARSE PROFILE BLOCKS -----
def parse_profiles():
    """
    Return a list of profile blocks, where each block is a list of lines.
    - copy= blocks: contiguous lines following a leading 'copy=' line
    - profilesets: group ALL lines by the same profileset name (in quotes)
      so that each unique profileset counts as a single unit for chunking.
    """
    lines = [l.strip() for l in open(PROFILES_FILE, encoding="utf-8") if l.strip()]

    # 1) Collect contiguous copy= blocks
    blocks: list[list[str]] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("copy="):
            block = [lines[i]]
            i += 1
            while i < len(lines) and not (lines[i].startswith("copy=") or lines[i].startswith("profileset.")):
                block.append(lines[i])
                i += 1
            blocks.append(block)
        else:
            i += 1

    # 2) Aggregate ALL profileset lines by name (regardless of adjacency)
    import re as _re
    ps_re = _re.compile(r'^profileset\."([^"]+)"\+=')
    ps_groups: dict[str, list[str]] = {}
    for line in lines:
        m = ps_re.match(line)
        if m:
            name = m.group(1)
            ps_groups.setdefault(name, []).append(line)

    # Append each profileset group as a distinct block
    for name, rows in ps_groups.items():
        if rows:
            blocks.append(list(rows))

    return blocks

# ----- FIND simc.exe -----
def find_simc():
    if SIMC_PATH and os.path.isfile(SIMC_PATH):
        return SIMC_PATH
    for exe in ("simc.exe", "simc"):
        p = shutil.which(exe)
        if p: return p
        for p in (
            os.path.join(SCRIPT_DIR, "simc.exe"),
            os.path.join(ROOT_DIR, "simc.exe"),
            os.path.join(PARENT_DIR, "simc.exe"),

     ):
         if os.path.isfile(p): return p
    ans = input("Enter full path to simc.exe: ").strip().strip('"')
    if os.path.isfile(ans): return ans
    print("❌ Could not locate simc.exe.")
    sys.exit(1)

# ----- PROGRESS BAR -----
def refresh(done, total, last_file):
    pct = done / total if total else 1.0
    filled = int(BAR_LENGTH * pct)
    bar = "#" * filled + "-" * (BAR_LENGTH - filled)
    sys.stdout.write(f"\rSimulation Progress: [{bar}] {done}/{total} → {last_file}")
    sys.stdout.flush()

# ----- SIMULATION WORKER -----
def simulate_and_cleanup(simc_path, simc_file, html_file, result_queue):
    cmd = [
        simc_path,
        f"input={simc_file}",
        f"iterations={ITERATIONS}",
        f"{OUTPUT_FORMAT}={html_file}"
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Sim failed: {simc_file} → {e.stderr.decode().strip()}")
    # Remove chunk file after sim completes
    try:
        os.remove(simc_file)
    except Exception:
        pass
    if os.path.isfile(html_file):
        # Move into job-specific folder
        os.makedirs(JOB_OUTPUT_DIR, exist_ok=True)
        shutil.move(html_file, os.path.join(JOB_OUTPUT_DIR, html_file))
    result_queue.put(html_file)

# ----- MAIN FLOW -----
if __name__ == "__main__":
    start_time = time.perf_counter()
    # Ensure temp directory exists for chunk files
    os.makedirs(TEMP_DIR, exist_ok=True)
    apl_text_raw     = open(APL_FILE, "r", encoding="utf-8").read()
    actor_names      = get_actor_names(apl_text_raw)
    profile_blocks   = parse_profiles()
    # Resolve output format: if not set via config, prompt user
    if OUTPUT_FORMAT not in {"html", "json"}:
        try:
            choice = ""
            while choice not in {"html", "json"}:
                choice = input("Select output format ('html' or 'json'): ").strip().lower()
            OUTPUT_FORMAT = choice
        except Exception:
            # Fallback to default html if input is not available
            OUTPUT_FORMAT = OUTPUT_FORMAT or "html"
    simc_path        = find_simc()

    os.makedirs(JOB_OUTPUT_DIR, exist_ok=True)

    print(f"\n🔍 Found {len(actor_names)-1} # Explore actors")
    print(f"📎 Found {len(profile_blocks)} profile blocks\n")

    jobs = []
    for actor in actor_names:
        clean_actor = actor.strip('"')  # ✅ clean actor name for identifiers
        # Chunk by 100 profile units, where each unit is either a single copy= block
        # or a grouped profileset block containing all lines for that profileset name.
        chunks = [profile_blocks[i:i+PROFILES_PER_SIM] for i in range(0, len(profile_blocks), PROFILES_PER_SIM)]
        apl_text = replace_mage_line(apl_text_raw, actor)

        # Only scan between # Explore and mage= for copy= actors
        actor_lines = []
        lines = apl_text_raw.splitlines()
        explore_start = None
        mage_start = None
        for idx, line in enumerate(lines):
            if line.strip().startswith("# Explore") and explore_start is None:
                explore_start = idx
            elif line.strip().startswith("mage=") and explore_start is not None and mage_start is None:
                mage_start = idx
        # If both markers found, scan only that region
        if explore_start is not None and mage_start is not None:
            region = lines[explore_start+1:mage_start]
            for i, line in enumerate(region):
                if line.strip().startswith("# copy="):
                    name = line.split("=", 1)[1].strip()
                    if name == actor:
                        # Collect all following commented lines until next # copy=
                        j = i + 1
                        while j < len(region):
                            next_line = region[j].strip()
                            if next_line.startswith("# copy="):
                                break
                            if next_line and next_line.startswith("#"):
                                # Uncomment the line
                                actor_lines.append(next_line[1:].lstrip())
                            j += 1
                        break

        for chunk_index, block in enumerate(chunks):
            simc_file = os.path.join(TEMP_DIR, f"temp_{JOB_ID}_chunk_{len(jobs):05d}.simc")
            html_file = f"{JOB_ID}_{clean_actor}_{chunk_index+1:03d}.{OUTPUT_FORMAT}"

            with open(simc_file, "w", encoding="utf-8") as f:
                f.write(apl_text + "\n\n")
                if actor_lines:
                    f.write("\n".join(actor_lines) + "\n\n")
                for profile_index, profile in enumerate(block):
                    renamed = []
                    for line in profile:
                        if line.startswith("copy="):
                            # Keep the original build name from the output file (no chunk-based renaming)
                            renamed.append(line)
                        elif line.startswith("profileset."):
                            # Keep original profileset name; do not prefix with actor/chunk
                            try:
                                # Extract name within quotes and the "+=" tail (and rebuild unchanged)
                                q1 = line.find('"')
                                q2 = line.find('"', q1+1)
                                base_name = line[q1+1:q2]
                                tail = line[q2+1:]  # starts with +=...
                                renamed.append(f'profileset."{base_name}"{tail}')
                            except Exception:
                                renamed.append(line)
                        else:
                            renamed.append(line)
                    f.write("\n".join(renamed) + "\n\n")

            jobs.append((simc_file, html_file))

    total_jobs = len(jobs)
    result_queue = queue.Queue()
    refresh(0, total_jobs, "")

    def worker(simc_file, html_file):
        simulate_and_cleanup(simc_path, simc_file, html_file, result_queue)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for simc_file, html_file in jobs:
            pool.submit(worker, simc_file, html_file)

        done = 0
        last_file = ""
        while done < total_jobs:
            try:
                completed = result_queue.get(timeout=0.2)
                done += 1
                last_file = completed
                refresh(done, total_jobs, last_file)
            except queue.Empty:
                pass

    elapsed = time.perf_counter() - start_time
    # Format elapsed as H:MM:SS.mmm
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    time_str = f"{hours}:{minutes:02d}:{seconds:05.2f}" if hours else f"{minutes}:{seconds:05.2f}"
    print(f"\n\n✅ All {total_jobs} simulations complete. {OUTPUT_FORMAT.upper()} files saved to simc_data/{JOB_ID}/ folder.")
    print(f"⏱️ Elapsed time: {time_str}\n")