import subprocess
import webbrowser
import time
import sys
from pathlib import Path

# 💡 Change this to the full path where your scripts are located
SCRIPT_DIR = Path(__file__).parent.resolve()

def run_script(script_name, wait=True, new_terminal=False):
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        sys.exit(1)

    python_exe = sys.executable or "python"

    if new_terminal:
        # Launch in a new PowerShell window to avoid CMD's UNC path warning
        window_title = (
            f"TalentPickerWindow_{int(time.time())}"
            if script_name == "talent_picker.py"
            else f"ScriptWindow_{int(time.time())}"
        )
        ps_command = (
            f"$host.UI.RawUI.WindowTitle='{window_title}'; "
            f"Set-Location -LiteralPath '{SCRIPT_DIR}'; "
            f"& '{python_exe}' '{script_path}'"
        )
        proc = subprocess.Popen(
            ["powershell.exe", "-NoExit", "-NoProfile", "-Command", ps_command],
            shell=False,
        )
        return {"title": window_title, "pid": proc.pid}
    else:
        proc = subprocess.Popen([python_exe, str(script_path)], shell=False, cwd=str(SCRIPT_DIR))
        if wait:
            proc.wait()
    return None

def close_window_by_title(window_title: str):
    """Close a console window by its exact title using taskkill filter.
    Uses shell=False to avoid UNC warnings from cmd.exe.
    """
    if not window_title:
        return
    try:
        # The wildcard helps match potential suffix added by the shell
        cmd = [
            "taskkill",
            "/FI",
            f"WINDOWTITLE eq {window_title}*",
            "/T",
            "/F",
        ]
        subprocess.run(cmd, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"⚠️ Could not close window '{window_title}': {e}")

def terminate_process_tree(pid: int):
    """Force-terminate a process tree by PID (Windows)."""
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"⚠️ Could not terminate PID {pid}: {e}")

def close_windows_by_title_prefix(prefix: str):
    """Close any console windows whose title starts with prefix."""
    try:
        cmd = [
            "taskkill",
            "/FI",
            f"WINDOWTITLE eq {prefix}*",
            "/T",
            "/F",
        ]
        subprocess.run(cmd, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"⚠️ Could not close windows with prefix '{prefix}': {e}")

def cleanup_spawned_processes(talent_proc_info: dict | None):
    # Kill tracked talent picker process if still around
    if talent_proc_info and isinstance(talent_proc_info, dict):
        if 'pid' in talent_proc_info:
            terminate_process_tree(talent_proc_info['pid'])
        if 'title' in talent_proc_info:
            close_window_by_title(talent_proc_info['title'])
    # Best-effort: close any leftover helper windows by known prefixes
    close_windows_by_title_prefix("TalentPickerWindow_")
    close_windows_by_title_prefix("ScriptWindow_")

print("\n📥 Step 0: Update SimulationCraft?")
# Ask user whether to update SimulationCraft before proceeding
while True:
    choice = input("Update SimulationCraft? y/n: ").strip().lower()
    if choice in ("y", "n"):
        break
    print("Please enter 'y' or 'n'.")

if choice == "y":
    run_script("simc_install.py", wait=True)
else:
    print("⏭️ Skipping SimulationCraft update.")


print("\n📥 Step 1: Fetching Talent Data...")
run_script("fetch_talents_json.py", wait=True)

# Web app that allows you to prepare talent builds for the next script.

print("\n🎛️ Step 2: Launching Talent Picker, pick your talents...")
# Launch the talent picker and keep process info for later termination
talent_picker_proc = run_script("talent_dashboard.py", wait=True, new_terminal=True)
time.sleep(2)
print("🌐 Opening Chrome to http://127.0.0.1:8050/")
webbrowser.open("http://127.0.0.1:8050")
time.sleep(10)

# Prompt to confirm talents were generated before proceeding
while True:
    run_3 = input("Have you generated the talent output file from Step 2? y/n: ").strip().lower()
    if run_3 in ("y", "n"):
        break
    print("Please enter 'y' or 'n'.")

if run_3 == "y":
    run_script("gear_builder.py", wait=True)
else:
    print("⏭️ Skipping Step 3 until talents are generated. Re-run when ready.")

# Optional gate for Steps 5 and 6
while True:
    run_56 = input("Run Steps 4 and 5 (build reports and collect metrics)? y/n: ").strip().lower()
    if run_56 in ("y", "n"):
        break
    print("Please enter 'y' or 'n'.")

# Close the Talent Picker window before proceeding with any subsequent steps,
# regardless of the user's choice for Steps 4 and 5. This avoids port/contention
# and lingering windows.
if 'talent_picker_proc' in globals() and talent_picker_proc:
    print("\n🛑 Closing Talent Picker before continuing...")
    if isinstance(talent_picker_proc, dict) and 'pid' in talent_picker_proc:
        terminate_process_tree(talent_picker_proc['pid'])
        time.sleep(1)
    if isinstance(talent_picker_proc, dict) and 'title' in talent_picker_proc:
        close_window_by_title(talent_picker_proc['title'])
        time.sleep(1)

if run_56 == "y":
    # This script will take the above copies 
    print("\n⏳ Step 4: Chunking data, running SimulationCraft, generating .html reports...")
    run_script("build_simc_reports.py", wait=True)

    print("\n📊 Step 5: Scanning .html reports and generating .csv dataset...")
    run_script("collect_data_metrics.py", wait=True)
else:
    print("⏭️ Skipping Steps 4 and 5 by user choice.")

# Talent Picker has already been closed above to ensure clean transition

run_script("3d_dashboard.py", wait=True, new_terminal=True)
print("\n🎉 Pipeline complete! Your final dashboard is live.")
print("🌐 Opening Chrome to http://127.0.0.1:8050/")
time.sleep(4)

# Final cleanup prompt
print("\n🧹 Cleanup")
while True:
    end_choice = input("Close terminal and kill any remaining helper windows? y/n: ").strip().lower()
    if end_choice in ("y", "n"):
        break
    print("Please enter 'y' or 'n'.")

if end_choice == "y":
    cleanup_spawned_processes(globals().get('talent_picker_proc'))
    print("✅ Cleanup complete. Exiting...")
    sys.exit(0)
else:
    print("ℹ️ Leaving terminal open. You can close it at any time.")
