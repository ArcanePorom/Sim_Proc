import requests
import zipfile
import io
import os
import subprocess
import sys
import shutil
import json
import stat
import time
import tempfile
import configparser

# Define paths
# Install SimulationCraft one level above the Sim_Proc folder
root_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(root_dir)
base_dir = os.path.join(parent_dir, "SimulationCraft")
target_dir = parent_dir
target_exe = os.path.join(target_dir, "simc.exe")

# Resolve branch from CLI or environment (fallback to "midnight")
def resolve_branch(argv: list[str]) -> str:
    """Resolve SimulationCraft branch to install.
    Precedence: --branch flag > SIMC_BRANCH env > config.ini [sim].simc_branch > default
    """
    default_branch = "midnight"
    # Try reading config.ini next to this script
    try:
        cfg = configparser.ConfigParser()
        cfg.read(os.path.join(root_dir, "config.ini"))
        cfg_branch = cfg.get("sim", "simc_branch", fallback=None)
    except Exception:
        cfg_branch = None
    # Support forms: --branch midnight  OR  --branch=midnight
    if "--branch" in argv:
        try:
            idx = argv.index("--branch")
            if idx + 1 < len(argv):
                return argv[idx + 1].strip()
        except ValueError:
            pass
    for a in argv:
        if a.startswith("--branch="):
            return a.split("=", 1)[1].strip()
    env_branch = os.environ.get("SIMC_BRANCH")
    if env_branch:
        return env_branch.strip()
    if cfg_branch:
        return cfg_branch.strip()
    return default_branch

def run_command(cmd, cwd=None):
    # Avoid shell=True to prevent CMD.EXE UNC path issues. Pass list argv directly.
    printable = ' '.join(cmd)
    print(f"\n🔧 Running: {printable}")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Command failed: {printable}")

def download_and_extract_zip(branch: str, force_update: bool = False):
    zip_url = f"https://github.com/simulationcraft/simc/archive/refs/heads/{branch}.zip"
    gh_api_branch = f"https://api.github.com/repos/simulationcraft/simc/branches/{branch}"
    os.makedirs(base_dir, exist_ok=True)

    # Helper: get latest commit SHA for branch from GitHub API
    def get_latest_sha():
        headers = {"User-Agent": "Mozilla/5.0"}
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        r = requests.get(gh_api_branch, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        # Expected structure: { commit: { sha: "..." } }
        return data.get("commit", {}).get("sha")

    # Helper: robust rmtree even on read-only files
    def _on_rm_error(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass

    # Determine if update needed
    latest_sha = None
    try:
        latest_sha = get_latest_sha()
    except Exception as e:
        print(f"⚠️ Could not query GitHub API for latest commit ({e}); will proceed with local cache if present.")

    # Use a branch-specific extract directory
    extract_dir = os.path.join(base_dir, f"simc-{branch}")
    meta_file = os.path.join(extract_dir, ".source_meta.json")
    current_sha = None
    if os.path.isdir(extract_dir) and os.path.isfile(meta_file):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if meta.get("branch") == branch:
                    current_sha = meta.get("commit_sha")
        except Exception:
            current_sha = None

    need_download = False
    reason = ""
    if force_update:
        need_download = True
        reason = "forced update"
    elif not os.path.isdir(extract_dir):
        need_download = True
        reason = "no existing source"
    elif latest_sha and latest_sha != current_sha:
        need_download = True
        reason = f"new commit {latest_sha[:7]} available (was {str(current_sha)[:7] if current_sha else 'unknown'})"
    elif not latest_sha and not current_sha:
        # No API available and no meta — ensure we have something by re-downloading
        need_download = True
        reason = "no metadata and API unavailable"

    if not need_download:
        print(f"📁 Source up to date at {extract_dir} (commit {current_sha[:7] if current_sha else 'unknown'}); skipping download.")
        return extract_dir

    print(f"⬇️ Downloading source from {zip_url}... ({reason})")
    headers = {"User-Agent": "Mozilla/5.0"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(zip_url, headers=headers, timeout=120)
    if response.status_code != 200:
        raise Exception(f"Download failed (HTTP {response.status_code}).")

    # Helper: recursively make files writable
    def _make_writable_recursive(path: str):
        for root, dirs, files in os.walk(path):
            for name in files:
                p = os.path.join(root, name)
                try:
                    os.chmod(p, stat.S_IWRITE)
                except Exception:
                    pass
            for name in dirs:
                p = os.path.join(root, name)
                try:
                    os.chmod(p, stat.S_IWRITE)
                except Exception:
                    pass

    # Helper: remove path with retries
    def _rmtree_retry(path: str, retries: int = 5, delay: float = 0.5):
        last_err = None
        for i in range(retries):
            try:
                if not os.path.exists(path):
                    return True
                _make_writable_recursive(path)
                shutil.rmtree(path, onerror=_on_rm_error)
                return True
            except Exception as e:
                last_err = e
                time.sleep(delay)
        if last_err:
            raise last_err
        return False

    # Extract to a temporary folder first to avoid partial updates on errors
    tmp_root = os.path.join(base_dir, "_tmp_extract")
    if os.path.isdir(tmp_root):
        _rmtree_retry(tmp_root)
    os.makedirs(tmp_root, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(tmp_root)

    # Identify the single top-level folder produced by GitHub zip
    entries = [e for e in os.listdir(tmp_root) if os.path.isdir(os.path.join(tmp_root, e))]
    if not entries:
        raise Exception("Zip extraction did not produce a directory")
    extracted_root = os.path.join(tmp_root, entries[0])

    # Replace existing extract_dir atomically where possible
    use_dir = extract_dir
    replace_ok = True
    if os.path.isdir(extract_dir):
        # Try to rename the old dir out of the way to minimize in-use issues
        backup_dir = extract_dir + f".old-{int(time.time())}"
        try:
            os.replace(extract_dir, backup_dir)
        except Exception:
            # Fall back to rmtree
            try:
                _rmtree_retry(extract_dir)
            except Exception as e:
                print(f"⚠️ Could not remove existing source ({e}); will use a new directory for this run.")
                replace_ok = False
    if replace_ok:
        try:
            shutil.move(extracted_root, extract_dir)
            use_dir = extract_dir
        except Exception as e:
            print(f"⚠️ Could not move new source into place ({e}); using extracted directory directly.")
            use_dir = extracted_root
    else:
        use_dir = extracted_root

    # Clean up tmp root only if the directory in use is not inside tmp_root
    def _is_subpath(path: str, parent: str) -> bool:
        try:
            return os.path.commonpath([os.path.abspath(path), os.path.abspath(parent)]) == os.path.abspath(parent)
        except Exception:
            return False

    if not _is_subpath(use_dir, tmp_root):
        try:
            _rmtree_retry(tmp_root)
        except Exception:
            pass
    print(f"📁 Source extracted to {use_dir}")

    # Write metadata
    try:
        # Write metadata into the directory we will actually use
        os.makedirs(use_dir, exist_ok=True)
        with open(os.path.join(use_dir, ".source_meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "branch": branch,
                "commit_sha": latest_sha,
                "zip_url": zip_url
            }, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not write metadata: {e}")
    return use_dir

def build_simc(source_dir: str):
    build_dir = os.path.join(source_dir, "build")
    # Clean build dir to prevent CMakeCache path mismatches across locations.
    if os.path.isdir(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    # Step 1: Configure CMake
    # Use explicit -S/-B and avoid setting cwd to a UNC path.
    run_command([
        "cmake",
        "-S", source_dir,
        "-B", build_dir,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_GUI=OFF"
    ])

    # Step 2: Build
    run_command([
        "cmake",
        "--build", build_dir,
        "--config", "Release"
    ])

    print("\n✅ Build complete!")

    # Step 3: Move simc.exe
    built_exe = os.path.join(build_dir, "Release", "simc.exe")
    if os.path.exists(built_exe):
        shutil.move(built_exe, target_exe)
        print(f"📦 simc.exe moved to: {target_exe}")
        # Write build info next to the exe for easy verification
        try:
            meta_path = os.path.join(source_dir, ".source_meta.json")
            info_path = os.path.join(target_dir, "simc_build_info.txt")
            info = {}
            if os.path.isfile(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
            with open(info_path, "w", encoding="utf-8") as f:
                f.write("SimulationCraft build source info\n")
                f.write(f"branch: {info.get('branch', 'unknown')}\n")
                f.write(f"commit_sha: {info.get('commit_sha', 'unknown')}\n")
                f.write(f"zip_url: {info.get('zip_url', 'n/a')}\n")
            print(f"🧾 Wrote build info: {info_path}")
        except Exception as e:
            print(f"⚠️ Could not write simc_build_info.txt: {e}")
    else:
        print("⚠️ Build succeeded, but simc.exe not found in expected location.")

if __name__ == "__main__":
    try:
        branch = resolve_branch(sys.argv)
        force = "--force-update" in sys.argv
        print(f"🌿 Using branch: {branch}")
        src_dir = download_and_extract_zip(branch=branch, force_update=force)
        build_simc(src_dir)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
