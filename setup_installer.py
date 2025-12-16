import os
import sys
import subprocess
import urllib.request
from pathlib import Path

SIM_PROC_ROOT = Path(__file__).parent.resolve()
VENV_DIR = SIM_PROC_ROOT / ".venv"
REQ_FILE = SIM_PROC_ROOT / "requirements.txt"
PYTHON_MIN_VERSION = (3, 10)
PYTHON_DOWNLOAD_URL = "https://www.python.org/ftp/python/3.12.6/python-3.12.6-amd64.exe"


def run(cmd, cwd=None, env=None):
    print(f"-> {cmd}")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, shell=False, check=False)


def have_python():
    try:
        out = subprocess.run([sys.executable, "-V"], capture_output=True, text=True, shell=False)
        return out.returncode == 0
    except Exception:
        return False


def download_python_installer(target: Path):
    print(f"Downloading Python from {PYTHON_DOWNLOAD_URL} ...")
    urllib.request.urlretrieve(PYTHON_DOWNLOAD_URL, str(target))
    print(f"Saved to {target}")


def install_python_silently(installer_path: Path):
    print("Running silent Python install...")
    args = [str(installer_path), "/quiet", "InstallAllUsers=1", "PrependPath=1", "Include_test=0"]
    run(args)


def ensure_python():
    if have_python():
        print(f"Python detected at {sys.executable}")
        return
    print("Python not detected via sys.executable. Attempting download...")
    installer = SIM_PROC_ROOT / "python-installer.exe"
    download_python_installer(installer)
    install_python_silently(installer)
    print("Python installation attempted. Please restart shell if needed and re-run.")


def ensure_venv():
    if VENV_DIR.exists():
        print(f"Virtual environment exists at {VENV_DIR}")
    else:
        print("Creating virtual environment...")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    python_venv = VENV_DIR / "Scripts" / "python.exe"
    if not python_venv.exists():
        raise RuntimeError("Venv python not found; venv creation likely failed.")
    return python_venv


def pip_install(python_venv: Path, offline: bool = False):
    print("Upgrading pip...")
    run([str(python_venv), "-m", "pip", "install", "--upgrade", "pip"])
    if not REQ_FILE.exists():
        raise FileNotFoundError(f"requirements.txt not found at {REQ_FILE}")
    print("Installing requirements (skip-if-present)...")
    vendor = SIM_PROC_ROOT / "vendor"
    # Install line-by-line to avoid aborting entirely if one package fails.
    failures = []
    # Map PyPI names to import names for quick presence check
    import_map = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'plotly': 'plotly',
        'dash': 'dash',
        'dash-bootstrap-components': 'dash_bootstrap_components',
        'requests': 'requests',
        'beautifulsoup4': 'bs4',
        'matplotlib': 'matplotlib',
        'pefile': 'pefile',
        'bitarray': 'bitarray',
        'fixedint': 'fixedint'
    }
    for line in REQ_FILE.read_text(encoding="utf-8").splitlines():
        pkg = line.strip()
        if not pkg or pkg.startswith('#'):
            continue
        base_name = pkg.split('==')[0].split('>=')[0].split('>')[0].split('<')[0]
        import_name = import_map.get(base_name, None)
        if import_name:
            # Check if already installed in venv; skip if import succeeds
            check_code = f"import importlib; importlib.import_module('{import_name}'); print('HAVE')"
            res_check = subprocess.run([str(python_venv), "-c", check_code], capture_output=True, text=True, shell=False)
            if res_check.returncode == 0 and res_check.stdout.strip().endswith('HAVE'):
                print(f"Skipping already-present package: {pkg}")
                continue
        if offline and vendor.exists():
            cmd = [str(python_venv), "-m", "pip", "install", "--no-index", "--find-links", str(vendor), pkg]
        else:
            cmd = [str(python_venv), "-m", "pip", "install", pkg]
        res = run(cmd)
        if res.returncode != 0:
            failures.append(pkg)
    if failures:
        print("\nSome packages failed to install:")
        for f in failures:
            print(f" - {f}")
        print("Continuing with verification; optional or ancillary packages can be handled separately.")


def verify_imports(python_venv: Path):
    verifier = SIM_PROC_ROOT / "verify_env.py"
    code = (
        "import importlib\n"
        "mods = [\n"
        "    'pandas','numpy','sklearn','plotly','dash','dash_bootstrap_components',\n"
        "    'requests','bs4','matplotlib','pefile','bitarray','fixedint'\n"
        "]\n"
        "have = []\n"
        "missing = []\n"
        "for m in mods:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "        have.append(m)\n"
        "    except Exception as e:\n"
        "        missing.append((m, str(e)))\n"
        "print('Environment verification:')\n"
        "if have:\n"
        "    print('Present:')\n"
        "    for m in have:\n"
        "        print(' -', m)\n"
        "if missing:\n"
        "    print('Missing:')\n"
        "    for m,e in missing:\n"
        "        print(' -', m, ':', e)\n"
        "if not missing:\n"
        "    print('OK: all required imports succeeded')\n"
    )
    verifier.write_text(code, encoding="utf-8")
    print("Verifying imports...")
    run([str(python_venv), str(verifier)])

def dependencies_satisfied(python_cmd: str) -> bool:
    """Check if required modules are importable using provided python executable."""
    mods = [
        'pandas','numpy','sklearn','plotly','dash','dash_bootstrap_components',
        'requests','bs4','matplotlib','pefile','bitarray','fixedint'
    ]
    py_code = (
        "import importlib,sys; mods=" + str(mods) + 
        "; missing=[m for m in mods if __import__('importlib').import_module(m) is None if False]; print('OK')"
    )
    try:
        res = subprocess.run([python_cmd, "-c", py_code], capture_output=True, text=True, shell=False)
        return res.returncode == 0 and res.stdout.strip().endswith('OK')
    except Exception:
        return False


# Note: This installer intentionally excludes SimulationCraft setup.
# It only ensures Python and required packages for Sim_Proc scripts.


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Sim_Proc Installer")
    ap.add_argument("--offline", action="store_true", help="Install from local vendor cache")
    ap.add_argument("--force", action="store_true", help="Force re-install even if dependencies are already present")
    args = ap.parse_args()

    ensure_python()

    # If system Python already has all deps and user didn't force, skip entirely.
    if not args.force and dependencies_satisfied(sys.executable):
        print("All required Python dependencies appear to be present in your current Python. Skipping venv and install.")
        print(f"To run: python .\\sim_proc.py (using your existing environment)")
        return

    python_venv = ensure_venv()
    # If venv already satisfies deps and not forced, skip installing packages.
    venv_python = str(python_venv)
    if not args.force and dependencies_satisfied(venv_python):
        print("Venv already has required dependencies. Skipping package installation.")
    else:
        pip_install(python_venv, offline=args.offline)
    verify_imports(python_venv)
    print("\nInstaller finished. To run the pipeline:")
    print(f"  {VENV_DIR}\\Scripts\\Activate.ps1")
    print(f"  python .\\sim_proc.py")

if __name__ == "__main__":
    main()
