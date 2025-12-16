# Sim_Proc

A Windows-friendly pipeline to build SimulationCraft profiles, run chunked sims, extract metrics, and visualize results with Dash.

## Features
- Chunked SimulationCraft runs with progress bar and parallel workers
- HTML/JSON parsing with rich metrics and automatic cleanup
- Talent picker web UI and 3D results dashboard
- One-command dependency setup, plus a no-Python bootstrap path
- Optional SimulationCraft build helper (installs into the parent folder)

## Requirements
- Windows 10/11
- Python 3.10+ (installer provided)
- SimulationCraft simc.exe (can be built via simc_install.py)

## Quick Start
1) Open PowerShell in this folder.
2) Install Python deps (creates .venv and installs requirements):

```powershell
python .\setup_installer.py
```

- No Python? Use the bootstrap:
```powershell
powershell -ExecutionPolicy Bypass -File .\no_python_installer.ps1
```

3) (Optional) Build SimulationCraft into the parent folder:
```powershell
python .\simc_install.py --branch midnight
```
This places simc.exe beside this Sim_Proc folder (not inside it).

4) Run the pipeline:
```powershell
python .\sim_proc.py
```
Follow prompts to pick talents, build reports, collect metrics, and launch the dashboard.

## Key Scripts
- sim_proc.py: Orchestrates the full flow end-to-end.
- talent_dashboard.py: Web UI to pick talents and generate profile input.
- build_simc_reports.py: Splits work into chunks and runs simc.exe in parallel.
- collect_data_metrics.py: Parses HTML/JSON outputs into a combined CSV and cleans up.
- 3d_dashboard.py: Interactive 3D visualization of results.
- setup_installer.py: Creates venv and installs Python dependencies.
- no_python_installer.ps1: Downloads Python and then runs the installer.
- simc_install.py: Downloads sources and builds simc.exe into the parent directory.

## Configuration (optional)
Create config.ini in this folder to override defaults.

[sim]
- iterations: e.g. 10000
- output_format: html or json
- profiles_per_sim: chunk size, e.g. 100
- workers_core_fraction: fraction of cores to use, e.g. 0.3
- simc_branch: branch to fetch when building SimulationCraft, e.g. midnight

[cleanup]
- delete_html: true|false
- delete_json: true|false

## How simc.exe is found
build_simc_reports.py resolves simc.exe by checking:
1. SIMC_PATH env var (if set)
2. PATH (e.g. Program Files install)
3. Local paths: this folder, the current working directory, then the parent folder (new default install target)
4. Prompts for a path if not found

## Typical Workflow
1. python sim_proc.py → follow prompts to pick talents and generate input
2. build_simc_reports.py runs sims in parallel and saves outputs under simc_data/<job>
3. collect_data_metrics.py aggregates HTML/JSON into sim_proc_combined_results.csv and cleans up
4. 3d_dashboard.py launches the visualization

## Notes
- This repo intentionally excludes SimulationCraft binaries and heavy output data. See .gitignore.
- If ports are busy (8050), close previous dashboards or change the port.
- You can re-run individual steps by running the scripts directly.

## License
Choose a license for your repo when you publish to GitHub.
