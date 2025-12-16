#!/usr/bin/env python3
"""
Fetch Raidbots talents.json for a selected environment and save as talents.json in this folder.

Usage:
  - Interactive: python fetch_talents_json.py  # prompts: live, beta, or xptr
  - Non-interactive: python fetch_talents_json.py <live|beta|xptr>
"""
from __future__ import annotations
import sys
import os
import urllib.request
import urllib.error
import configparser

URLS = {
    "live": "https://www.raidbots.com/static/data/live/talents.json",
    "beta": "https://www.raidbots.com/static/data/beta/talents.json",
    "xptr": "https://www.raidbots.com/static/data/xptr/talents.json",
}


def prompt_mode() -> str:
    while True:
        try:
            mode = input("Environment (live, beta, or xptr): ").strip().lower()
        except EOFError:
            mode = ""
        if mode in URLS:
            return mode
        # accept shorthand
        if mode in {"l", "b", "x"}:
            return {"l": "live", "b": "beta", "x": "xptr"}[mode]
        print("Please enter exactly: live, beta, or xptr.")


def fetch_and_save(mode: str, out_path: str) -> int:
    url = URLS[mode]
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    with open(out_path, "wb") as f:
        f.write(data)
    return len(data)


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    mode = None
    if argv:
        arg = argv[0].strip().lower()
        if arg in URLS:
            mode = arg
    if not mode:
        # Try config.ini [talents].env
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cfg = configparser.ConfigParser()
        try:
            cfg.read(os.path.join(script_dir, "config.ini"))
            cfg_mode = cfg.get("talents", "env", fallback=None)
            if cfg_mode and cfg_mode.lower() in URLS:
                mode = cfg_mode.lower()
        except Exception:
            mode = None
    if not mode:
        # Fallback to prompt mode
        mode = prompt_mode()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(script_dir, "talents.json")

    try:
        size = fetch_and_save(mode, out_file)
        print(f"Saved talents.json ({size} bytes) from '{mode}' to: {out_file}")
        return 0
    except urllib.error.HTTPError as e:
        print(f"HTTP error {e.code} fetching {mode}: {e}")
        return 1
    except urllib.error.URLError as e:
        print(f"Network error fetching {mode}: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
