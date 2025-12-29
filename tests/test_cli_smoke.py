import os
import sys
import subprocess
from pathlib import Path


def run_cmd(args):
    proc = subprocess.run([sys.executable, "-m"] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode, proc.stdout.decode(errors="ignore"), proc.stderr.decode(errors="ignore")


def test_stlar_help_runs():
    code, out, err = run_cmd(["stlar"])
    assert code == 0 or (code == 1 and "usage" in out.lower() + err.lower()), f"Unexpected exit code: {code}\n{out}\n{err}"


def test_hilbert_help_runs():
    code, out, err = run_cmd(["stlar", "hilbert-batch", "--help"])
    # argparse prints help and exits 0
    assert code == 0, f"hilbert-batch --help failed: {code}\n{out}\n{err}"


def test_cli_module_imports():
    # Basic import smoke test to catch obvious import errors
    import importlib
    m = importlib.import_module("hfoGUI.cli")
    assert hasattr(m, "build_parser")
