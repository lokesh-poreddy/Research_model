"""Exports the RDE-Bench tasks and (if present) run_demo.py's results to
MATLAB-compatible .mat files under matlab_data/, for use with the scripts in
matlab/. Run this after run_demo.py so rde_bench_results.mat reflects your
own run rather than the one captured for the project documentation.

Usage:
    python run_demo.py                 # produces demo_results.json
    python export_to_matlab.py         # produces matlab_data/*.mat
"""
from __future__ import annotations
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from researchforge.benchmarks.tasks import digits_task, synthetic_ecg_task
from researchforge.interop.matlab_export import (
    export_task_to_mat, export_bench_results_to_mat, verify_roundtrip,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "matlab_data")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, task_fn in [("digits", digits_task), ("synthetic_ecg", synthetic_ecg_task)]:
        task = task_fn(seed=0)
        out_path = os.path.join(OUT_DIR, f"{name}_task.mat")
        export_task_to_mat(task, out_path)
        info = verify_roundtrip(out_path)
        print(f"Wrote {out_path} (fields: {', '.join(info['top_level_keys'])})")

    results_path = os.path.join(HERE, "demo_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        out_path = os.path.join(OUT_DIR, "rde_bench_results.mat")
        export_bench_results_to_mat(results, out_path)
        info = verify_roundtrip(out_path)
        print(f"Wrote {out_path} (tasks: {', '.join(info['top_level_keys'])})")
    else:
        print(f"Note: {results_path} not found -- run 'python run_demo.py' first "
              f"to also export rde_bench_results.mat")

    print(f"\nDone. In MATLAB or Octave, cd into matlab/ and run "
          f"visualize_ecg_task or plot_rde_bench_results.")


if __name__ == "__main__":
    main()
