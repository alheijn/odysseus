from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path


def report_header(title: str) -> None:
    print(f"\n=== {title} ===")


def try_import(module_name: str) -> bool:
    print(f"Importing {module_name} ...")
    try:
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", "<built-in>")
        print(f"OK: {module_name} -> {module_file}")
        return True
    except Exception as exc:
        print(f"FAILED: {module_name}")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def main() -> int:
    report_header("Python")
    print(f"Executable: {sys.executable}")
    print(f"Version: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', '<not set>')}")

    report_header("Core Imports")
    all_ok = True
    for module_name in [
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "sklearn",
        "sklearn.metrics",
        "sklearn.metrics.cluster._expected_mutual_info_fast",
        "sklearn.model_selection",
        "sklearn.neighbors",
    ]:
        all_ok = try_import(module_name) and all_ok

    report_header("Relevant Paths")
    sklearn_cluster_dir = Path(sys.prefix) / "Lib" / "site-packages" / "sklearn" / "metrics" / "cluster"
    print(f"Expected sklearn cluster dir: {sklearn_cluster_dir}")
    if sklearn_cluster_dir.exists():
        for path in sorted(sklearn_cluster_dir.iterdir()):
            if path.suffix.lower() in {".pyd", ".dll"}:
                print(f"Native extension: {path}")
    else:
        print("Directory not found.")

    report_header("Result")
    if all_ok:
        print("Environment looks usable for KNN.")
        return 0

    print("Environment is not usable for KNN as-is.")
    print("If the failing module is blocked by Windows policy, inspect AppLocker or Code Integrity logs.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
