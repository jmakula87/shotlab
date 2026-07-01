#!/usr/bin/env python
"""Run every tests/test_*.py and report a single pass/fail summary.

Each test file is self-contained (prints "N/N passed" and exits non-zero on
failure), so we just run them as subprocesses and tally. Usage:  python run_tests.py
"""

import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    files = sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py")))
    if not files:
        print("no tests found")
        return 1
    failed = []
    total_files = len(files)
    for f in files:
        name = os.path.basename(f)
        res = subprocess.run([sys.executable, "-X", "utf8", f],
                             capture_output=True, text=True)
        last = (res.stdout.strip().splitlines() or ["(no output)"])[-1]
        ok = res.returncode == 0
        print(f"{'PASS' if ok else 'FAIL'}  {name:34s} {last}")
        if not ok:
            failed.append(name)
            if res.stderr.strip():
                print("    " + res.stderr.strip().splitlines()[-1])
    print("-" * 60)
    print(f"{total_files - len(failed)}/{total_files} test files passed"
          + (f"  — FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
