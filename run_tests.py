"""run_tests.py - run every offline check.

None of these need a browser or a model. They load the real modules and call the
real functions, so they are slow to start (country polygons, OCR models, CLIP)
but they catch the things that matter: coordinate maths, clamping, log handling,
and state shared between modules.

    python run_tests.py
    python run_tests.py --quick     # skip the suites that load the heavy models
"""
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent

# (module, needs the heavy model imports)
SUITES = [
    ("test_browser.py", False),
    ("test_duels_api.py", False),
    ("test_logic.py", True),
]


def main() -> int:
    quick = "--quick" in sys.argv
    suites = [(name, heavy) for name, heavy in SUITES if not (quick and heavy)]
    results = []
    for name, _ in suites:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}", flush=True)
        started = time.time()
        proc = subprocess.run([sys.executable, str(PROJECT_DIR / name)],
                              cwd=PROJECT_DIR, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        out = proc.stdout or ""
        # Reprint only the outcome lines; the detail is in the suite's own output
        # and would bury the summary when three suites run back to back.
        for line in out.splitlines():
            if line.strip().startswith(("FAIL", "ERROR", "RESULT")) or not line.startswith("  "):
                print(line)
        if proc.returncode != 0 and not out.strip():
            print(proc.stderr.strip()[-2000:])
        results.append((name, proc.returncode == 0, time.time() - started))

    print(f"\n{'=' * 60}")
    failed = [n for n, ok, _ in results if not ok]
    for name, ok, secs in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<22} {secs:5.1f}s")
    if quick:
        print("  (--quick skipped: " + ", ".join(n for n, h in SUITES if h) + ")")
    if failed:
        print(f"\n{len(failed)} suite(s) failed: {', '.join(failed)}")
        return 1
    print("\nAll suites passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
