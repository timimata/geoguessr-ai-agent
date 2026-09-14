import json
from collections import defaultdict
from pathlib import Path

from storage import load_rounds


def main() -> None:
    rounds = load_rounds()
    by_model: dict[str, list] = defaultdict(list)
    for r in rounds:
        m = r.get("model") or "(unknown)"
        if r.get("actual") and r.get("error_km") is not None:
            by_model[m].append(r)

    if not by_model:
        print("No rounds with actual locations.")
        return

    print(f"{'model':<40} {'n':>4} {'hit%':>6} {'avg':>7} {'med':<6} {'<100':>5} {'<500':>5}")
    print("-" * 80)
    for model, rs in sorted(by_model.items()):
        n = len(rs)
        hits = sum(1 for r in rs if r.get("country_hit") is True)
        errs = sorted(r["error_km"] for r in rs)
        avg = sum(errs) / n
        med = errs[n // 2]
        under_100 = sum(1 for e in errs if e < 100)
        under_500 = sum(1 for e in errs if e < 500)
        print(
            f"{model[:40]:<40} {n:>4} {100*hits/n:>5.1f}% "
            f"{avg:>6.0f}km {med:>5.0f}km {under_100:>5} {under_500:>5}"
        )


if __name__ == "__main__":
    main()
