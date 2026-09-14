"""
benchmark.py — offline evaluation of the decision pipeline on past rounds.

Every logged round has a screenshot and the true location, so the whole
pipeline (OCR → RAG → prompt → model → checks → clamp → hedge) can be scored
without a browser. The RAG index is told to ignore the round being evaluated so
it cannot retrieve itself.

Usage
    python benchmark.py --build --n 150 --seed 42     # freeze a sample into benchmark_set.json
    python benchmark.py --label baseline              # run the current pipeline on that set
    python benchmark.py --label no_soil --off soil    # same set, one feature switched off
    python benchmark.py --label quick --limit 20      # smoke test on the first 20 rounds
    python benchmark.py --compare baseline no_soil    # side-by-side table of saved runs
    python benchmark.py --calibration                 # reported confidence vs real hit-rate

Each run writes benchmark_results/<label>_<timestamp>.json with per-round rows,
so runs can be diffed later. Feature names: see FEATURES in config.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
SET_FILE = PROJECT_DIR / "benchmark_set.json"
RESULTS_DIR = PROJECT_DIR / "benchmark_results"
LOG_FILE = PROJECT_DIR / "log.json"


# ---------------------------------------------------------------------------
# Set construction
# ---------------------------------------------------------------------------
def build_set(n: int, seed: int, since: str | None) -> list[dict]:
    rounds = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    pool = []
    for r in rounds:
        actual = r.get("actual") or {}
        shot = r.get("screenshot")
        if not shot or not actual.get("country") or r.get("error_km") is None:
            continue
        if since and str(r.get("timestamp", "")) < since:
            continue
        path = PROJECT_DIR / shot
        if not path.exists():
            continue
        pool.append({
            "id": path.stem,
            "screenshot": str(path.relative_to(PROJECT_DIR)),
            "actual": {"latitude": actual["latitude"], "longitude": actual["longitude"],
                       "country": actual["country"], "admin1": actual.get("admin1", "")},
            "timestamp": r.get("timestamp"),
            "live_model": r.get("model"),
            "live_error_km": r.get("error_km"),
            "live_country_hit": r.get("country_hit"),
        })
    rng = random.Random(seed)
    rng.shuffle(pool)
    chosen = sorted(pool[:n], key=lambda x: x["timestamp"] or "")
    SET_FILE.write_text(json.dumps({"seed": seed, "n": len(chosen), "since": since,
                                    "created": datetime.now().isoformat(timespec="seconds"),
                                    "rounds": chosen}, indent=1, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    top = Counter(c["actual"]["country"] for c in chosen).most_common(8)
    print(f"benchmark_set.json: {len(chosen)} rounds (pool {len(pool)}), seed {seed}")
    print("  top countries:", ", ".join(f"{c} {k}" for c, k in top))
    return chosen


def load_set() -> list[dict]:
    if not SET_FILE.exists():
        sys.exit("benchmark_set.json not found — run with --build first")
    return json.loads(SET_FILE.read_text(encoding="utf-8"))["rounds"]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def score_row(geo, lat: float, lon: float, actual: dict) -> tuple[float, bool, str]:
    err = geo.haversine_km(lat, lon, actual["latitude"], actual["longitude"])
    pin_country, _, _ = geo.lookup_country(lat, lon)
    a = actual["country"].strip().lower()
    p = (pin_country or "").strip().lower()
    hit = bool(a) and bool(p) and (p == a or a in p or p in a)
    return err, hit, pin_country


def summarize(rows: list[dict], stage: str) -> dict:
    errs = [r[stage]["error_km"] for r in rows if r.get(stage)]
    hits = [r[stage]["hit"] for r in rows if r.get(stage)]
    if not errs:
        return {}
    import math
    scores = [5000 * math.exp(-e / 1492.7) for e in errs]
    return {
        "n": len(errs),
        "hit_rate": sum(hits) / len(hits),
        "median_km": statistics.median(errs),
        "mean_km": statistics.mean(errs),
        "mean_score": statistics.mean(scores),
        "under_200km": sum(e < 200 for e in errs) / len(errs),
        "over_2000km": sum(e > 2000 for e in errs) / len(errs),
    }


def fmt_summary(s: dict) -> str:
    if not s:
        return "(no data)"
    return (f"n={s['n']:<4d} hit={s['hit_rate']*100:5.1f}%  score={s['mean_score']:6.0f}  "
            f"median={s['median_km']:6.0f}km  <200km={s['under_200km']*100:4.1f}%  "
            f">2000km={s['over_2000km']*100:4.1f}%")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
async def run(args) -> None:
    if args.off:
        os.environ["FEATURES_OFF"] = ",".join(args.off)
    os.environ["MODE"] = args.mode  # load_dotenv() never overrides an existing variable
    # Heavy imports: country polygons, OCR models, CLIP.
    import config
    import geo
    import pipeline
    import rag
    config.QUIET = not args.verbose
    rag._load_confusion_pairs()
    pipeline.reload_history()
    if args.off:
        unknown = [f for f in args.off if f not in config.FEATURES]
        if unknown:
            sys.exit(f"unknown feature(s): {unknown}. Known: {sorted(config.FEATURES)}")
    model = args.model or config.MODEL_IDS[0]
    rounds = load_set()
    if args.limit:
        rounds = rounds[:args.limit]
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    print(f"run '{args.label}': {len(rounds)} rounds · model {model} · workers {args.workers}"
          f" · features off: {args.off or 'none'}", flush=True)
    if not any(f in args.off for f in ("ocr", "ocr_script", "ocr_literal")):
        try:
            import torch
            if not torch.cuda.is_available():
                print("  note: no CUDA — EasyOCR runs on CPU and dominates the runtime;"
                      " use --limit for quick checks", flush=True)
        except Exception:
            pass

    sem = asyncio.Semaphore(args.workers)
    results: list[dict] = []
    t_run = time.time()

    async def one(i: int, r: dict) -> None:
        async with sem:
            path = PROJECT_DIR / r["screenshot"]
            png = path.read_bytes()
            t0 = time.time()
            try:
                d = await pipeline.decide_guess(png, path, model, recent_wrong=[],
                                                exclude_rag_ids={r["id"]})
            except Exception as e:
                print(f"  [{i+1}/{len(rounds)}] {r['id']} FAILED: {e}")
                results.append({"id": r["id"], "actual": r["actual"], "error": str(e)})
                return
            row = {
                "id": r["id"], "actual": r["actual"], "seconds": round(time.time() - t0, 1),
                "model_calls": d.model_calls, "issues": d.issues, "fallback": d.fallback,
                "clamped_km": d.clamped_km, "hedge": d.hedge,
                "confidence_raw": d.confidence_raw, "confidence_calibrated": d.confidence_calibrated,
                "rag_countries": d.rag_countries, "ocr_text": d.ocr_text[:120],
                "compass": d.compass,
                "fired": {
                    "rag": bool(d.rag_countries),
                    "ocr": bool(d.ocr_text),
                    "compass": bool(d.compass),
                    "confusion_pairs": bool(d.context_fired.get("confusion_pairs")),
                    "metas": bool(d.context_fired.get("metas")),
                    "soil": bool(d.context_fired.get("soil")),
                    "blacklist": bool(d.context_fired.get("blacklist")),
                    "region_prefilter": bool(d.context_fired.get("region_prefilter")),
                    "calibration": d.confidence_calibrated is not None,
                },
                "guess": {"country": d.guess.country, "region": d.guess.region,
                          "lat": d.guess.latitude, "lon": d.guess.longitude,
                          "reasoning": d.guess.reasoning,
                          "candidates": d.guess.candidates},
            }
            # Stage metrics: initial answer, after correction/clamp (pre-hedge), final.
            stages = {"initial": d.initial}
            pre_hedge_lat, pre_hedge_lon = d.guess.latitude, d.guess.longitude
            if d.hedge:
                pre_hedge_lat, pre_hedge_lon = d.hedge["from"]
            for name, g in stages.items():
                if g is not None:
                    err, hit, pc = score_row(geo, g.latitude, g.longitude, r["actual"])
                    row[name] = {"country": g.country, "error_km": round(err, 1), "hit": hit, "pin_country": pc}
            err, hit, pc = score_row(geo, pre_hedge_lat, pre_hedge_lon, r["actual"])
            row["pre_hedge"] = {"country": d.guess.country, "error_km": round(err, 1), "hit": hit, "pin_country": pc}
            err, hit, pc = score_row(geo, d.guess.latitude, d.guess.longitude, r["actual"])
            row["final"] = {"country": d.guess.country, "error_km": round(err, 1), "hit": hit, "pin_country": pc}
            results.append(row)
            # Checkpoint: a full run takes a while (EasyOCR and CLIP are slow on
            # CPU), so keep partial results on disk rather than only at the end.
            if len(results) % 10 == 0:
                out_path.write_text(json.dumps({"label": args.label, "partial": True,
                                                "rows": results}, indent=1, ensure_ascii=False),
                                    encoding="utf-8")
            mark = "✓" if hit else "✗"
            extras = []
            if d.issues:
                extras.append("issues=" + ",".join(d.issues))
            if d.hedge:
                extras.append(f"hedge→{d.hedge['toward']} {d.hedge['moved_km']}km")
            if d.clamped_km:
                extras.append(f"clamp {d.clamped_km:.0f}km")
            print(f"  [{len(results)}/{len(rounds)}] {mark} {d.guess.country:<22} actual {r['actual']['country']:<22} "
                  f"{err:7.0f}km  calls={d.model_calls} {row['seconds']:5.1f}s  {' '.join(extras)}", flush=True)

    await asyncio.gather(*(one(i, r) for i, r in enumerate(rounds)))

    ok = [r for r in results if "final" in r]
    report = {
        "label": args.label, "model": model, "features_off": args.off,
        "set": str(SET_FILE.name), "n_requested": len(rounds), "n_ok": len(ok),
        "seconds_total": round(time.time() - t_run, 1),
        "avg_model_calls": statistics.mean(r["model_calls"] for r in ok) if ok else None,
        "avg_seconds": statistics.mean(r["seconds"] for r in ok) if ok else None,
        "corrections": sum(1 for r in ok if r["issues"]),
        "hedges": sum(1 for r in ok if r["hedge"]),
        "clamps": sum(1 for r in ok if r["clamped_km"]),
        "fallbacks": sum(1 for r in ok if r["fallback"]),
        "fired": _fired_counts(ok),
        "silent_features": _silent_features(ok),
        "stages": {s: summarize(ok, s) for s in ("initial", "pre_hedge", "final")},
        "rows": results,
    }
    out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print_report(report)
    print(f"\nsaved {out_path.relative_to(PROJECT_DIR)}")


# Feature names whose absence across a whole run means something is broken, not
# that the round simply did not call for it. A flag that is on and never fires in
# 150 rounds is the failure mode that let a cross-module aliasing bug disable the
# confusion-pair hint silently for several benchmark runs.
_EXPECT_TO_FIRE = ("rag", "metas", "confusion_pairs", "region_prefilter", "calibration")


def _fired_counts(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for r in rows:
        for name, did in (r.get("fired") or {}).items():
            counts[name] = counts.get(name, 0) + bool(did)
    return counts


def _silent_features(rows: list[dict]) -> list[str]:
    """Enabled features that produced nothing across the entire run."""
    import config as _config
    counts = _fired_counts(rows)
    silent = []
    for name in _EXPECT_TO_FIRE:
        if not _config.FEATURES.get(name, True):
            continue
        if counts.get(name, 0) == 0:
            silent.append(name)
    return silent


def print_report(rep: dict) -> None:
    print(f"\n=== {rep['label']} · {rep['model']} · off={rep['features_off'] or 'none'} ===")
    print(f"rounds ok {rep['n_ok']}/{rep['n_requested']} · avg calls {rep['avg_model_calls']:.2f} · "
          f"avg {rep['avg_seconds']:.1f}s/round · corrections {rep['corrections']} · "
          f"hedges {rep['hedges']} · clamps {rep['clamps']} · fallbacks {rep['fallbacks']}")
    for stage in ("initial", "pre_hedge", "final"):
        print(f"  {stage:<10} {fmt_summary(rep['stages'].get(stage, {}))}")
    fired = rep.get("fired") or {}
    if fired:
        n = max(rep["n_ok"], 1)
        print("  fired      " + "  ".join(f"{k}={v * 100 // n}%" for k, v in sorted(fired.items())))
    silent = rep.get("silent_features") or []
    if silent:
        print(f"  WARNING: enabled but never fired in {rep['n_ok']} rounds: "
              f"{', '.join(silent)} — treat this run as measuring a broken pipeline")
    ok = [r for r in rep["rows"] if "final" in r]
    from collections import Counter
    conf = Counter((r["final"]["pin_country"], r["actual"]["country"]) for r in ok if not r["final"]["hit"])
    if conf:
        print("  top confusions (pin → actual): " +
              "; ".join(f"{g}→{a} {n}" for (g, a), n in conf.most_common(6)))


def find_result(label: str) -> Path:
    p = Path(label)
    if p.exists():
        return p
    cands = sorted(RESULTS_DIR.glob(f"{label}_*.json"))
    if not cands:
        sys.exit(f"no saved run for label '{label}' in {RESULTS_DIR}")
    return cands[-1]


def compare(labels: list[str]) -> None:
    reps = [json.loads(find_result(l).read_text(encoding="utf-8")) for l in labels]
    print(f"{'run':<22}{'n':>4}{'hit%':>7}{'score':>7}{'median':>8}{'<200':>6}{'>2000':>7}{'calls':>7}{'s/rd':>6}")
    for rep in reps:
        s = rep["stages"]["final"]
        print(f"{rep['label'][:21]:<22}{s['n']:>4}{s['hit_rate']*100:>7.1f}{s['mean_score']:>7.0f}"
              f"{s['median_km']:>8.0f}{s['under_200km']*100:>6.1f}{s['over_2000km']*100:>7.1f}"
              f"{rep['avg_model_calls']:>7.2f}{rep['avg_seconds']:>6.1f}")
    if len(reps) == 2:
        a = {r["id"]: r for r in reps[0]["rows"] if "final" in r}
        b = {r["id"]: r for r in reps[1]["rows"] if "final" in r}
        common = sorted(set(a) & set(b))
        wins = [i for i in common if b[i]["final"]["hit"] and not a[i]["final"]["hit"]]
        losses = [i for i in common if a[i]["final"]["hit"] and not b[i]["final"]["hit"]]
        import math
        dscore = [5000 * (math.exp(-b[i]["final"]["error_km"] / 1492.7) - math.exp(-a[i]["final"]["error_km"] / 1492.7))
                  for i in common]
        print(f"\n{reps[1]['label']} vs {reps[0]['label']} on {len(common)} common rounds: "
              f"+{len(wins)} countries gained, -{len(losses)} lost, "
              f"Δscore/round {statistics.mean(dscore) if dscore else 0:+.0f}")
        for i in wins[:5]:
            print(f"  gained {i}: {a[i]['final']['country']} → {b[i]['final']['country']} (actual {a[i]['actual']['country']})")
        for i in losses[:5]:
            print(f"  lost   {i}: {a[i]['final']['country']} → {b[i]['final']['country']} (actual {a[i]['actual']['country']})")


def calibration_table() -> None:
    rounds = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    from collections import defaultdict
    by_model: dict[str, dict[float, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in rounds:
        if r.get("country_hit") is None:
            continue
        c = (r.get("guess") or {}).get("confidence", 0.0)
        try:
            c = float(c)
        except (TypeError, ValueError):
            continue
        if c > 1:
            c /= 100
        b = round(max(0.0, min(c, 1.0)) * 20) / 20
        by_model[r.get("model") or "?"][b].append(bool(r["country_hit"]))
    for model, bins in by_model.items():
        n = sum(len(v) for v in bins.values())
        if n < 30:
            continue
        print(f"\n{model} (n={n}) — reported confidence → real country hit-rate")
        for b in sorted(bins):
            v = bins[b]
            if len(v) >= 5:
                bar = "█" * int(round(20 * sum(v) / len(v)))
                print(f"  {b:4.2f}  {sum(v)/len(v)*100:5.1f}%  n={len(v):<4d} {bar}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="(re)create benchmark_set.json")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--since", help="only rounds with timestamp >= YYYYMMDD (for --build)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--model", help="model id (default: first of MODEL_IDS)")
    ap.add_argument("--mode", default="solo", help="bot MODE to evaluate under (solo/duels/team-duels)")
    ap.add_argument("--off", default="", help="comma-separated features to disable")
    ap.add_argument("--limit", type=int, default=0, help="only the first N rounds of the set")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--verbose", action="store_true", help="print the per-round pipeline chatter")
    ap.add_argument("--compare", nargs="+", metavar="LABEL")
    ap.add_argument("--calibration", action="store_true")
    args = ap.parse_args()
    args.off = [s.strip() for s in args.off.split(",") if s.strip()]

    if args.calibration:
        calibration_table()
        return
    if args.compare:
        compare(args.compare)
        return
    if args.build:
        build_set(args.n, args.seed, args.since)
        if args.label == "run":
            return
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
