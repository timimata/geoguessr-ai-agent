"""compress_screenshots.py - re-encode stored round screenshots from PNG to JPEG.

New rounds are already saved as JPEG. This converts the backlog, which is where
almost all of the disk goes. Roughly 4x smaller on real panoramas.

    python compress_screenshots.py             # report what would change
    python compress_screenshots.py --apply     # write the JPEGs, keep the PNGs
    python compress_screenshots.py --apply --delete-originals

The conversion is lossy and cannot be undone, so --apply keeps the PNGs unless
you also pass --delete-originals. Log entries and the vector index refer to
rounds by file name, so both are rewritten to point at the new files.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
SHOTS = PROJECT_DIR / "screenshots"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="actually write the JPEGs")
    ap.add_argument("--delete-originals", action="store_true",
                    help="remove each PNG after its JPEG is verified")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--limit", type=int, default=0, help="only the first N files")
    args = ap.parse_args()

    if not SHOTS.exists():
        print("screenshots/ not found.")
        return 1
    pngs = sorted(SHOTS.glob("*.png"))
    if args.limit:
        pngs = pngs[:args.limit]
    if not pngs:
        print("No PNG screenshots left to convert.")
        return 0

    from PIL import Image

    before = sum(p.stat().st_size for p in pngs)
    print(f"{len(pngs)} PNG screenshots, {human(before)}")

    if not args.apply:
        sample = pngs[:20]
        s_before = s_after = 0
        for p in sample:
            s_before += p.stat().st_size
            img = Image.open(p).convert("RGB")
            from io import BytesIO
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=args.quality, optimize=True)
            s_after += buf.tell()
        ratio = s_before / max(s_after, 1)
        print(f"Sampled {len(sample)}: {human(s_before)} -> {human(s_after)} ({ratio:.1f}x)")
        print(f"Projected total: {human(before)} -> {human(before / ratio)}")
        print("\nDry run. Re-run with --apply to convert (PNGs are kept unless "
              "you also pass --delete-originals).")
        return 0

    converted, after, failed = 0, 0, []
    renames: dict[str, str] = {}
    for i, p in enumerate(pngs, 1):
        jpg = p.with_suffix(".jpg")
        try:
            if not jpg.exists():
                Image.open(p).convert("RGB").save(
                    jpg, format="JPEG", quality=args.quality, optimize=True)
            Image.open(jpg).verify()          # refuse to delete a PNG for a broken JPEG
            after += jpg.stat().st_size
            renames[str(p.relative_to(PROJECT_DIR))] = str(jpg.relative_to(PROJECT_DIR))
            converted += 1
            if args.delete_originals:
                p.unlink()
        except Exception as e:
            failed.append((p.name, str(e)))
        if i % 200 == 0:
            print(f"  {i}/{len(pngs)}...", flush=True)

    print(f"Converted {converted}, failed {len(failed)}. "
          f"{human(before)} -> {human(after)}")
    for name, err in failed[:5]:
        print(f"  failed: {name}: {err}")

    # Point the log at the new files.
    sys.path.insert(0, str(PROJECT_DIR))
    from storage import load_rounds, write_rounds
    rounds = load_rounds(force=True)
    touched = 0
    for r in rounds:
        shot = r.get("screenshot")
        if shot:
            key = shot.replace("\\", "/")
            for old, new in renames.items():
                if old.replace("\\", "/") == key:
                    r["screenshot"] = new
                    touched += 1
                    break
    if touched:
        write_rounds(rounds)
        print(f"Updated {touched} log entries to the .jpg paths.")

    # The frozen benchmark set points at files by path too, so it has to follow
    # or every future benchmark run fails on a missing screenshot.
    bset = PROJECT_DIR / "benchmark_set.json"
    if bset.exists() and renames:
        data = json.loads(bset.read_text(encoding="utf-8"))
        moved = 0
        for row in data.get("rounds", []):
            key = (row.get("screenshot") or "").replace("\\", "/")
            for old_p, new_p in renames.items():
                if old_p.replace("\\", "/") == key:
                    row["screenshot"] = new_p
                    moved += 1
                    break
        if moved:
            bset.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"Updated {moved} entries in {bset.name}.")

    print("\nThe vector index still holds the old embeddings, which is fine: "
          "ids are file stems and those did not change.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
