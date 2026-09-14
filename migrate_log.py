"""migrate_log.py - convert log.json (one big JSON array) to log.jsonl.

The array had to be rewritten in full on every round. JSON Lines appends instead,
and a torn write costs one round rather than the whole history.

    python migrate_log.py            # show what would happen
    python migrate_log.py --apply    # write log.jsonl (log.json is left in place)

log.json is never deleted. Once log.jsonl exists it takes precedence, so delete
or rename the old file yourself when you are satisfied.
"""
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
OLD = PROJECT_DIR / "log.json"
NEW = PROJECT_DIR / "log.jsonl"


def main() -> int:
    apply = "--apply" in sys.argv
    if not OLD.exists():
        print(f"{OLD.name} not found — nothing to migrate.")
        return 0
    try:
        rounds = json.loads(OLD.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"{OLD.name} is not valid JSON: {e}")
        return 1
    if not isinstance(rounds, list):
        print(f"{OLD.name} does not hold a list of rounds.")
        return 1

    if NEW.exists():
        existing = sum(1 for line in NEW.open(encoding="utf-8") if line.strip())
        print(f"{NEW.name} already exists with {existing} rounds. Delete it first "
              f"if you meant to re-run the migration.")
        return 1

    scored = sum(1 for r in rounds if r.get("actual"))
    old_mb = OLD.stat().st_size / 1e6
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rounds)
    print(f"{OLD.name}: {len(rounds)} rounds ({scored} scored), {old_mb:.1f} MB")
    print(f"{NEW.name}: would be {len(body.encode('utf-8')) / 1e6:.1f} MB")
    if not apply:
        print("\nDry run. Re-run with --apply to write it.")
        return 0

    tmp = NEW.with_suffix(".jsonl.tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(NEW)
    check = sum(1 for line in NEW.open(encoding="utf-8") if line.strip())
    if check != len(rounds):
        print(f"MISMATCH: wrote {check} lines for {len(rounds)} rounds — "
              f"{OLD.name} is untouched, investigate before deleting it.")
        return 1
    print(f"Wrote {NEW.name} with {check} rounds. {OLD.name} left untouched as a backup.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
