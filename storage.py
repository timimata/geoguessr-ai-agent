"""Reading and writing the round log.

The log used to be one JSON array rewritten in full on every round, and re-read
several times per round on top of that. It is now JSON Lines: appending a round
is a single write at the end of the file, and reads come from a cache that is
only re-parsed when the file changes underneath it.

`log.json` is still read when `log.jsonl` is absent, so an existing install keeps
working; `python migrate_log.py` converts it once.
"""
import json
import os

import config
from config import say
from geo import haversine_km, lookup_country


def log_path():
    """The active log file: log.jsonl if present, else the legacy log.json."""
    jsonl = config.LOG_FILE.with_suffix(".jsonl")
    if jsonl.exists():
        return jsonl
    return config.LOG_FILE


def _is_jsonl(path) -> bool:
    return path.suffix == ".jsonl"


# (path, size, mtime_ns) -> parsed rounds. Guards against re-parsing 2 MB of JSON
# several times per round while still noticing writes from another process.
_cache: tuple | None = None
_cache_rounds: list[dict] = []


def _stat_key(path):
    try:
        st = path.stat()
        return (str(path), st.st_size, st.st_mtime_ns)
    except OSError:
        return (str(path), -1, -1)


def load_rounds(force: bool = False) -> list[dict]:
    """Every logged round, oldest first. Cached until the file changes."""
    global _cache, _cache_rounds
    path = log_path()
    key = _stat_key(path)
    if not force and _cache == key:
        return _cache_rounds
    rounds: list[dict] = []
    if path.exists():
        try:
            if _is_jsonl(path):
                with path.open("r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rounds.append(json.loads(line))
                        except json.JSONDecodeError:
                            # One torn line (a crash mid-append) must not cost the
                            # whole history, unlike a single JSON array.
                            say(f"[log] skipping malformed line {line_no} of {path.name}")
            else:
                rounds = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            say(f"[log] could not read {path.name}: {e}")
            rounds = []
    _cache, _cache_rounds = _stat_key(path), rounds
    return rounds


def append_round(entry: dict) -> None:
    """Add one round. O(1) on JSONL; falls back to rewriting a legacy array."""
    global _cache, _cache_rounds
    # Warm the cache first: appending to a cold cache would leave it holding only
    # this one round, and every later read would then see a one-round history.
    load_rounds()
    path = log_path()
    if _is_jsonl(path) or not path.exists():
        path = config.LOG_FILE.with_suffix(".jsonl")
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    else:
        data = load_rounds()
        data.append(entry)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    _cache_rounds = _cache_rounds + [entry]
    _cache = _stat_key(path)


# Kept so older call sites and scripts keep working.
append_log = append_round


def write_rounds(rounds: list[dict]) -> None:
    """Replace the whole log. Only for edits and migrations, never per round."""
    global _cache, _cache_rounds
    path = log_path()
    tmp = path.with_suffix(path.suffix + ".tmp")
    if _is_jsonl(path):
        with tmp.open("w", encoding="utf-8") as f:
            for r in rounds:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        tmp.write_text(json.dumps(rounds, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    _cache_rounds = rounds
    _cache = _stat_key(path)


def count_rounds_in_log() -> int:
    return len(load_rounds())


def recent_wrong_countries(n: int = 8) -> list[str]:
    """Distinct countries the bot guessed wrong in the last `n` rounds, so the
    prompt can warn against walking into the same trap twice."""
    out: list[str] = []
    seen: set[str] = set()
    for r in reversed(load_rounds()[-n:]):
        if r.get("country_hit") is False:
            g = (r.get("guess") or {}).get("country", "")
            if g and g != "?" and g not in seen:
                seen.add(g)
                out.append(g)
    return out


def score_entry(entry: dict, actual: tuple[float, float],
                placed: tuple[float, float] | None) -> dict:
    """Fill in actual / placed / error_km / country_hit on a log entry."""
    guess = entry.get("guess") or {}
    country, admin1, cc = lookup_country(*actual)
    entry["actual"] = {"latitude": actual[0], "longitude": actual[1],
                       "country": country, "admin1": admin1, "cc": cc}
    if placed:
        pc, pa, pcc = lookup_country(*placed)
        entry["placed"] = {"latitude": placed[0], "longitude": placed[1],
                           "country": pc, "admin1": pa, "cc": pcc}
        ref = placed
        placed_country = pc
        if guess.get("latitude") is not None and guess.get("longitude") is not None:
            entry["projection_drift_km"] = haversine_km(
                guess["latitude"], guess["longitude"], *placed)
    else:
        ref = (guess.get("latitude"), guess.get("longitude"))
        placed_country = guess.get("country") or ""
    entry["error_km"] = haversine_km(ref[0], ref[1], *actual) if None not in ref else None
    pl = (placed_country or "").strip().lower()
    entry["country_hit"] = bool(country) and bool(pl) and (
        pl == country.lower() or country.lower() in pl or pl in country.lower())
    return entry


def update_log_entry(timestamp: str, patch: dict) -> bool:
    """Patch the round with this timestamp. Rewrites the file, so it is for the
    rare correction (deferred duel scoring), not the per-round path."""
    rounds = load_rounds(force=True)
    for entry in reversed(rounds):
        if entry.get("timestamp") == timestamp:
            entry.update(patch)
            write_rounds(rounds)
            return True
    return False
