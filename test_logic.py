"""test_logic.py - checks on the decision logic that needs no browser and no model.

Exercises the real functions rather than reimplementing them: polygon clamping,
the open-water rescue, confidence parsing and calibration, country lookup, and a
live query against the vector index if one has been built.

Run: python test_logic.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

config.QUIET = True

import geo
import llm
import pipeline
import rag

_failures: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    if not ok:
        _failures.append(label)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))


def check_true(label: str, got) -> None:
    check(label, bool(got), True)


def guess(country, lat, lon, region="", conf=0.8, candidates=None, continent=""):
    return llm.Guess(country=country, region=region, latitude=lat, longitude=lon,
                     confidence=conf, reasoning="test", continent=continent,
                     candidates=candidates or [])


def test_geo():
    print("country lookup and distances")
    name, admin1, cc = geo.lookup_country(38.72, -9.14)   # Lisbon
    check("Lisbon -> Portugal", cc, "PT")
    check_true("Lisbon admin1 is set", admin1)
    d = geo.haversine_km(38.72, -9.14, 40.42, -3.70)      # Lisbon -> Madrid
    check_true("Lisbon to Madrid is 450-550 km", 450 < d < 550)
    check("continent of Chile", geo.continent_of("Chile"), "South America")
    check("continent via alias", geo.continent_of("USA"), "North America")
    check_true("mid-Atlantic is far from land",
               geo.distance_to_nearest_land_km(30.0, -40.0) > 80.0)
    check_true("central Poland is near land",
               geo.distance_to_nearest_land_km(52.0, 20.0) <= 80.0)


def test_clamping():
    print("polygon clamping")
    if not geo._COUNTRY_POLYGONS:
        print("  SKIP  countries.geojson not loaded (run: python obter_fronteiras.py)")
        return
    g = guess("Portugal", 38.0, -12.0)                    # in the Atlantic
    moved = pipeline._clamp_to_country(g)
    check_true("a guess off the coast is moved", moved and moved > 0)
    check_true("and lands inside Portugal", geo.lookup_country(g.latitude, g.longitude)[2] == "PT")

    inside = guess("Portugal", 39.5, -8.0)
    check("a guess already inside is left alone", pipeline._clamp_to_country(inside), None)

    # The model naming one country while pointing at another is the case the
    # clamp exists for: Germany declared, coordinates in Kazakhstan.
    g2 = guess("Germany", 48.0, 67.0)
    pipeline._clamp_to_country(g2)
    check("Germany pin dragged out of Kazakhstan",
          geo.lookup_country(g2.latitude, g2.longitude)[2], "DE")


def test_water_rescue():
    print("open-water rescue")
    pipeline.reload_history()
    g = guess("Portugal", 36.0, -14.0)
    moved = pipeline._rescue_from_water(g)
    check_true("a pin in open water is moved", moved and moved > 0)
    check_true("onto land", geo.distance_to_nearest_land_km(g.latitude, g.longitude) <= 80.0)
    check("a pin on land is left alone",
          pipeline._rescue_from_water(guess("Portugal", 39.5, -8.0)), None)


def test_confidence():
    print("confidence parsing")
    for raw, want in [(0.85, 0.85), (85, 0.85), (150, 1.0), (-1, 0.0),
                      ("0.9", 0.9), (None, 0.0)]:
        check(f"{raw!r} normalises", llm._parse_confidence(raw), want)
    check("NaN normalises", llm._parse_confidence(float("nan")), 0.0)

    print("confidence calibration")
    pipeline.reload_history()
    models = [m for m, rows in pipeline._CALIB.items() if len(rows) >= 40]
    if not models:
        print("  SKIP  not enough scored rounds in the log yet")
        return
    m = models[0]
    cal = pipeline.calibrated_confidence(m, 0.9)
    check_true(f"{m} at 0.90 returns a probability", cal is not None and 0.0 <= cal <= 1.0)


def test_issue_detection():
    print("automatic checks on an answer")
    sig = pipeline._Signals(rag_countries=["Argentina", "Chile", "Uruguay"])
    bad = guess("United States", 31.0, 100.0, continent="Europe")  # coords in China
    kinds = [k for k, _ in pipeline._detect_issues(bad, sig)]
    check_true("continent contradiction is caught", "continent_mismatch" in kinds)
    check_true("coordinates in another country are caught", "country_coords" in kinds)
    check("hard issues counted", pipeline._hard_issue_count(kinds) >= 2, True)

    clean = guess("Portugal", 39.5, -8.0, continent="Europe")
    check("a consistent answer raises nothing",
          [k for k, _ in pipeline._detect_issues(clean, pipeline._Signals())], [])


def test_scoring():
    print("round scoring")
    check("a perfect guess scores 5000", round(pipeline.geoguessr_round_score(0)), 5000)
    check_true("score decays with distance",
               pipeline.geoguessr_round_score(1000) > pipeline.geoguessr_round_score(3000))
    check_true("rag policy keeps a correct country", rag.is_rag_worthy(2500.0, True))
    check("rag policy drops a distant miss", rag.is_rag_worthy(2500.0, False), False)
    check_true("a near miss on the right country is a perfect reference",
               rag.is_perfect_reference(50.0, True))
    check("a near miss on the wrong country is not",
          rag.is_perfect_reference(50.0, False), False)


def test_vision():
    print("compass and biome detection")
    shot = next(Path("screenshots").glob("*.png"), None)
    if not shot:
        print("  SKIP  no screenshots to analyse")
        return
    import vision
    from PIL import Image
    png = shot.read_bytes()
    facing = vision.crop_compass_meta(png)
    valid = ("", "north", "north-east", "east", "south-east",
             "south", "south-west", "west", "north-west")
    check_true(f"compass returns a known direction ({facing or 'none found'})",
               facing in valid)
    hint = vision.analyze_ground_color(Image.open(shot).convert("RGB"))
    check_true("biome analysis returns text or nothing",
               hint is None or isinstance(hint, str))
    # The crop must not be empty, which is what a stale UI-position guess looks like.
    check_true("compass crop is non-empty", len(vision.crop_car_meta(png)) > 100)


def test_cross_module_state():
    print("state shared between modules")
    import json
    import storage
    import tempfile

    # A module that does `from rag import _CONFUSION_PAIRS` freezes the empty
    # startup value, because _load_confusion_pairs() rebinds rag's global. The
    # hint then silently never reaches the prompt. Accessors prevent that.
    rag._load_confusion_pairs()
    check("pipeline sees the same pairs as rag",
          len(pipeline.confusion_pairs()), len(rag.confusion_pairs()))
    check("no refresh needed right after loading",
          rag.needs_confusion_refresh(rag._last_confusion_refresh_count), False)
    check_true("refresh needed 25 rounds later",
               rag.needs_confusion_refresh(rag._last_confusion_refresh_count + 25))

    # Appending before any read must not shrink the cached history to one round.
    real = config.LOG_FILE
    tmp = Path(tempfile.gettempdir()) / "geoguessr_cache_test.json"
    jsonl = tmp.with_suffix(".jsonl")
    try:
        lines = [json.dumps({"timestamp": f"t{i}"}) for i in range(30)]
        jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        config.LOG_FILE = tmp
        storage._cache, storage._cache_rounds = None, []
        storage.append_round({"timestamp": "fresh"})
        check("a cold append keeps the whole history", storage.count_rounds_in_log(), 31)
    finally:
        config.LOG_FILE = real
        jsonl.unlink(missing_ok=True)
        storage._cache, storage._cache_rounds = None, []


def test_rag_query():
    print("vector index query")
    coll, _ = rag.get_chroma_collection()
    if not coll:
        print("  SKIP  no vector index (run: python indexador.py)")
        return
    n = coll.count()
    print(f"  index holds {n} rounds")
    shot = next(Path("screenshots").glob("*.png"), None)
    if not shot:
        print("  SKIP  no screenshots to query with")
        return
    text, countries, top, dist, scores = rag.build_rag_examples(shot, max_examples=3)
    check_true("a query returns without error", isinstance(countries, list))
    # An index with rounds in it must return something. Returning nothing means a
    # broken encoder, not an empty database: a Chroma embedding function that does
    # not subclass EmbeddingFunction indexes fine and then fails every query.
    check_true(f"a non-empty index ({rag.RAG_EMBEDDING}) actually retrieves", bool(countries))
    if countries:
        check_true("matches carry a country", all(isinstance(c, str) for c in countries))
        check_true("the top distance is within the threshold",
                   dist is None or dist <= rag.RAG_DISTANCE_THRESHOLD)
        print(f"  nearest: {countries} (top distance {dist:.3f})" if dist else f"  nearest: {countries}")
    # Excluding the round itself must never return it.
    text2, _, _, _, _ = rag.build_rag_examples(shot, max_examples=3, exclude_ids={shot.stem})
    check_true("self-exclusion is honoured", shot.stem not in (text2 or ""))


def main() -> int:
    for fn in (test_geo, test_clamping, test_water_rescue, test_confidence,
               test_issue_detection, test_scoring, test_vision,
               test_cross_module_state, test_rag_query):
        try:
            fn()
        except Exception as e:
            _failures.append(fn.__name__)
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
    print()
    if _failures:
        print(f"RESULT: {len(_failures)} failure(s): {', '.join(_failures)}")
        return 1
    print("RESULT: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
