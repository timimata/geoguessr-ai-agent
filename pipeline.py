"""The decision pipeline: screenshot in, final pin out, with no browser access.

play_round() in bot.py and benchmark.py both call decide_guess(), which is what
makes a logged round replayable offline.
"""
import asyncio
import json
import math
import os
import random
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from PIL import Image

import config
from config import (IS_MULTIPLAYER, _C, feature_on, _ts, say)
from geo import (WORLD_BORDERS, _COUNTRY_ALIASES, _COUNTRY_CENTROIDS,
                 _COUNTRY_POLYGONS, Point, continent_of, country_bbox_hint,
                 distance_to_nearest_land_km, haversine_km, lookup_country,
                 nearest_points, region_coord_hint)
from llm import Guess, ask_model
from rag import build_rag_examples, confusion_pairs, get_country_metas
from storage import load_rounds
from vision import (_SCRIPT_COUNTRIES, _ocr_reader, analyze_ground_color,
                    crop_compass_meta, detect_country_from_ocr, detect_script,
                    extract_text_from_image, mask_duels_ui)

# ===========================================================================
# Decision pipeline (browser-independent).
#
# play_round() and benchmark.py both call decide_guess(); everything that
# touches the Playwright page stays in play_round(). The pipeline is:
#
#   signals   OCR + compass + RAG in parallel
#   context   prompt hints (blacklist, script constraints, metas, ...)
#   call #1   main model call (up to 3 attempts on transport errors)
#   issues    deterministic checks on the answer (OCR contradiction, continent
#             contradiction, RAG disagreement, coords outside country / ocean)
#   call #2   ONE corrective call listing every issue (only if any)
#   clamp     polygon clamp if coords are still outside the declared country
#   calib     replace raw confidence with the historical hit-rate at that value
#   hedge     move the pin between candidates when expected score improves
#
# Worst case is 2 model calls per round (was up to 7).
# ===========================================================================

# Transport retries for the main call. Instant retries are useless against the
# failure they actually hit (rate limiting), so they back off; in duels the round
# is on a clock, so the total wait is capped.
MODEL_ATTEMPTS = int(os.getenv("MODEL_ATTEMPTS", "3"))
MODEL_RETRY_BUDGET_S = float(os.getenv("MODEL_RETRY_BUDGET_S", "8"))


def _retry_delay(attempt: int, exc: Exception) -> float:
    """Seconds to wait before the next attempt: 1s, 2s, 4s with jitter, and
    longer when the endpoint said it was rate limiting."""
    base = 2.0 ** attempt
    text = f"{type(exc).__name__} {exc}".lower()
    if "rate" in text or "429" in text or "quota" in text or "overloaded" in text:
        base *= 3.0
    return min(base, 30.0) * random.uniform(0.8, 1.2)


def _p(msg: str = "") -> None:
    """Per-round pipeline chatter. Silenced by config.QUIET (benchmark runs)."""
    say(msg)


def _norm_country(name: str) -> str:
    s = (name or "").strip().lower()
    return _COUNTRY_ALIASES.get(s, s)


def _norm_place(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = s.replace("ß", "ss")
    for a, b in (("ü", "ue"), ("ö", "oe"), ("ä", "ae"),
                 ("å", "aa"), ("ø", "oe"), ("æ", "ae")):
        s = s.replace(a, b)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = "".join(c if c.isalnum() else " " for c in s)
    return " ".join(s.split())


def validate_guess_coords(g: Guess) -> tuple[bool, bool, bool, str, str]:
    """(country_ok, region_ok, on_land, coord_country, coord_admin1) via reverse geocoding."""
    cc, a1, c_iso = lookup_country(g.latitude, g.longitude)
    dc = (g.country or "").strip().lower()
    dr = (g.region or "").strip().lower()
    cc_lc = cc.lower()
    a1_lc = a1.lower()
    dc_n = _COUNTRY_ALIASES.get(dc, dc)
    cc_n = _COUNTRY_ALIASES.get(cc_lc, cc_lc)
    dc_norm = _norm_place(dc_n)
    cc_norm = _norm_place(cc_n)
    c_ok = (
        not dc or not cc or cc_n == dc_n
        or dc_n in cc_n or cc_n in dc_n
        or dc == c_iso.lower()
        or (dc_norm and cc_norm and (dc_norm == cc_norm
            or dc_norm in cc_norm or cc_norm in dc_norm))
    )
    dr_norm = _norm_place(dr)
    a1_norm = _norm_place(a1_lc)
    r_ok = (
        not dr or not a1 or a1_lc == dr
        or dr in a1_lc or a1_lc in dr
        or (dr_norm and a1_norm and (dr_norm == a1_norm
            or dr_norm in a1_norm or a1_norm in dr_norm))
    )
    on_land = distance_to_nearest_land_km(g.latitude, g.longitude) <= 80.0
    return c_ok, r_ok, on_land, cc, a1


# ---------------------------------------------------------------------------
# History-derived tables: confidence calibration + per-country location prior.
# Loaded once from the round log (call reload_history() after many new rounds).
# ---------------------------------------------------------------------------
_CALIB: dict[str, list[tuple[float, bool, str]]] = {}          # model -> [(conf, hit, stem)]
_HISTORY_ACTUALS: dict[str, list[tuple[float, float, str]]] = {}  # country -> [(lat, lon, stem)]
_WITHIN_ERR: dict[str, list[float]] = {}                          # model -> [error_km when country hit]
_HISTORY_LOADED = False


def _norm_conf(c) -> float:
    try:
        c = float(c)
    except (TypeError, ValueError):
        return 0.0
    if c > 1.0:
        c = c / 100.0
    return min(max(c, 0.0), 1.0)


def reload_history() -> None:
    global _CALIB, _HISTORY_ACTUALS, _HISTORY_LOADED, _WITHIN_ERR
    calib: dict[str, list[tuple[float, bool, str]]] = {}
    actuals: dict[str, list[tuple[float, float, str]]] = {}
    within: dict[str, list[float]] = {}
    for r in load_rounds():
        if True:
            actual = r.get("actual") or {}
            if not actual or r.get("country_hit") is None:
                continue
            stem = Path(r.get("screenshot") or "").stem
            model = r.get("model") or "?"
            conf = _norm_conf((r.get("guess") or {}).get("confidence", 0.0))
            calib.setdefault(model, []).append((conf, bool(r.get("country_hit")), stem))
            if r.get("country_hit") and r.get("error_km") is not None:
                within.setdefault(model, []).append(float(r["error_km"]))
            c = _norm_country(actual.get("country", ""))
            try:
                actuals.setdefault(c, []).append(
                    (float(actual["latitude"]), float(actual["longitude"]), stem)
                )
            except (KeyError, TypeError, ValueError):
                pass
    _CALIB, _HISTORY_ACTUALS, _WITHIN_ERR, _HISTORY_LOADED = calib, actuals, within, True


def _ensure_history() -> None:
    if not _HISTORY_LOADED:
        reload_history()


def calibrated_confidence(model: str, conf: float, exclude_ids: set[str] | None = None) -> float | None:
    """Historical P(country correct | model reported ~conf). None if too little data."""
    _ensure_history()
    rows = list(_CALIB.get(model, []))
    if len(rows) < 40:
        rows = [row for m_rows in _CALIB.values() for row in m_rows]
    if exclude_ids:
        rows = [row for row in rows if row[2] not in exclude_ids]
    if len(rows) < 40:
        return None
    for win in (0.05, 0.10, 0.20):
        sel = [h for c, h, _ in rows if abs(c - conf) <= win]
        if len(sel) >= 15:
            return sum(sel) / len(sel)
    return sum(h for _, h, _ in rows) / len(rows)


# ---------------------------------------------------------------------------
# Expected-score hedging between candidate countries.
# GeoGuessr world-map score ≈ 5000·exp(-km/1492.7). When the model is torn
# between two neighbours (USA/Canada, UK/Ireland, Germany/Austria) a pin between
# them has a higher expected score than committing to either.
# ---------------------------------------------------------------------------
SCORE_DECAY_KM = 1492.7


def geoguessr_round_score(error_km: float) -> float:
    return 5000.0 * math.exp(-max(error_km, 0.0) / SCORE_DECAY_KM)


def _within_country_error_km(model: str) -> float:
    """Median error when the country was right — how far the model's pin usually
    is from the truth even on a correct country. Used as the spread of the
    primary hypothesis when hedging."""
    _ensure_history()
    rows = _WITHIN_ERR.get(model) or [e for v in _WITHIN_ERR.values() for e in v]
    if len(rows) < 20:
        return 300.0
    rows = sorted(rows)
    return max(60.0, min(rows[len(rows) // 2], 800.0))


def _samples_for_country(country_n: str, near_lat: float, near_lon: float,
                         exclude_ids: set[str] | None = None,
                         k: int = 20) -> list[tuple[float, float, float]]:
    """Where the truth would plausibly be if the scene is in `country_n`:
    the k past ACTUAL locations in that country closest to the primary pin,
    weighted by proximity (coverage prior conditioned on the model's pin).
    Returns [(weight, lat, lon)] with weights summing to 1."""
    _ensure_history()
    pts = [(la, lo) for la, lo, stem in _HISTORY_ACTUALS.get(country_n, [])
           if (not exclude_ids or stem not in exclude_ids) and abs(lo - near_lon) <= 180.0]
    if not pts:
        c = _COUNTRY_CENTROIDS.get(country_n)
        return [(1.0, c[0], c[1])] if c else []
    pts.sort(key=lambda p: haversine_km(near_lat, near_lon, p[0], p[1]))
    pts = pts[:k]
    ws = [math.exp(-haversine_km(near_lat, near_lon, la, lo) / 1500.0) for la, lo in pts]
    tot = sum(ws) or 1.0
    return [(w / tot, la, lo) for w, (la, lo) in zip(ws, pts)]


def _primary_samples(lat: float, lon: float, spread_km: float) -> list[tuple[float, float, float]]:
    """9-point stencil around the model's pin: centre + 8 directions at `spread_km`."""
    out = [(0.2, lat, lon)]
    dlat = spread_km / 111.0
    dlon = spread_km / max(111.0 * math.cos(math.radians(lat)), 1e-3)
    for i in range(8):
        a = math.radians(i * 45.0)
        out.append((0.1, lat + dlat * math.sin(a), lon + dlon * math.cos(a)))
    return out


def hedge_guess(guess: Guess, p_primary: float, model: str = "",
                exclude_ids: set[str] | None = None) -> dict | None:
    """Move the pin toward a rival candidate when that raises the expected score.

    Expected score = Σ_country P(country) · E[5000·exp(-km/1492.7)] where the
    truth-given-country distribution is the model's pin ± its usual within-country
    error (primary) or past actual locations in that country (alternatives).
    Returns a hedge record or None when committing to the primary pin is best."""
    if not guess.candidates or not guess.country:
        return None
    primary_n = _norm_country(guess.country)
    alts: list[tuple[str, str, float]] = []
    for c, conf in guess.candidates:
        n = _norm_country(c)
        if not n or n == primary_n or conf <= 0:
            continue
        alts.append((n, c, float(conf)))
    if not alts:
        return None
    p = min(max(p_primary, 0.0), 0.999)
    if p >= 0.85:
        return None  # not worth the loss on a likely-correct primary
    rest = 1.0 - p
    tot = sum(a[2] for a in alts)

    hyps: list[tuple[float, str, list[tuple[float, float, float]]]] = [
        (p, guess.country, _primary_samples(guess.latitude, guess.longitude,
                                             _within_country_error_km(model)))
    ]
    for n, c, conf in alts:
        s = _samples_for_country(n, guess.latitude, guess.longitude, exclude_ids)
        if s:
            hyps.append((rest * conf / tot, c, s))
    if len(hyps) < 2:
        return None

    def exp_score(lat: float, lon: float) -> float:
        total = 0.0
        for pr, _, samples in hyps:
            total += pr * sum(w * geoguessr_round_score(haversine_km(lat, lon, la, lo))
                              for w, la, lo in samples)
        return total

    base = exp_score(guess.latitude, guess.longitude)
    best_s, best_lat, best_lon, best_alt, best_t = base, guess.latitude, guess.longitude, None, 0.0
    for pr, cname, samples in hyps[1:]:
        # Target = weighted centre of that hypothesis' samples.
        tlat = sum(w * la for w, la, _ in samples)
        tlon = sum(w * lo for w, _, lo in samples)
        for t in (0.15, 0.30, 0.45, 0.60, 0.80, 1.00):
            lat = guess.latitude + (tlat - guess.latitude) * t
            lon = guess.longitude + (tlon - guess.longitude) * t
            s = exp_score(lat, lon)
            if s > best_s:
                best_s, best_lat, best_lon, best_alt, best_t = s, lat, lon, cname, t
    if best_alt is None or best_s < base * 1.02:
        return None
    return {
        "from": [round(guess.latitude, 4), round(guess.longitude, 4)],
        "to": [round(best_lat, 4), round(best_lon, 4)],
        "toward": best_alt,
        "t": best_t,
        "p_primary": round(p, 3),
        "expected_before": round(base),
        "expected_after": round(best_s),
        "moved_km": round(haversine_km(guess.latitude, guess.longitude, best_lat, best_lon)),
    }


# ---------------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    guess: Guess
    initial: Guess | None = None          # first model answer
    corrected: Guess | None = None        # answer after the correction call (if any)
    model_calls: int = 0
    issues: list[str] = field(default_factory=list)
    clamped_km: float | None = None
    confidence_raw: float = 0.0
    confidence_calibrated: float | None = None
    hedge: dict | None = None
    ocr_text: str = ""
    compass: str | None = None
    rag_countries: list[str] = field(default_factory=list)
    rag_scores: dict[str, float] = field(default_factory=dict)
    fallback: bool = False
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class _Signals:
    ocr_text: str = ""
    ocr_script: str | None = None
    ocr_literal_country: str | None = None
    compass: str | None = None
    compass_bytes: bytes | None = None
    rag_text: str = ""
    rag_countries: list[str] = field(default_factory=list)
    rag_top_country: str | None = None
    rag_top_dist: float | None = None
    rag_scores: dict[str, float] = field(default_factory=dict)


_OCR_UI_NOISE = (
    "round 1", "round 2", "round 3", "round 4", "round 5",
    "round 6", "round 7", "round 8", "watch replay",
    "game breakdown", "your score", "your health",
    "has guessed", "hos gvessed", "hasguessed",
    "team blue", "team red",
)


async def _gather_signals(png: bytes, shot_path: Path,
                          exclude_rag_ids: set[str] | None) -> _Signals:
    sig = _Signals()
    pil_img = Image.open(BytesIO(png)).convert("RGB")
    w_img, h_img = pil_img.size
    if feature_on("compass"):
        compass_box = (int(w_img * 0.35), 0, int(w_img * 0.65), int(h_img * 0.20))
        buf = BytesIO()
        pil_img.crop(compass_box).save(buf, format="JPEG", quality=85)
        sig.compass_bytes = buf.getvalue()

    ocr_input = mask_duels_ui(png) if IS_MULTIPLAYER else png
    ocr_c = (asyncio.to_thread(extract_text_from_image, ocr_input)
             if feature_on("ocr") and _ocr_reader is not None else asyncio.sleep(0, result=""))
    comp_c = (asyncio.to_thread(crop_compass_meta, png)
              if feature_on("compass") else asyncio.sleep(0, result=""))
    rag_c = (asyncio.to_thread(build_rag_examples, shot_path, 3, exclude_rag_ids)
             if feature_on("rag") else asyncio.sleep(0, result=("", [], None, None, {})))
    res = await asyncio.gather(ocr_c, comp_c, rag_c, return_exceptions=True)

    ocr_text = res[0] if isinstance(res[0], str) else ""
    if ocr_text and IS_MULTIPLAYER:
        lc = ocr_text.lower()
        hit = [s for s in _OCR_UI_NOISE if s in lc]
        if hit:
            _p(f"  [ocr] discarded — UI noise detected ('{hit[0]}')")
            ocr_text = ""
    sig.ocr_text = ocr_text
    if ocr_text:
        sig.ocr_script = detect_script(ocr_text) if feature_on("ocr_script") else None
        sig.ocr_literal_country = detect_country_from_ocr(ocr_text) if feature_on("ocr_literal") else None

    comp = res[1] if isinstance(res[1], str) else ""
    sig.compass = comp or None

    rag = res[2] if isinstance(res[2], tuple) and len(res[2]) == 5 else ("", [], None, None, {})
    sig.rag_text, sig.rag_countries, sig.rag_top_country, sig.rag_top_dist, sig.rag_scores = rag
    return sig


def _build_context(sig: _Signals, pil_img, recent_wrong: list[str]) -> list[dict]:
    """All prompt hints that go with the main call, each behind a feature flag."""
    extra: list[dict] = []

    if feature_on("blacklist") and recent_wrong:
        extra.append({
            "role": "user",
            "content": (
                f"SESSION HISTORY: in the last few rounds you incorrectly guessed: "
                f"{', '.join(recent_wrong[:5])}. Do NOT default to any of these again "
                f"unless you have SPECIFIC visual evidence (unique landmark, readable "
                f"sign text, plate code) that confirms it. Prefer the RAG top-1 if you "
                f"would otherwise pick one of these by default."
            ),
        })
        _p(f"  [blacklist] recent wrong: {', '.join(recent_wrong[:5])}")

    if sig.ocr_text:
        _p(f"  [ocr] texto detetado: {sig.ocr_text[:80]}...")
        extra.append({
            "role": "user",
            "content": f"OCR System detected the following text in the environment: '{sig.ocr_text}'. Use this to identify the language, alphabet (e.g., Cyrillic vs Latin), or specific city/street names."
        })
        if sig.ocr_script:
            allowed = _SCRIPT_COUNTRIES.get(sig.ocr_script, ())
            if allowed:
                _p(f"  [ocr-script] {sig.ocr_script} → restringido a: {', '.join(allowed)}")
                extra.append({
                    "role": "user",
                    "content": (
                        f"HARD CONSTRAINT: the OCR text contains {sig.ocr_script.upper()} script. "
                        f"The answer MUST be one of: {', '.join(allowed)}. "
                        f"No other country can produce this script on public signage."
                    )
                })
        if sig.ocr_literal_country:
            _p(f"  [ocr-literal] país identificado pelo texto OCR: {sig.ocr_literal_country}")
            extra.append({
                "role": "user",
                "content": (
                    f"HARD CONSTRAINT: the OCR text contains a verbatim reference to "
                    f"{sig.ocr_literal_country} (country name or unique local token). "
                    f"The answer MUST be {sig.ocr_literal_country}."
                )
            })

    if sig.compass:
        cd = sig.compass.lower()
        rel_map = {
            "north":       ("ahead=N", "right=E", "behind=S", "left=W"),
            "north-east":  ("ahead=NE", "right=SE", "behind=SW", "left=NW"),
            "east":        ("ahead=E", "right=S", "behind=W", "left=N"),
            "south-east":  ("ahead=SE", "right=SW", "behind=NW", "left=NE"),
            "south":       ("ahead=S", "right=W", "behind=N", "left=E"),
            "south-west":  ("ahead=SW", "right=NW", "behind=NE", "left=SE"),
            "west":        ("ahead=W", "right=N", "behind=E", "left=S"),
            "north-west":  ("ahead=NW", "right=NE", "behind=SE", "left=SW"),
        }
        rel_line = " ".join(rel_map.get(cd, ()))
        extra.append({
            "role": "user",
            "content": (
                f"COMPASS + SUN ANALYSIS (opencv-detected).\n"
                f"Camera is facing {sig.compass}. Screen-direction → world-bearing: {rel_line}.\n"
                f"HEMISPHERE INFERENCE (use this):\n"
                f"1. Look at shadows of vertical objects (poles, people, trees, signs).\n"
                f"2. Translate the shadow direction on screen into a world bearing using "
                f"the mapping above (e.g., if shadow points LEFT on screen and camera faces "
                f"North, shadow points West → sun is in the East → it is morning).\n"
                f"3. If shadows point NORTH (±45°) → sun in the south → NORTHERN hemisphere.\n"
                f"4. If shadows point SOUTH (±45°) → sun in the north → SOUTHERN hemisphere.\n"
                f"5. Very short/near-vertical shadows → near the tropics (sun overhead).\n"
                f"State your inferred hemisphere in reasoning and let it narrow down the "
                f"candidate country list."
            ),
        })
        _p(f"  [compass meta] OpenCV detectou: a câmara está virada para {sig.compass}")

    similar = sig.rag_countries
    if sig.rag_text:
        _p(f"  [rag] visually similar to: {', '.join(similar)}")
        extra.append({"role": "user", "content": sig.rag_text})

    if feature_on("soil"):
        try:
            soil_hint = analyze_ground_color(pil_img)
            if soil_hint and "DENSE GREEN TROPICAL" in soil_hint and similar:
                non_tropical = {
                    "russian federation", "russia", "finland", "sweden", "norway",
                    "iceland", "denmark", "estonia", "latvia", "lithuania", "belarus",
                    "canada", "united kingdom", "ireland", "netherlands", "germany",
                    "poland", "czechia", "slovakia", "austria", "switzerland",
                    "france", "ukraine", "romania", "hungary",
                }
                overlap = {c.lower() for c in similar[:3]} & non_tropical
                if overlap:
                    _p(f"  [soil] suppressing tropical hint — RAG suggests non-tropical: {overlap}")
                    soil_hint = None
            if soil_hint:
                extra.append({"role": "user", "content": soil_hint})
                _p(f"  [soil] {soil_hint[:80]}...")
        except Exception as e:
            _p(f"  [soil error] {e}")

    if sig.rag_text and feature_on("metas"):
        top_countries = similar[:3]
        metas_text = get_country_metas(top_countries) if top_countries else ""
        if metas_text:
            _p(f"  [meta] loaded cheat sheets for {', '.join(top_countries)} ({len(metas_text)} chars)")
            _top_conts = {continent_of(c) for c in top_countries if continent_of(c)}
            _top_norm = {_norm_country(c) for c in top_countries}
            parts = [
                "IMPORTANT — DO NOT FALL FOR FAKE-UNIQUE CLUES. Many features that cheat "
                "sheets claim are 'US-specific' or 'X-specific' actually exist in several "
                "countries:\n"
                "• Yellow center road lines: USA, Canada, Mexico, Japan, Norway, Finland, "
                "and also parts of Russia, Albania, South Korea — NOT uniquely US.\n"
                "• Left-hand driving: UK, Ireland, Japan, Australia, NZ, India, South Africa, "
                "Thailand, Indonesia, Malaysia, Hong Kong, Macau, Kenya, Bhutan — cross-check.\n"
                "• Latin alphabet on signs: most of the world — NOT evidence of Europe.\n"
                "• Cyrillic: Russia, Ukraine, Belarus, Bulgaria, Serbia, Kazakhstan, Mongolia, "
                "North Macedonia — NOT uniquely Russia.\n"
            ]
            if "North America" in _top_conts or any(c in _top_norm for c in ("united states", "canada", "mexico")):
                parts.append(
                    "• US vs Canada: both yellow center lines + English. Canada = km/metric, "
                    "occasional ARRÊT bilingual stop, maple-leaf flags, less billboard density.\n"
                )
            if "Europe" in _top_conts:
                parts.append(
                    "• Germany vs Austria: Germany = YELLOW town entry signs, 'Einbahnstraße', "
                    "2-bolt bollards, wind turbines. Austria = WHITE town signs, 'EINBAHN'.\n"
                    "• France vs neighbors: pointed white bollards + full reflector band + small "
                    "yellow D-number signs = France. Absent → consider Spain/Portugal/Italy.\n"
                    "• Romania vs Eastern Europe: holey poles all the way down + yellow pole "
                    "stickers = Romania. Poland has thick HORIZONTAL RED BAND bollards.\n"
                    "• Sweden vs Finland: short white edge dashes + 'väg'/'gata' = Sweden. "
                    "Finland = double vowels (aamu, katu). Norway = YELLOW center lines.\n"
                    "• Türkiye vs Romania: Ğ/İ/Ş chars + TR plate red strip = Türkiye. "
                    "Minaret = Türkiye.\n"
                )
            if "Africa" in _top_conts or "south africa" in _top_norm:
                parts.append(
                    "• South Africa vs Europe: LEFT-HAND traffic + YELLOW outer road lines + "
                    "bird poles. No European country has all three.\n"
                )
            if "South America" in _top_conts:
                parts.append(
                    "• South America — DO NOT default to Brazil. Distinguishing clues:\n"
                    "  - Brazil: red soil, white+blue-stripe Mercosul plates, rectangular "
                    "concrete poles, double yellow center lines, Portuguese (-ção/-nh).\n"
                    "  - Argentina: black/white+blue plates, white-and-red chevrons, Pampas "
                    "flat grassland, wide tree-lined roads.\n"
                    "  - Chile: narrow country, Atacama desert north, Andes east, EU-style "
                    "white+blue plates.\n"
                    "  - Peru: Andes or Pacific coastal desert, adobe construction, yellow "
                    "plates, poor roads.\n"
                    "  - Colombia: yellow plates, black-white cross on sign backs.\n"
                    "  - Ecuador: Andes always nearby, yellow plates.\n"
                    "  - Uruguay: flat grassland, Mercosul plates, red/white chevrons.\n"
                )
            parts.append(
                "Always prefer the #1 RAG match, but override when you see a country-specific "
                "clue from the cheat sheet that definitively identifies a different country.\n"
            )
            extra.append({
                "role": "user",
                "content": (
                    f"{''.join(parts)}\n"
                    f"Here are expert GeoGuessr cheat sheets for the TOP visually-similar "
                    f"countries. Cross-reference these with the visual clues (poles, lines, "
                    f"cars) to find the exact match:\n\n{metas_text}"
                )
            })

    if feature_on("india_hint") and any(c.lower() == "india" for c in similar[:3]):
        extra.append({
            "role": "user",
            "content": (
                "INDIA REGIONAL REASONING REQUIRED: If you conclude this is India, you MUST "
                "first identify the region before picking coordinates:\n"
                "- NORTH INDIA (lat 28-32°N): Delhi NCR (28.6°N,77.2°E), Punjab, Haryana, UP — "
                "flat plains, wheat fields, wide highways, Hindi/Punjabi signage.\n"
                "- SOUTH INDIA (lat 10-15°N): Karnataka/Bangalore (12.9°N,77.6°E), Tamil Nadu, "
                "Kerala — lush green, palm trees, Kannada/Tamil/Malayalam script.\n"
                "- WEST INDIA (lat 18-23°N): Mumbai (19°N,73°E), Gujarat, Rajasthan — "
                "arid/semi-arid, Marathi/Gujarati/Hindi signs, flat desert (Rajasthan).\n"
                "- EAST INDIA (lat 20-25°N): Kolkata (22.5°N,88.3°E), West Bengal, Odisha — "
                "humid, Bengali script, rice paddies, red soil.\n"
                "- NORTHEAST INDIA (lat 24-28°N, lon 88-97°E): Assam, Meghalaya — "
                "very green, hilly, Assamese/Bengali script.\n"
                "- CENTRAL INDIA (lat 20-24°N, lon 75-82°E): MP, Chhattisgarh — "
                "dry deciduous forest, rocky terrain.\n"
                "Do NOT default to Bangalore (12.97,77.59) unless visual clues confirm South India."
            )
        })
        _p("  [india] injecting regional reasoning constraint")

    if feature_on("south_africa_hint") and any(c.lower() == "south africa" for c in similar[:3]):
        extra.append({
            "role": "user",
            "content": (
                "SOUTH AFRICA IDENTIFICATION: Key confirms — LEFT-HAND traffic, YELLOW outer road "
                "lines (unique in Africa), 'bird poles' (concrete poles with 1-5 horizontal bars + "
                "white insulators), green N-road/R-road signs. "
                "If you see yellow outer road lines + left-hand traffic = South Africa, NOT Spain/Europe/Australia. "
                "Australia has white outer lines. No European country drives on the left with yellow outer lines.\n"
                "REGIONAL COORDS: Cape Town (-33.9,18.4), Garden Route (-33.5,22.0), "
                "Karoo dry plateau (-32,24), Joburg/Gauteng (-26.2,28.0), Durban/KZN (-29.9,31.0), "
                "Limpopo bush (-23.9,29.5)."
            )
        })
        _p("  [south africa] injecting identification constraint")

    if feature_on("region_prefilter") and len(similar) >= 2:
        rag_conts = [c for c in (continent_of(c) for c in similar[:3]) if c]
        if rag_conts:
            from collections import Counter as _CCounter
            cont, cnt = _CCounter(rag_conts).most_common(1)[0]
            if cnt >= 2 or len(rag_conts) == 1:
                extra.insert(0, {"role": "user", "content": (
                    f"STRONG HINT (not absolute): {cnt}/{len(rag_conts)} visually "
                    f"similar reference images are from {cont} "
                    f"({', '.join(similar[:3])}). Lean towards {cont} "
                    f"UNLESS you see specific text, plate format, alphabet, or unique landmark "
                    f"evidence that points elsewhere. RAG is visual-only and can be wrong on "
                    f"biome-similar regions across continents."
                )})
                _p(f"  [region] pre-filter: {cont} {cnt}/{len(rag_conts)} ({', '.join(similar[:3])})")

    _pairs = confusion_pairs()
    if feature_on("confusion_pairs") and _pairs and similar:
        suspected = {c.lower() for c in similar[:3]}
        relevant = [(g, a, c) for g, a, c in _pairs
                    if g.lower() in suspected or a.lower() in suspected]
        if relevant:
            lines = "\n".join(f"  - '{g}' was guessed {c}x but was actually '{a}'"
                              for g, a, c in relevant[:5])
            extra.append({
                "role": "user",
                "content": (
                    f"HISTORICAL CONFUSION PAIRS (your own past errors for suspected countries):\n"
                    f"{lines}\n"
                    f"If you are about to guess one of these 'wrong' countries, pause and "
                    f"double-check the specific clues that distinguish it from the actual one."
                ),
            })
            _p(f"  [confusion] {len(relevant)} relevant pair(s): "
               + ", ".join(f"{g}→{a}" for g, a, _ in relevant[:3]))
    return extra


def _detect_issues(guess: Guess, sig: _Signals) -> list[tuple[str, str]]:
    """Deterministic checks on a model answer. Returns [(kind, instruction)]."""
    issues: list[tuple[str, str]] = []

    if sig.ocr_literal_country and guess.country:
        if _norm_country(sig.ocr_literal_country) != _norm_country(guess.country):
            issues.append(("ocr_literal", (
                f"The OCR text contains a verbatim mention of {sig.ocr_literal_country}, but you "
                f"answered '{guess.country}'. Unless that word is clearly not a place reference, "
                f"set country='{sig.ocr_literal_country}' and put the coordinates inside it."
            )))

    if guess.continent and guess.country:
        declared = guess.continent.strip().lower()
        declared = {"central america": "north america", "americas": "north america"}.get(declared, declared)
        actual_cont = (continent_of(guess.country) or "").lower()
        if actual_cont and declared and actual_cont != declared:
            issues.append(("continent_mismatch", (
                f"You declared continent='{guess.continent}' but '{guess.country}' is in "
                f"{continent_of(guess.country)}. Either change the country to one actually in "
                f"{guess.continent}, or change the continent to match {guess.country}."
            )))

    similar = sig.rag_countries
    if feature_on("correction_rag_continent") and len(similar) >= 2 and guess.country:
        rag_conts = [c for c in (continent_of(c) for c in similar[:3]) if c]
        guess_cont = continent_of(guess.country)
        if rag_conts and guess_cont and all(c == rag_conts[0] for c in rag_conts) and guess_cont != rag_conts[0]:
            issues.append(("rag_continent", (
                f"All {len(rag_conts)} visually similar reference rounds are in {rag_conts[0]} "
                f"({', '.join(similar[:3])}), but you placed the pin in {guess.country} ({guess_cont}). "
                f"Unless you have CONCRETE evidence (OCR text, alphabet, landmark, plate format) that "
                f"rules out every {rag_conts[0]} country, reconsider one of: {', '.join(similar[:3])}."
            )))

    if feature_on("correction_cand_rag") and guess.candidates and sig.rag_scores and guess.country:
        ranked = sorted(sig.rag_scores.items(), key=lambda kv: kv[1], reverse=True)
        rag_top, rag_top_score = ranked[0]
        rag_second = ranked[1][1] if len(ranked) > 1 else 0.0
        primary_n = _norm_country(guess.country)
        rag_top_n = _norm_country(rag_top)
        cand_map = {_norm_country(c): (c, conf) for c, conf in guess.candidates}
        if (rag_top_n and primary_n != rag_top_n and rag_top_n in cand_map
                and rag_top_score >= 1.5 * max(rag_second, 0.01)):
            alt_country, alt_conf = cand_map[rag_top_n]
            primary_conf = cand_map.get(primary_n, (guess.country, guess.confidence))[1]
            if primary_conf < 0.60 and alt_conf >= 0.30:
                issues.append(("cand_rag", (
                    f"Your own candidates list includes {alt_country} (conf {alt_conf:.2f}) and the "
                    f"weighted visual match points to {alt_country} with score {rag_top_score:.2f} "
                    f"(vs next {rag_second:.2f}). This convergent evidence is stronger than your "
                    f"primary {guess.country} (conf {primary_conf:.2f}). Switch to {alt_country} unless "
                    f"you have concrete visual evidence (OCR text, plate code, named landmark) against it."
                )))

    try:
        c_ok, r_ok, on_land, coord_country, coord_admin1 = validate_guess_coords(guess)
    except Exception as e:
        _p(f"  validation skipped: {e}")
        return issues
    hint = region_coord_hint(guess.country, guess.region)
    bbox = country_bbox_hint(guess.country)
    hint_line = f" {hint} Use coordinates close to that reference." if hint else ""
    bbox_line = f" {bbox}" if bbox else ""
    if not on_land:
        issues.append(("ocean", (
            f"Your coordinates ({guess.latitude:.3f}, {guess.longitude:.3f}) are in open water. "
            f"Move them INLAND inside {guess.country or 'the country you identified'}.{hint_line}{bbox_line} "
            f"Street View is only on land."
        )))
    elif not c_ok:
        issues.append(("country_coords", (
            f"You said country='{guess.country}' but the coordinates ({guess.latitude:.3f}, "
            f"{guess.longitude:.3f}) are in {coord_country}. If the image is in {guess.country}, "
            f"fix the coordinates to be inside {guess.country}.{hint_line}{bbox_line} If the image is "
            f"actually in {coord_country}, change the country field instead."
        )))
    elif not r_ok:
        _p(f"  region note: said {guess.region} but coords → {coord_admin1} (accepted, country OK)")
    return issues


# Objectively broken answers, as opposed to advisory disagreements: a pin in the sea,
# coordinates in a different country than the one named, a continent that contradicts
# the country, or OCR text naming a different country outright.
_HARD_ISSUES = {"ocean", "country_coords", "continent_mismatch", "ocr_literal"}


def _hard_issue_count(kinds: list[str]) -> int:
    return sum(1 for k in kinds if k in _HARD_ISSUES)


def _guess_json(g: Guess) -> str:
    return json.dumps({
        "continent": g.continent, "country": g.country, "region": g.region,
        "latitude": round(g.latitude, 4), "longitude": round(g.longitude, 4),
        "confidence": round(g.confidence, 2), "reasoning": g.reasoning,
        "candidates": [{"country": c, "confidence": p} for c, p in g.candidates],
    }, ensure_ascii=False)


def _is_null_island(g: Guess) -> bool:
    return abs(g.latitude) < 0.5 and abs(g.longitude) < 0.5


def _clamp_to_country(guess: Guess) -> float | None:
    """Snap coords into the declared country's polygon. Returns km moved, or None."""
    if not (feature_on("clamp") and guess.country and WORLD_BORDERS):
        return None
    target = _norm_country(guess.country)
    geom = _COUNTRY_POLYGONS.get(target)
    if geom is None:
        return None
    pt = Point(guess.longitude, guess.latitude)
    if geom.contains(pt):
        return None
    try:
        pol_pt, _ = nearest_points(geom, pt)
        cent = geom.representative_point()
        alpha = 0.20  # pull inland so projection drift (~100 km) can't spill over the border
        lat_safe = pol_pt.y + (cent.y - pol_pt.y) * alpha
        lon_safe = pol_pt.x + (cent.x - pol_pt.x) * alpha
        d_km = haversine_km(guess.latitude, guess.longitude, lat_safe, lon_safe)
        _p(f"  [clamp] {guess.country}: ({guess.latitude:.2f},{guess.longitude:.2f}) → ({lat_safe:.2f},{lon_safe:.2f}) Δ{d_km:.0f}km")
        guess.latitude, guess.longitude = lat_safe, lon_safe
        return d_km
    except Exception as e:
        _p(f"  [clamp error] {e}")
        return None


def _rescue_from_water(guess: Guess, exclude_ids: set[str] | None = None) -> float | None:
    """Last deterministic guard: a pin still in open water after the review call and
    the polygon clamp is moved to the nearest known land in the declared country.
    Costs no model call. Returns km moved, or None when the pin was already fine."""
    try:
        if distance_to_nearest_land_km(guess.latitude, guess.longitude) <= 80.0:
            return None
    except Exception:
        return None
    target = _norm_country(guess.country)
    pts = [(la, lo) for la, lo, stem in _HISTORY_ACTUALS.get(target, [])
           if not exclude_ids or stem not in exclude_ids]
    if pts:
        lat, lon = min(pts, key=lambda p: haversine_km(guess.latitude, guess.longitude, p[0], p[1]))
    elif target in _COUNTRY_CENTROIDS:
        lat, lon = _COUNTRY_CENTROIDS[target]
    else:
        return None
    d_km = haversine_km(guess.latitude, guess.longitude, lat, lon)
    _p(f"  [water-rescue] {guess.country}: ({guess.latitude:.2f},{guess.longitude:.2f}) "
       f"was in open water → ({lat:.2f},{lon:.2f}) Δ{d_km:.0f}km")
    guess.latitude, guess.longitude = lat, lon
    return d_km


def _print_guess(label: str, g: Guess) -> None:
    col = _C.GREEN if g.confidence >= 0.80 else (_C.YELLOW if g.confidence >= 0.55 else _C.RED)
    _p(f"  {_C.GREY}{_ts()}{_C.RESET} {_C.BOLD}{label:<9}{_C.RESET} {_C.CYAN}{g.country}{_C.RESET} / {g.region or '-'}  ({g.latitude:.2f}, {g.longitude:.2f})  {col}conf {g.confidence:.2f}{_C.RESET}")
    if g.reasoning:
        _p(f"  {_C.GREY}{_ts()}{_C.RESET} {_C.DIM}think     {g.reasoning}{_C.RESET}")
    if g.candidates:
        _p(f"  {_C.GREY}{_ts()}{_C.RESET} {_C.DIM}cands     " + ", ".join(f"{c}({p:.2f})" for c, p in g.candidates[:3]) + _C.RESET)


async def decide_guess(
    png: bytes,
    shot_path: Path,
    model_used: str,
    *,
    recent_wrong: list[str] | None = None,
    exclude_rag_ids: set[str] | None = None,
) -> Decision:
    """Turn a Street View screenshot into a final pin. No browser access."""
    t_start = time.time()
    timings: dict[str, float] = {}
    pil_img = Image.open(BytesIO(png)).convert("RGB")

    sig = await _gather_signals(png, shot_path, exclude_rag_ids)
    timings["signals"] = time.time() - t_start
    _p(f"  [parallel] OCR+compass+RAG concluídos em {timings['signals']:.1f}s")

    extra = _build_context(sig, pil_img, recent_wrong or [])
    compass_bytes = sig.compass_bytes if feature_on("compass") else None

    # --- call #1 ---------------------------------------------------------
    t0 = time.time()
    calls = 0
    guess: Guess | None = None
    deadline = t0 + (MODEL_RETRY_BUDGET_S if IS_MULTIPLAYER else 0.0)
    for attempt in range(MODEL_ATTEMPTS):
        calls += 1
        try:
            # Last attempt drops the extra context, in case the payload itself is
            # what the endpoint is rejecting.
            use_extra = extra if (extra and attempt < MODEL_ATTEMPTS - 1) else None
            guess = await asyncio.to_thread(
                ask_model, png, use_extra, model_used, None, compass_bytes)
            break
        except Exception as e:
            _p(f"  model error (attempt {attempt + 1}/{MODEL_ATTEMPTS}): {e}")
            if attempt == MODEL_ATTEMPTS - 1:
                break
            # Back off before retrying. Retrying instantly turns a rate limit into
            # three failures in a few milliseconds and then a fallback pin, which
            # scores nothing. In duels the whole round is on a clock, so the
            # backoff is capped by what time is left.
            delay = _retry_delay(attempt, e)
            if IS_MULTIPLAYER:
                delay = min(delay, max(deadline - time.time(), 0.0))
                if delay <= 0.0:
                    _p("  retry budget spent — not waiting again")
                    continue
            _p(f"  waiting {delay:.1f}s before retry")
            await asyncio.sleep(delay)
    timings["call1"] = time.time() - t0

    fallback = False
    if guess is None:
        fallback = True
        fb_country = sig.rag_top_country or (sig.rag_countries[0] if sig.rag_countries else None)
        fb_lat, fb_lon = 48.85, 2.35  # Paris: anything beats (0,0)
        if fb_country:
            c_n = _norm_country(fb_country)
            if c_n in _COUNTRY_CENTROIDS:
                fb_lat, fb_lon = _COUNTRY_CENTROIDS[c_n]
        _p(f"  {_C.RED}✗ all model attempts failed — fallback to {fb_country or 'Paris'} ({fb_lat:.2f}, {fb_lon:.2f}){_C.RESET}")
        guess = Guess(country=fb_country or "?", region="?", latitude=fb_lat, longitude=fb_lon,
                      confidence=0.0, reasoning="fallback")

    initial = Guess(**guess.__dict__)
    _print_guess("GUESS", guess)

    # --- issues + call #2 -------------------------------------------------
    issue_list = [] if fallback else _detect_issues(guess, sig)
    issues = [k for k, _ in issue_list]
    corrected: Guess | None = None
    if issue_list and feature_on("correction_call"):
        for k, txt in issue_list:
            _p(f"  ⚠ {k}: {txt[:110]}{'…' if len(txt) > 110 else ''}")
        numbered = "\n".join(f"{i}. {t}" for i, (_, t) in enumerate(issue_list, 1))
        review = {
            "role": "user",
            "content": (
                f"REVIEW YOUR PREVIOUS ANSWER. It was:\n{_guess_json(guess)}\n\n"
                f"Problems detected by automatic checks:\n{numbered}\n\n"
                f"Fix every problem above and return only the corrected JSON. Keep your country "
                f"unless a problem says otherwise, and always place the coordinates inside the "
                f"country you name. Western longitudes are NEGATIVE (Ireland≈-6°, Portugal≈-8°, "
                f"US East≈-70°, Brazil≈-50°); eastern longitudes are POSITIVE (Poland≈+18°, "
                f"India≈+77°, Japan≈+138°). Southern latitudes are NEGATIVE."
            ),
        }
        t0 = time.time()
        calls += 1
        try:
            guess2 = await asyncio.to_thread(
                ask_model, png, (extra or []) + [review], model_used, None, compass_bytes)
            if not guess2 or _is_null_island(guess2):
                _p("  correction: rejected (lat/lon ≈ 0,0) — keeping previous")
            else:
                # A review must not make the answer objectively worse: if it introduces
                # a hard problem the original did not have, keep the original.
                hard_before = _hard_issue_count(issues)
                hard_after = _hard_issue_count([k for k, _ in _detect_issues(guess2, sig)])
                if hard_after == 0 or hard_after < hard_before:
                    guess = guess2
                    corrected = Guess(**guess2.__dict__)
                    _print_guess("REVISED", guess)
                else:
                    _p(f"  correction: rejected — would leave {hard_after} hard issue(s) "
                       f"vs {hard_before} before; keeping previous")
        except Exception as e:
            _p(f"  correction failed: {e} — keeping previous")
        timings["call2"] = time.time() - t0

    # --- deterministic clamp ---------------------------------------------
    clamped_km = None if fallback else _clamp_to_country(guess)
    if not fallback:
        clamped_km = _rescue_from_water(guess, exclude_rag_ids) or clamped_km

    # --- calibration ------------------------------------------------------
    conf_raw = guess.confidence
    conf_cal = calibrated_confidence(model_used, conf_raw, exclude_rag_ids) if feature_on("calibration") else None
    if conf_cal is not None:
        _p(f"  [calib] confidence {conf_raw:.2f} → historical hit-rate {conf_cal:.2f}")

    # --- hedge ------------------------------------------------------------
    hedge = None
    if feature_on("hedge") and not fallback:
        p_primary = conf_cal if conf_cal is not None else conf_raw
        hedge = hedge_guess(guess, p_primary, model_used, exclude_rag_ids)
        if hedge:
            _p(f"  [hedge] p={hedge['p_primary']:.2f} → moving {hedge['moved_km']}km toward "
               f"{hedge['toward']} (E[score] {hedge['expected_before']} → {hedge['expected_after']})")
            guess.latitude, guess.longitude = hedge["to"]

    timings["total"] = time.time() - t_start
    return Decision(
        guess=guess, initial=initial, corrected=corrected, model_calls=calls,
        issues=issues, clamped_km=clamped_km, confidence_raw=conf_raw,
        confidence_calibrated=conf_cal, hedge=hedge, ocr_text=sig.ocr_text,
        compass=sig.compass, rag_countries=list(sig.rag_countries),
        rag_scores=dict(sig.rag_scores), fallback=fallback, timings=timings,
    )
