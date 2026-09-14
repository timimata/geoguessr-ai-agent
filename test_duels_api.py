"""test_duels_api.py - offline checks for the duels result path.

The duels reader cannot be exercised without a live match, so a fake Playwright
page serves canned game-server payloads. Covers round selection, our own guess
versus a stale one, the solo/duels dispatch, polling, and deferred scoring.

Run: python test_duels_api.py
"""
import asyncio
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ["MODE"] = "duels"
os.environ["FEATURES_OFF"] = "ocr"  # skip the slow model load
sys.path.insert(0, str(Path(__file__).resolve().parent))
import browser
import config
import storage
import bot


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


DUEL_PAYLOAD = {
    "currentRoundNumber": 3,
    "rounds": [
        {"roundNumber": 1, "panorama": {"lat": 48.85, "lng": 2.35, "countryCode": "fr"}},
        {"roundNumber": 2, "panorama": {"lat": 35.68, "lng": 139.69, "countryCode": "jp"}},
        {"roundNumber": 3, "panorama": {"lat": -33.92, "lng": 18.42, "countryCode": "za"}},
    ],
    "teams": [
        {"players": [{"playerId": "me-123", "guesses": [
            {"roundNumber": 1, "lat": 48.0, "lng": 2.0, "created": now_iso()},
            {"roundNumber": 2, "lat": 35.0, "lng": 139.0, "created": now_iso()},
            {"roundNumber": 3, "lat": -34.0, "lng": 18.5, "created": now_iso()},
        ]}]},
        {"players": [{"playerId": "them-456", "guesses": []}]},
    ],
}


class FakePage:
    """Minimal stand-in: routes page.evaluate by what the JS source mentions."""

    def __init__(self, url, payload=DUEL_PAYLOAD, profile_ok=True, http_error=None):
        self.url = url
        self.payload = payload
        self.profile_ok = profile_ok
        self.http_error = http_error
        self.calls = []

    async def evaluate(self, script, arg=None):
        self.calls.append(script)
        if "profiles" in script:
            return {"user": {"id": "me-123", "nick": "tiago"}} if self.profile_ok else None
        if "game-server" in script:
            if self.http_error:
                return {"__error": self.http_error}
            return self.payload
        if "/api/v3/games/" in script:
            return {"rounds": [{"lat": 10.0, "lng": 20.0}],
                    "player": {"guesses": [{"lat": 10.1, "lng": 20.1}]}}
        return None


async def main():
    ok = True

    def check(label, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {label}: {got}" + ("" if good else f"  (want {want})"))

    print("duels — happy path")
    browser._my_player_id = None
    p = FakePage("https://www.geoguessr.com/duels/abc-123-def")
    r = await browser.read_duel_round_from_api(p, 3)
    check("round_number", r["round_number"], 3)
    check("actual", r["actual"], (-33.92, 18.42))
    check("placed", r["placed"], (-34.0, 18.5))
    check("country_code", r["country_code"], "za")

    print("team-duels URL")
    browser._my_player_id = None
    p = FakePage("https://www.geoguessr.com/team-duels/xyz-999")
    r = await browser.read_duel_round_from_api(p, 3)
    check("resolves", r is not None and r["actual"] == (-33.92, 18.42), True)

    print("profile lookup fails — still returns the true location")
    browser._my_player_id = None
    p = FakePage("https://www.geoguessr.com/duels/abc", profile_ok=False)
    r = await browser.read_duel_round_from_api(p, 2)
    check("actual from our own round counter", r["actual"], (35.68, 139.69))
    check("placed unknown", r["placed"], None)

    print("game-server returns an error")
    browser._my_player_id = None
    p = FakePage("https://www.geoguessr.com/duels/abc", http_error=404)
    check("returns None", await browser.read_duel_round_from_api(p, 1), None)

    print("stale guess from a previous match is ignored")
    browser._my_player_id = None
    stale = {**DUEL_PAYLOAD, "teams": [
        {"players": [{"playerId": "me-123", "guesses": [
            {"roundNumber": 3, "lat": 1.0, "lng": 1.0, "created": "2020-01-01T00:00:00Z"}]}]},
    ]}
    p = FakePage("https://www.geoguessr.com/duels/abc", payload=stale)
    r = await browser.read_duel_round_from_api(p, 2)
    check("uses our own counter, not the advanced server round", r["round_number"], 2)
    check("no placed from stale guess", r["placed"], None)

    print("round hint out of range — server round number wins")
    browser._my_player_id = None
    p = FakePage("https://www.geoguessr.com/duels/abc", payload=stale)
    r = await browser.read_duel_round_from_api(p, 99)
    check("server round", r["round_number"], 3)

    print("dispatch: solo URL uses the v3 games API")
    p = FakePage("https://www.geoguessr.com/game/TOKEN123")
    r = await browser.read_result_from_api(p, 1)
    check("solo actual", r["actual"], (10.0, 20.0))
    check("solo placed", r["placed"], (10.1, 20.1))

    print("polling stops as soon as the guess is visible")
    browser._my_player_id = None
    p = FakePage("https://www.geoguessr.com/duels/abc")
    r = await browser.read_result_from_api(p, 3, attempts=6, delay_s=0.01)
    check("one game-server fetch", sum("game-server" in c for c in p.calls), 1)

    print("deferred scoring patches a queued round")
    browser._my_player_id = None
    import json
    tmp_log = Path(tempfile.gettempdir()) / "geoguessr_test_log.json"
    entry = {"timestamp": "20260914_120000", "round": 0, "actual": None, "error_km": None,
             "country_hit": None, "guess": {"country": "South Africa", "latitude": -34.0,
                                            "longitude": 18.5, "reasoning": "r"}}
    tmp_log.write_text(json.dumps([entry], indent=1), encoding="utf-8")
    real_log = config.LOG_FILE
    config.LOG_FILE = tmp_log
    storage.LOG_FILE = tmp_log
    bot._pending_scores = [{"timestamp": "20260914_120000", "round_num": 3,
                            "screenshot": "does_not_exist.png",
                            "guess": entry["guess"]}]
    await bot.resolve_pending_scores(FakePage("https://www.geoguessr.com/duels/abc"))
    patched = json.loads(tmp_log.read_text(encoding="utf-8"))[0]
    config.LOG_FILE = real_log
    storage.LOG_FILE = real_log
    check("queue drained", bot._pending_scores, [])
    check("actual filled", patched["actual"]["country"], "South Africa")
    check("round number corrected", patched["round"], 3)
    check("country_hit", patched["country_hit"], True)
    check("error under 20km", patched["error_km"] < 20, True)

    print("\nRESULT:", "all passed" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


sys.exit(asyncio.run(main()))
