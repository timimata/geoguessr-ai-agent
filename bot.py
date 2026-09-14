"""GeoGuessr agent entry point: the live round loop and the match loop.

The decision logic lives in pipeline.py so that it can be replayed offline by
benchmark.py; this module is the part that needs a real browser.
"""
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from playwright.async_api import Page, async_playwright
from playwright_stealth import Stealth

from browser import (capture_streetview, click_play_again, human_delay,
                     is_duels_game_over, place_guess, read_result_from_api,
                     wait_for_round)
import browser as _browser
import config
from config import (FEATURES, IS_MULTIPLAYER, MODE, MODEL_ID,
                    MODEL_IDS, PROJECT_DIR, SCREENSHOTS_DIR, USER_DATA_DIR, _C,
                    _ts, feature_on, round_elapsed, start_round_clock)
from geo import haversine_km, lookup_country, _COUNTRY_ALIASES
from llm import Guess
from vision import SCREENSHOT_SUFFIX, save_screenshot
from pipeline import decide_guess, reload_history
from rag import (_load_confusion_pairs, index_round_into_rag,
                 needs_confusion_refresh, warm_up)
import rag as _rag
from storage import (append_round, count_rounds_in_log, load_rounds, log_path,
                     recent_wrong_countries, score_entry, update_log_entry)

# Duel rounds whose result was not readable yet: a round only resolves once every
# player has guessed, which can outlast the post-guess wait. They are retried at
# the start of the next round and at the end of the match.
_pending_scores: list[dict] = []


async def resolve_pending_scores(page: Page) -> None:
    global _pending_scores
    if not _pending_scores:
        return
    still: list[dict] = []
    for pend in _pending_scores:
        data = await read_result_from_api(page, pend["round_num"])
        actual = data.get("actual") if data else None
        if not actual:
            pend["tries"] = pend.get("tries", 0) + 1
            if pend["tries"] < 4:
                still.append(pend)
            else:
                print(f"  {_C.YELLOW}[deferred] giving up on round {pend['round_num']} "
                      f"({pend['timestamp']}) - never resolved{_C.RESET}")
            continue
        entry = {"guess": pend["guess"]}
        score_entry(entry, actual, data.get("placed"))
        patch = {k: entry[k] for k in
                 ("actual", "placed", "error_km", "country_hit", "projection_drift_km")
                 if k in entry}
        if data.get("round_number"):
            patch["round"] = int(data["round_number"])
        if update_log_entry(pend["timestamp"], patch):
            hit = "OK" if entry["country_hit"] else "X"
            print(f"  {_C.GREEN}[deferred]{_C.RESET} round {pend['round_num']} scored: "
                  f"{entry['actual']['country']} - {entry['error_km']:.0f} km {hit}")
            index_round_into_rag(Path(pend["screenshot"]), entry["actual"],
                                 pend["guess"].get("reasoning", ""),
                                 entry["error_km"], entry["country_hit"])
    _pending_scores = still


async def play_round(page: Page, round_num: int) -> None:
    start_round_clock()
    bar = "═" * 64
    print(f"\n{_C.CYAN}{bar}")
    print(f"  ROUND {round_num}  ·  mode={MODE}  ·  model={MODEL_ID}")
    print(f"{bar}{_C.RESET}")
    # A duel round from earlier may only have resolved now that everyone guessed.
    await resolve_pending_scores(page)

    if not await wait_for_round(page):
        print(f"  {_C.RED}✗ timed out waiting for panorama{_C.RESET}")
        return

    png, _ = await capture_streetview(page)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shot_path = save_screenshot(png, SCREENSHOTS_DIR / f"round_{ts}{SCREENSHOT_SUFFIX}")

    # Round-robin model selection across MODEL_IDS using total rounds so far,
    # so A/B tests stay balanced across restarts.
    total_so_far = count_rounds_in_log()
    model_used = MODEL_IDS[total_so_far % len(MODEL_IDS)]
    if len(MODEL_IDS) > 1:
        print(f"  model   : {model_used}")

    recent_wrong = recent_wrong_countries(n=8) if feature_on("blacklist") else []
    decision = await decide_guess(png, shot_path, model_used, recent_wrong=recent_wrong)
    guess = decision.guess

    placed_ok = await place_guess(page, guess)
    if not placed_ok:
        return

    await human_delay(3.0, 5.0)
    api_data = await read_result_from_api(
        page, round_num,
        attempts=6 if IS_MULTIPLAYER else 1,  # duels: wait for the opponent to guess
        delay_s=2.5,
    )
    actual = api_data["actual"] if api_data else None
    placed = api_data.get("placed") if api_data else None
    if api_data and api_data.get("round_number"):
        round_num = int(api_data["round_number"])

    # error_km = distance from where the pin ACTUALLY landed (placed) to actual.
    ref = placed or (guess.latitude, guess.longitude)
    error_km = haversine_km(ref[0], ref[1], *actual) if actual else None
    projection_drift_km = (
        haversine_km(guess.latitude, guess.longitude, placed[0], placed[1]) if placed else None
    )

    actual_entry = None
    country_hit = None
    if actual:
        country, admin1, cc = lookup_country(*actual)
        actual_entry = {
            "latitude": actual[0], "longitude": actual[1],
            "country": country, "admin1": admin1, "cc": cc,
        }
        if placed:
            placed_country, _, _ = lookup_country(*placed)
        else:
            placed_country = guess.country or ""
        placed_country_lc = placed_country.strip().lower()
        country_hit = bool(country) and bool(placed_country_lc) and (
            placed_country_lc == country.lower()
            or country.lower() in placed_country_lc
            or placed_country_lc in country.lower()
        )

    placed_entry = None
    if placed:
        pc, pa, pcc = lookup_country(*placed)
        placed_entry = {"latitude": placed[0], "longitude": placed[1],
                        "country": pc, "admin1": pa, "cc": pcc}

    if guess.country:
        guess.country = _COUNTRY_ALIASES.get(guess.country.lower(), guess.country.title())

    append_round({
        "timestamp": ts,
        "round": round_num,
        "game_mode": f"nmpz-{MODE}",
        "model": model_used,
        "screenshot": str(shot_path.relative_to(PROJECT_DIR)),
        "guess": guess.__dict__,
        "initial_guess": decision.initial.__dict__ if decision.initial else None,
        "placed": placed_entry,
        "actual": actual_entry,
        "error_km": error_km,
        "projection_drift_km": projection_drift_km,
        "country_hit": country_hit,
        "model_calls": decision.model_calls,
        "issues": decision.issues,
        "clamped_km": decision.clamped_km,
        "confidence_calibrated": decision.confidence_calibrated,
        "hedge": decision.hedge,
        "decision_s": round(decision.timings.get("total", 0.0), 1),
        "features_off": sorted(k for k, v in FEATURES.items() if not v),
    })

    current_count = count_rounds_in_log()
    if needs_confusion_refresh(current_count):
        _load_confusion_pairs()
        reload_history()

    if error_km is not None:
        hit_str = "✓" if country_hit else "✗"
        print(f"  actual  : {actual_entry['country']}/{actual_entry['admin1'] or '-'} ({actual[0]:.2f}, {actual[1]:.2f})")
        if placed_entry:
            print(f"  placed  : {placed_entry['country']}/{placed_entry['admin1'] or '-'} ({placed[0]:.2f}, {placed[1]:.2f})")
            if projection_drift_km is not None and projection_drift_km > 50:
                print(f"  ⚠ projection drift: {projection_drift_km:.0f} km between asked and clicked")
        print(f"  result  : {error_km:.0f} km  country {hit_str}")

        index_round_into_rag(shot_path, actual_entry, guess.reasoning, error_km, country_hit)
    elif IS_MULTIPLAYER:
        _pending_scores.append({
            "timestamp": ts, "round_num": round_num,
            "screenshot": str(shot_path), "guess": dict(guess.__dict__),
        })
        print(f"  {_C.YELLOW}result  : duel round not resolved yet — queued for deferred scoring{_C.RESET}")

    # Click 'Next round' / 'View results'.
    for sel in [
        "button[data-qa='close-round-result']",
        "button:has-text('Next')",
        "button:has-text('Play again')",
    ]:
        try:
            btn = await page.query_selector(sel)
        except Exception:
            btn = None
        if btn:
            await human_delay(1.0, 2.0)
            await btn.click()
            break

    _round_dur = round_elapsed()
    if guess.country and guess.country != "?":
        if error_km is not None:
            _hit_icon = f"{_C.GREEN}✓{_C.RESET}" if country_hit else f"{_C.RED}✗{_C.RESET}"
            _err_col = _C.GREEN if error_km < 200 else (_C.YELLOW if error_km < 2000 else _C.RED)
            print(f"  {_C.BOLD}└─ {_hit_icon} {guess.country} → actual {actual_entry['country'] if actual_entry else '?'}  ·  {_err_col}{error_km:.0f} km{_C.RESET}  ·  {_C.DIM}{decision.model_calls} call(s) · {_round_dur:.1f}s total{_C.RESET}")
        else:
            print(f"  {_C.BOLD}└─ {_C.CYAN}{guess.country}{_C.RESET}/{guess.region or '-'} pinned  ·  {_C.DIM}{decision.model_calls} call(s) · {_round_dur:.1f}s total{_C.RESET}")
    else:
        print(f"  {_C.BOLD}└─ {_C.RED}no guess submitted{_C.RESET}  ·  {_C.DIM}{_round_dur:.1f}s total{_C.RESET}")


async def main() -> None:
    _load_confusion_pairs()
    reload_history()
    # Pay the retrieval model's load time now, not while a round is running.
    await asyncio.to_thread(warm_up)
    async with Stealth().use_async(async_playwright()) as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(USER_DATA_DIR),
            channel="chrome",
            headless=False,
            viewport={"width": 1366, "height": 850},
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
            ignore_default_args=["--enable-automation"],
        )

        # Capture ALL google.maps.Map instances (Duels has multiple — small
        # minimap + the big guess map). window.__geoMap keeps the latest for
        # backwards compatibility, but window.__getGuessMap() picks the Map
        # whose div is inside the guess-map container, which is the one the
        # user actually sees and clicks on.
        await context.add_init_script(
            """(() => {
                window.__geoMaps = [];
                window.__geoMap = null;
                window.__geoPano = null;
                window.__getGuessMap = () => {
                    const container = document.querySelector(
                        "[data-qa='guess-map'], .guess-map, div[class*='guess-map']"
                    );
                    if (container) {
                        for (const m of window.__geoMaps) {
                            try {
                                const div = m.getDiv && m.getDiv();
                                if (div && container.contains(div)) return m;
                            } catch (e) { /* ignore */ }
                        }
                    }
                    // Fallback: largest map by current rendered area.
                    let best = null, bestArea = 0;
                    for (const m of window.__geoMaps) {
                        try {
                            const div = m.getDiv && m.getDiv();
                            if (!div) continue;
                            const r = div.getBoundingClientRect();
                            const a = r.width * r.height;
                            if (a > bestArea) { bestArea = a; best = m; }
                        } catch (e) { /* ignore */ }
                    }
                    return best || window.__geoMap;
                };
                const iv = setInterval(() => {
                    if (!window.google || !window.google.maps) return;
                    let mapOk = false, panoOk = false;
                    if (window.google.maps.Map) {
                        const Orig = window.google.maps.Map;
                        try {
                            window.google.maps.Map = new Proxy(Orig, {
                                construct(target, args) {
                                    const inst = Reflect.construct(target, args);
                                    window.__geoMaps.push(inst);
                                    window.__geoMap = inst;
                                    return inst;
                                }
                            });
                            mapOk = true;
                        } catch (e) { /* ignore */ }
                    }
                    if (window.google.maps.StreetViewPanorama) {
                        const OrigP = window.google.maps.StreetViewPanorama;
                        try {
                            window.google.maps.StreetViewPanorama = new Proxy(OrigP, {
                                construct(target, args) {
                                    const inst = Reflect.construct(target, args);
                                    window.__geoPano = inst;
                                    return inst;
                                }
                            });
                            panoOk = true;
                        } catch (e) { /* ignore */ }
                    }
                    if (mapOk && panoOk) clearInterval(iv);
                }, 30);
            })();"""
        )

        page = context.pages[0] if context.pages else await context.new_page()

        if IS_MULTIPLAYER:
            _label = "TEAM DUELS" if MODE == "team-duels" else "DUELS"
            print(f"{_label} MODE — open a private {_label.lower()} lobby with NMPZ, invite friends, start the match.")
            await page.goto("https://www.geoguessr.com/multiplayer", wait_until="domcontentloaded")
            print("Press Enter here once you are in the first round of the FIRST duel…")
            await asyncio.to_thread(input)

            global _last_canvas_sig
            match_count = 0
            while True:
                match_count += 1
                bar = "█" * 64
                print(f"\n{_C.MAGENTA}{bar}")
                print(f"  MATCH {match_count}")
                print(f"{bar}{_C.RESET}")

                round_num = 0
                consecutive_failures = 0
                last_logged_count = count_rounds_in_log()
                match_start_count = last_logged_count
                while True:
                    if await is_duels_game_over(page):
                        print(f"\n  {_C.YELLOW}[match {match_count}] ended after {round_num} rounds (game-over signal){_C.RESET}")
                        break
                    round_num += 1
                    try:
                        await play_round(page, round_num)
                    except Exception as e:
                        print(f"  {_C.RED}round {round_num} failed: {e}{_C.RESET}")

                    current_count = count_rounds_in_log()
                    if current_count == last_logged_count:
                        consecutive_failures += 1
                        print(f"  {_C.YELLOW}[match {match_count}] no new log entry — failure {consecutive_failures}/3{_C.RESET}")
                        if consecutive_failures >= 3:
                            print(f"\n  {_C.YELLOW}[match {match_count}] 3 consecutive rounds without progress — assuming match ended{_C.RESET}")
                            break
                    else:
                        consecutive_failures = 0
                        last_logged_count = current_count
                    await human_delay(1.5, 2.5)

                # The last rounds of the match resolve once the match is over.
                for _ in range(3):
                    if not _pending_scores:
                        break
                    await resolve_pending_scores(page)
                    if _pending_scores:
                        await asyncio.sleep(3.0)

                # If THIS match logged zero rounds, no new match likely starting — exit.
                rounds_logged_this_match = count_rounds_in_log() - match_start_count
                if rounds_logged_this_match == 0:
                    print(f"\n  {_C.RED}[duels] no rounds played in match {match_count} — exiting{_C.RESET}")
                    break

                # Try to click "CONTINUE" / "Back to lobby" to advance to the next match.
                clicked_continue = False
                for sel in [
                    "button:has-text('CONTINUE')",
                    "button:has-text('Continue')",
                    "button:has-text('Find a new game')",
                    "button:has-text('Find new game')",
                    "button:has-text('Play again')",
                    "button:has-text('Back to lobby')",
                ]:
                    try:
                        btn = await page.query_selector(sel)
                        if btn and await btn.is_visible():
                            await btn.click()
                            print(f"  {_C.GREEN}[match] clicked '{sel}'{_C.RESET}")
                            clicked_continue = True
                            break
                    except Exception:
                        continue
                if not clicked_continue:
                    print(f"  {_C.YELLOW}[match] no continue button found — you may need to start the next match manually{_C.RESET}")

                # Loading screen buffer.
                print(f"  {_C.DIM}[match] waiting 5s for loading screen of next match…{_C.RESET}")
                await asyncio.sleep(5.0)

                # Reset canvas signature so the new match's first round is treated as fresh.
                _last_canvas_sig = None
                print(f"  {_C.CYAN}[match] resuming — ready for next match{_C.RESET}")

            for _ in range(3):
                if not _pending_scores:
                    break
                await resolve_pending_scores(page)
                if _pending_scores:
                    await asyncio.sleep(3.0)

            print("\nAll duels finished. Press Enter to close, or Ctrl+C.")
            try:
                await asyncio.to_thread(input)
            except (EOFError, KeyboardInterrupt):
                pass
            await context.close()
            return

        print("Opening GeoGuessr World map — pick NMPZ and press Play.")
        await page.goto("https://www.geoguessr.com/maps/world", wait_until="domcontentloaded")
        print("Press Enter here once you are in the first round…")
        await asyncio.to_thread(input)

        target_rounds = 3000
        total = count_rounds_in_log()
        print(f"\nStarting autonomous mode — target {target_rounds} total rounds (have {total}).")

        while total < target_rounds:
            for i in range(1, 6):
                if total >= target_rounds:
                    break
                try:
                    await play_round(page, i)
                    total = count_rounds_in_log()
                    print(f"  [progress] {total}/{target_rounds}")
                except Exception as e:
                    print(f"  round {i} failed: {e}")
                await human_delay(2.0, 4.0)

            if total >= target_rounds:
                break

            print("\n[game over] clicking Play Again…")
            try:
                import stats as _stats
                all_rounds = load_rounds()
                leaderboard = _stats.top_games(all_rounds, 10)
                if leaderboard:
                    print("\n🏆 Top 10 games (GeoGuessr score / 25000):")
                    print(f"  {'#':<3}{'Game':<6}{'Score':<8}{'Per-round':<30}Countries")
                    for rank, tg in enumerate(leaderboard, 1):
                        per = ",".join(str(p) for p in tg["per_round"])
                        cs = ", ".join(tg["countries"])[:40]
                        print(f"  {rank:<3}#{tg['game_idx']:<5}{tg['total']:<8}{per:<30}{cs}")
            except Exception as e:
                print(f"  [leaderboard error] {e}")
            if not await click_play_again(page):
                print("  could not find Play Again — stopping")
                break
            await human_delay(3.0, 5.0)

        print(f"\nDone. {total} rounds logged in {log_path().name}")
        await asyncio.to_thread(input, "Press Enter to close…")
        await context.close()


if __name__ == "__main__":
    asyncio.run(main())
