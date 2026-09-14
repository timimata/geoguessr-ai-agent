"""Everything that touches the Playwright page: capturing the panorama, placing
the pin, detecting round boundaries and reading results back from GeoGuessr."""
import asyncio
import hashlib
import math
import random
import re
import time
from datetime import datetime

from playwright.async_api import Page

from config import IS_MULTIPLAYER, MODE, _C, say
from llm import Guess

async def human_delay(a: float = 0.4, b: float = 1.2) -> None:
    await asyncio.sleep(random.uniform(a, b))


async def human_mouse_move(page: Page, x: float, y: float, steps: int = 25) -> None:
    await page.mouse.move(x, y, steps=steps)


async def capture_streetview(page: Page) -> tuple[bytes, dict]:
    # GeoGuessr uses a canvas for Street View; try to screenshot just that canvas.
    viewport = page.viewport_size or {"width": 1366, "height": 850}
    selectors = [
        ".widget-scene-canvas",
        "canvas.mapsConsumerUiSceneCoreScene__canvas",
        "div.game-layout__panorama",
        "div[class*='panorama']",
        "canvas",
    ]
    for sel in selectors:
        for el in await page.query_selector_all(sel):
            box = await el.bounding_box()
            if not box:
                continue
            # Clamp to viewport and ensure the element is actually visible.
            x = max(0.0, box["x"])
            y = max(0.0, box["y"])
            w = min(box["width"], viewport["width"] - x)
            h = min(box["height"], viewport["height"] - y)
            if w < 300 or h < 200:
                continue
            clip = {"x": x, "y": y, "width": w, "height": h}
            try:
                png = await page.screenshot(clip=clip, type="png")
                return png, clip
            except Exception:
                continue
    # Fallback: viewport screenshot.
    png = await page.screenshot(type="png", full_page=False)
    return png, {"x": 0, "y": 0, "width": viewport["width"], "height": viewport["height"]}


async def find_minimap(page: Page, debug: bool = False) -> dict | None:
    # The GeoGuessr minimap is a Google Maps inside a container. We need the
    # on-screen rect of the actual map tiles, not the container (which may
    # include title/padding) nor stray overlay canvases.
    info = await page.evaluate(
        """() => {
            const container = document.querySelector("[data-qa='guess-map'], .guess-map, div[class*='guess-map']");
            if (!container) return null;
            const cRect = container.getBoundingClientRect();
            const canvases = [...container.querySelectorAll('canvas')].map(c => {
                const r = c.getBoundingClientRect();
                return {x: r.x, y: r.y, w: r.width, h: r.height, cls: c.className || ''};
            });
            // Google Maps tiles container (gm-style).
            const tileDiv = container.querySelector(".gm-style > div:first-child") || container.querySelector(".gm-style");
            let tile = null;
            if (tileDiv) {
                const r = tileDiv.getBoundingClientRect();
                tile = {x: r.x, y: r.y, w: r.width, h: r.height};
            }
            return {container: {x: cRect.x, y: cRect.y, w: cRect.width, h: cRect.height}, canvases, tile};
        }"""
    )
    if debug:
        print(f"  [minimap debug] {info}")
    if not info:
        return None
    # Prefer the .gm-style tile div (actual map area).
    if info.get("tile") and info["tile"]["w"] > 100 and info["tile"]["h"] > 100:
        t = info["tile"]
        return {"x": t["x"], "y": t["y"], "width": t["w"], "height": t["h"]}
    # Else biggest canvas inside container.
    canvases = [c for c in info["canvases"] if c["w"] > 100 and c["h"] > 100]
    if canvases:
        c = max(canvases, key=lambda c: c["w"] * c["h"])
        return {"x": c["x"], "y": c["y"], "width": c["w"], "height": c["h"]}
    # Fallback: container.
    c = info["container"]
    return {"x": c["x"], "y": c["y"], "width": c["w"], "height": c["h"]}


def latlon_to_minimap_xy_bounds(
    lat: float, lon: float, box: dict, bounds: dict
) -> tuple[float, float]:
    # Web Mercator within known geographic bounds of the map viewport.
    def merc_y(lat_):
        lat_c = max(-85.0511, min(85.0511, lat_))
        s = math.sin(math.radians(lat_c))
        return 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)

    north, south = bounds["north"], bounds["south"]
    west, east = bounds["west"], bounds["east"]
    my_top, my_bot = merc_y(north), merc_y(south)
    my = merc_y(lat)
    # Longitude may wrap the antimeridian; normalize to the same side as west.
    lon_n = lon
    if east < west:  # viewport crosses 180°
        if lon_n < west:
            lon_n += 360.0
        east_n = east + 360.0
        x_frac = (lon_n - west) / (east_n - west)
    else:
        x_frac = (lon_n - west) / (east - west)
    y_frac = (my - my_top) / (my_bot - my_top)
    x = box["x"] + max(0.0, min(1.0, x_frac)) * box["width"]
    y = box["y"] + max(0.0, min(1.0, y_frac)) * box["height"]
    return x, y


def latlon_to_world_mercator(lat: float, lon: float, box: dict) -> tuple[float, float]:
    """Direct Web Mercator projection assuming the box shows the full world
    (zoom ~1). Reliable for the Duels guess map which defaults to world view
    and doesn't require Google Maps internal state."""
    lat_c = max(-85.0511, min(85.0511, lat))
    s = math.sin(math.radians(lat_c))
    merc_y = 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)
    x_frac = (lon + 180) / 360
    x = box["x"] + max(0.0, min(1.0, x_frac)) * box["width"]
    y = box["y"] + max(0.0, min(1.0, merc_y)) * box["height"]
    return x, y


def latlon_to_minimap_xy(lat: float, lon: float, box: dict) -> tuple[float, float]:
    # If the box looks like a full-screen guess map (wide enough), use direct
    # Web Mercator projection — works when Google Maps internal state isn't
    # available (e.g., Duels where __geoMap isn't reliably captured).
    if box["width"] > 600:
        return latlon_to_world_mercator(lat, lon, box)
    # Otherwise empirical calibration tuned for small corner minimaps.
    LON_MIN, LON_MAX = -124.34, 156.80
    MERC_Y_MIN, MERC_Y_MAX = 0.2163, 0.6713
    lat_clamped = max(-85.0511, min(85.0511, lat))
    sin_lat = math.sin(math.radians(lat_clamped))
    merc_y = 0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)
    x_frac = (lon - LON_MIN) / (LON_MAX - LON_MIN)
    y_frac = (merc_y - MERC_Y_MIN) / (MERC_Y_MAX - MERC_Y_MIN)
    x = box["x"] + max(0.0, min(1.0, x_frac)) * box["width"]
    y = box["y"] + max(0.0, min(1.0, y_frac)) * box["height"]
    return x, y


async def read_map_bounds(page: Page) -> dict | None:
    # Uses the actually-visible guess map (in Duels there are multiple Map
    # instances on the page; we want the one inside the guess-map container).
    try:
        data = await page.evaluate(
            """() => {
                const map = (typeof window.__getGuessMap === 'function')
                    ? window.__getGuessMap()
                    : window.__geoMap;
                if (!map || typeof map.getBounds !== 'function') return null;
                const b = map.getBounds();
                if (!b) return null;
                const ne = b.getNorthEast(), sw = b.getSouthWest();
                return {
                    north: ne.lat(), south: sw.lat(),
                    east: ne.lng(), west: sw.lng(),
                };
            }"""
        )
        return data
    except Exception:
        return None


async def query_first(page: Page, selectors: list[str], label: str = ""):
    """First element matching any of `selectors`. A malformed or unsupported
    selector is skipped with a warning instead of aborting the whole round."""
    for sel in selectors:
        try:
            el = await page.query_selector(sel)
        except Exception as e:
            print(f"  [{label or 'selector'}] {sel!r} failed: {e}")
            continue
        if el:
            return el, sel
    return None, ""


async def expand_minimap(page: Page) -> None:
    # Click the maximize/expand button (arrow icon) to make the minimap full-screen.
    _sels = [
        "button[data-qa='guess-map-expand']",
        "button[aria-label*='Expand' i]",
        "button[aria-label*='fullscreen' i]",
        "button[class*='expand']",
        "[class*='guess-map'] button[class*='arrow']",
    ]
    btn, _ = await query_first(page, _sels, "expand-minimap")
    if btn:
        try:
            await btn.click()
            await human_delay(0.4, 0.8)
        except Exception:
            pass


async def place_guess(page: Page, guess: Guess) -> bool:
    minimap = await find_minimap(page)
    if not minimap:
        # Sometimes the minimap appears with a small delay (post-render or
        # transition state). Retry a few times before giving up.
        for attempt in range(8):
            await asyncio.sleep(0.5)
            minimap = await find_minimap(page)
            if minimap:
                print(f"  minimap found after {attempt+1} retries")
                break
    if not minimap:
        print("  minimap not found after retries — skipping click")
        return False

    # Hover minimap so GeoGuessr expands it to its interactive (bigger) size.
    hover_x = minimap["x"] + minimap["width"] / 2
    hover_y = minimap["y"] + minimap["height"] / 2
    await human_mouse_move(page, hover_x, hover_y, steps=20)
    await human_delay(0.3, 0.5)
    # Re-measure after hover-expand.
    minimap = await find_minimap(page, debug=True) or minimap

    # TRUQUE DE MESTRE: centrar/fazer zoom programaticamente. Tenta zoom 10 (preciso);
    # se as bounds não contiverem o alvo (pan falhou ou o minimap é global), reduz
    # o zoom para 7 e depois 5. Evita drifts catastróficos (>1000 km).
    async def _try_center(zoom: int) -> dict | None:
        try:
            await page.evaluate(
                """([lat, lng, z]) => {
                    const map = (typeof window.__getGuessMap === 'function')
                        ? window.__getGuessMap()
                        : window.__geoMap;
                    if (map && typeof map.setCenter === 'function') {
                        map.setZoom(z);
                        map.setCenter({lat: lat, lng: lng});
                    }
                }""",
                [guess.latitude, guess.longitude, zoom]
            )
            await human_delay(0.2, 0.4)
            return await read_map_bounds(page)
        except Exception as e:
            print(f"  [map debug] pan@z{zoom} failed: {e}")
            return None

    # In duels, the guess map can be at any zoom/center (e.g., remembered
    # from last round, or default not exactly world view). Reset it to a
    # known world view first via mouse-wheel zoom-out (15 steps is enough
    # for any starting zoom), then do Mercator click + zoom-in for precision.
    if IS_MULTIPLAYER and minimap["width"] > 600:
        center_x = minimap["x"] + minimap["width"] / 2
        center_y = minimap["y"] + minimap["height"] / 2
        # 1. Zoom out to world view from wherever the map currently is.
        await page.mouse.move(center_x, center_y)
        await human_delay(0.1, 0.2)
        print(f"  [map] duels — resetting to world view (zoom out)")
        for _ in range(15):
            try:
                await page.mouse.wheel(0, 240)  # positive = zoom out
            except Exception:
                break
            await asyncio.sleep(0.08)
        await human_delay(0.4, 0.6)
        # 2. Compute target pixel assuming world Mercator (-180..180, ±85).
        x, y = latlon_to_world_mercator(guess.latitude, guess.longitude, minimap)
        print(f"  [map] Mercator target ({x:.0f}, {y:.0f}) for ({guess.latitude:.2f}, {guess.longitude:.2f})")
        # 3. Move cursor to target so subsequent zoom anchors on it.
        await page.mouse.move(x, y)
        await human_delay(0.25, 0.4)
        # 4. Zoom in for precision (cursor stays over same lat/lon while
        #    Google Maps zooms centred on cursor).
        ZOOM_STEPS = 7
        for i in range(ZOOM_STEPS):
            try:
                await page.mouse.wheel(0, -240)  # negative = zoom in
            except Exception as e:
                print(f"  [map] zoom wheel step {i+1} failed: {e}")
                break
            await asyncio.sleep(0.18)
        await human_delay(0.35, 0.55)
        print(f"  [map] zoomed in — clicking at ({x:.0f}, {y:.0f})")
    else:
        bounds = None
        for z in (10, 7, 5):
            b = await _try_center(z)
            if not b:
                continue
            in_lat = b["south"] <= guess.latitude <= b["north"]
            lon_ok = (b["west"] <= guess.longitude <= b["east"]) if b["east"] >= b["west"] else (
                guess.longitude >= b["west"] or guess.longitude <= b["east"]
            )
            if in_lat and lon_ok:
                bounds = b
                if z != 10:
                    print(f"  [map] target outside zoom-10 viewport; using zoom {z}")
                break
        if bounds:
            x, y = latlon_to_minimap_xy_bounds(guess.latitude, guess.longitude, minimap, bounds)
        else:
            print("  map bounds: unavailable — using empirical fallback")
            x, y = latlon_to_minimap_xy(guess.latitude, guess.longitude, minimap)
    x = max(minimap["x"] + 2, min(minimap["x"] + minimap["width"] - 2, x))
    y = max(minimap["y"] + 2, min(minimap["y"] + minimap["height"] - 2, y))

    await human_mouse_move(page, x, y, steps=30)
    await human_delay(0.1, 0.2)
    await page.mouse.click(x, y)
    await human_delay(0.2, 0.4)

    # Confirm guess button. GeoGuessr ships hashed CSS module class names
    # (button_variantPrimary__aB3xY), so match the stable prefix with [class*=]
    # rather than a wildcard, which is not valid CSS and makes Playwright throw.
    for sel in [
        "button[data-qa='perform-guess']",
        "button[class*='button_variantPrimary']",
        "button:has-text('Guess')",
        "button:has-text('Place your pin')",
    ]:
        try:
            btn = await page.query_selector(sel)
        except Exception as e:
            print(f"  [guess-button] selector {sel!r} failed: {e}")
            continue
        if btn:
            try:
                await btn.click()
                return True
            except Exception as e:
                print(f"  [guess-button] click on {sel!r} failed: {e}")
    await page.keyboard.press("Space")
    return True


async def read_round_from_api(page: Page, round_index: int) -> dict | None:
    # Returns {actual: (lat,lng), placed: (lat,lng) or None} from GeoGuessr's API.
    url = page.url
    m = re.search(r"/(?:game|results)/([A-Za-z0-9]+)", url)
    if not m:
        return None
    token = m.group(1)
    try:
        data = await page.evaluate(
            """async (token) => {
                const r = await fetch(`/api/v3/games/${token}`, {credentials: 'include'});
                if (!r.ok) return null;
                return await r.json();
            }""",
            token,
        )
    except Exception:
        return None
    if not data or "rounds" not in data:
        return None
    rounds = data["rounds"]
    if round_index - 1 >= len(rounds):
        return None
    r = rounds[round_index - 1]
    try:
        actual = (float(r["lat"]), float(r["lng"]))
    except (KeyError, TypeError, ValueError):
        return None
    placed = None
    try:
        player = data.get("player") or {}
        guesses = player.get("guesses") or []
        if round_index - 1 < len(guesses):
            g = guesses[round_index - 1]
            placed = (float(g["lat"]), float(g["lng"]))
    except (KeyError, TypeError, ValueError):
        pass
    return {"actual": actual, "placed": placed}


_my_player_id: str | None = None


async def _get_my_player_id(page: Page) -> str | None:
    global _my_player_id
    if _my_player_id:
        return _my_player_id
    try:
        data = await page.evaluate(
            """async () => {
                const r = await fetch('/api/v3/profiles', {credentials: 'include'});
                if (!r.ok) return null;
                return await r.json();
            }"""
        )
        uid = ((data or {}).get("user") or {}).get("id") or (data or {}).get("id")
        if uid:
            _my_player_id = str(uid)
    except Exception as e:
        print(f"  [duels-api] could not read own profile: {e}")
    return _my_player_id


def _guess_is_recent(g: dict, max_age_s: float = 180.0) -> bool:
    """True if a duel guess was created in the last `max_age_s` seconds (or has no timestamp)."""
    created = g.get("created")
    if not created:
        return True
    try:
        from datetime import timezone
        t = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds() <= max_age_s
    except Exception:
        return True


async def read_duel_round_from_api(page: Page, round_hint: int) -> dict | None:
    """Duels / team-duels: the game-server exposes the whole match state, including
    every round's panorama coordinates and every player's guesses. Returns
    {actual, placed, round_number, country_code} or None."""
    m = re.search(r"/(?:duels|team-duels)/([A-Za-z0-9-]+)", page.url or "")
    if not m:
        return None
    game_id = m.group(1)
    try:
        data = await page.evaluate(
            """async (id) => {
                const r = await fetch(`https://game-server.geoguessr.com/api/duels/${id}`,
                                      {credentials: 'include'});
                if (!r.ok) return {__error: r.status};
                return await r.json();
            }""",
            game_id,
        )
    except Exception as e:
        print(f"  [duels-api] fetch failed: {e}")
        return None
    if not data or data.get("__error"):
        print(f"  [duels-api] game-server returned {data.get('__error') if data else 'nothing'}")
        return None
    rounds = data.get("rounds") or []
    if not rounds:
        return None

    me = await _get_my_player_id(page)
    my_guesses: list[dict] = []
    for team in data.get("teams") or []:
        for pl in team.get("players") or []:
            if me and str(pl.get("playerId")) == me:
                my_guesses = pl.get("guesses") or []

    round_no = None
    placed = None
    if my_guesses:
        g = max(my_guesses, key=lambda x: x.get("roundNumber", 0))
        if _guess_is_recent(g):
            round_no = g.get("roundNumber")
            try:
                placed = (float(g["lat"]), float(g["lng"]))
            except (KeyError, TypeError, ValueError):
                placed = None
    if not round_no:
        # Without our own guess to anchor on, trust the caller's round counter over
        # currentRoundNumber: the server may already have advanced to the next round
        # by the time we read, and scoring a guess against the wrong round silently
        # corrupts the log. Fall back to the server only when the hint is unusable.
        server_no = data.get("currentRoundNumber")
        if round_hint and any(r.get("roundNumber") == round_hint for r in rounds):
            round_no = round_hint
            if server_no and server_no != round_hint:
                print(f"  [duels-api] round {round_hint} (server says {server_no}) — "
                      f"using our own counter")
        else:
            round_no = server_no or round_hint

    rnd = next((r for r in rounds if r.get("roundNumber") == round_no), None)
    if rnd is None:
        idx = int(round_no) - 1
        rnd = rounds[idx] if 0 <= idx < len(rounds) else None
    if rnd is None:
        return None
    pano = rnd.get("panorama") or rnd
    try:
        actual = (float(pano["lat"]), float(pano["lng"]))
    except (KeyError, TypeError, ValueError):
        return None
    return {"actual": actual, "placed": placed, "round_number": round_no,
            "country_code": pano.get("countryCode")}


async def read_result_from_api(page: Page, round_num: int,
                               attempts: int = 1, delay_s: float = 2.0) -> dict | None:
    """Dispatch on the current URL: solo games use /api/v3/games, duels use the
    game-server. A duel guess takes a few seconds to show up in the match state
    (and the round only resolves once every player has guessed), so duels poll
    until the guess we just placed is visible."""
    url = page.url or ""
    is_duel = "/duels/" in url or "/team-duels/" in url
    last = None
    for i in range(max(attempts, 1)):
        if is_duel:
            last = await read_duel_round_from_api(page, round_num)
            if last and last.get("placed"):
                return last
        else:
            last = await read_round_from_api(page, round_num)
            if last:
                return last
        if i < attempts - 1:
            await asyncio.sleep(delay_s)
    return last


async def click_play_again(page: Page) -> bool:
    btn, _ = await query_first(page, [
        "button[data-qa='play-again-button']",
        "button:has-text('Play again')",
        "button:has-text('PLAY AGAIN')",
        "button:has-text('Jogar novamente')",
    ], "play-again")
    if btn:
        try:
            await btn.click()
            return True
        except Exception:
            pass
    return False


_last_canvas_sig: str | None = None


async def _canvas_signature(page: Page) -> str | None:
    """Visual hash of a small central region of the panorama canvas. Robust to
    GeoGuessr's internal JS structure — changes whenever the visible panorama
    content changes."""
    try:
        canvas = await page.query_selector(".widget-scene-canvas, canvas.mapsConsumerUiSceneCoreScene__canvas")
        if not canvas:
            return None
        box = await canvas.bounding_box()
        if not box or box["width"] < 400:
            return None
        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2
        sample = await page.screenshot(
            clip={"x": max(0.0, cx - 100), "y": max(0.0, cy - 100), "width": 200, "height": 200},
            type="png",
        )
        return hashlib.md5(sample).hexdigest()
    except Exception:
        return None


async def wait_for_round(page: Page, timeout_s: int = 120) -> bool:
    global _last_canvas_sig
    start = time.time()
    deadline = start + timeout_s
    pause_lo, pause_hi = (1.5, 2.0) if IS_MULTIPLAYER else (3.0, 4.0)

    # In duels, between rounds the canvas first shows a results overlay (world
    # map with player pins), THEN the new panorama. We must NOT capture during
    # the overlay. Strategy:
    #  1. Enforce a 5s minimum delay (results overlay typically lasts ~5-7s)
    #  2. Then take two hashes 1.5s apart — if equal (stable) AND different
    #     from last round, we have a fully-loaded new panorama
    #  3. If still transitioning after 25s, give up and proceed anyway
    is_duels_subsequent = IS_MULTIPLAYER and _last_canvas_sig is not None
    waited_for_canvas_msg = False

    if is_duels_subsequent:
        # Minimum wait to let the results overlay clear.
        await asyncio.sleep(5.0)

    stability_deadline = start + 25.0 if is_duels_subsequent else None

    while time.time() < deadline:
        canvas = await page.query_selector(".widget-scene-canvas, canvas.mapsConsumerUiSceneCoreScene__canvas")
        if not canvas:
            if not waited_for_canvas_msg:
                print("  [wait] waiting for panorama canvas to appear...")
                waited_for_canvas_msg = True
            await asyncio.sleep(1.0)
            continue
        box = await canvas.bounding_box()
        if not box or box["width"] < 400:
            await asyncio.sleep(1.0)
            continue

        if is_duels_subsequent and stability_deadline and time.time() < stability_deadline:
            # First: if the results breakdown panel is still visible (DOM check),
            # we are guaranteed to be in a between-rounds state — keep waiting
            # regardless of canvas hash.
            if await is_results_overlay_visible(page):
                await asyncio.sleep(0.8)
                continue
            sig_a = await _canvas_signature(page)
            if not sig_a:
                await asyncio.sleep(0.8)
                continue
            await asyncio.sleep(1.5)
            # Re-check overlay after the sleep — it can appear/disappear.
            if await is_results_overlay_visible(page):
                await asyncio.sleep(0.8)
                continue
            sig_b = await _canvas_signature(page)
            if not sig_b:
                await asyncio.sleep(0.8)
                continue
            # Both must be equal (canvas stable = fully rendered) AND different
            # from last round's panorama.
            if sig_a == sig_b and sig_a != _last_canvas_sig:
                print(f"  [canvas] stable new content (detected after {time.time()-start:.1f}s)")
                _last_canvas_sig = sig_a
                await human_delay(pause_lo, pause_hi)
                return True
            # Either still changing (overlay/transition) or still the previous
            # panorama. Keep polling.
            await asyncio.sleep(0.8)
            continue

        if is_duels_subsequent and stability_deadline and time.time() >= stability_deadline:
            print(f"  [canvas] stable new content not confirmed within 25s — proceeding anyway")
        sig = await _canvas_signature(page)
        if sig:
            _last_canvas_sig = sig
        await human_delay(pause_lo, pause_hi)
        return True

    print(f"  [wait] timed out after {timeout_s}s")
    return False


async def is_results_overlay_visible(page: Page) -> bool:
    """Detect the inter-round results / breakdown panel that appears between
    duels rounds. The panel typically shows 'Round X', score breakdown, and a
    world map — capturing during it pollutes OCR with leaderboard text and
    map labels."""
    for sel in [
        "div:has-text('Game Breakdown')",
        "button:has-text('WATCH REPLAY')",
        "button:has-text('Watch replay')",
        "[data-qa='round-result']",
        "[data-qa='duels-round-result']",
        "[class*='round-result']",
        "[class*='roundResult']",
        "[class*='breakdown']",
        "[class*='Breakdown']",
    ]:
        try:
            elem = await page.query_selector(sel)
            if elem and await elem.is_visible():
                return True
        except Exception:
            continue
    return False


async def is_duels_game_over(page: Page) -> bool:
    # Signal 1: URL no longer in an active duel/game.
    try:
        url = page.url or ""
        in_active = any(p in url for p in (
            "/duels/",
            "/team-duels/",
            "/game/",
            "/live-challenge/",
            "/battle-royale/",
        ))
        if url and not in_active:
            print(f"  [duels-end] URL no longer in active duel: {url}")
            return True
    except Exception:
        pass

    # Signal 2: end-of-match-specific UI elements. Avoid generic 'Continue' /
    # 'Play again' since those may also appear between rounds.
    selectors = [
        "button:has-text('Back to lobby')",
        "button:has-text('Return to lobby')",
        "button:has-text('Find a new game')",
        "button:has-text('Find new game')",
        "button:has-text('Leave game')",
        "div:has-text('Victory')",
        "div:has-text('Defeat')",
        "div:has-text('You won')",
        "div:has-text('You lost')",
        "div:has-text('VICTORY')",
        "div:has-text('DEFEAT')",
        "[data-qa='duels-end']",
        "[data-qa='game-end']",
        "[data-qa='match-end']",
        "[data-qa='duel-finished']",
        "[class*='gameOver']",
        "[class*='matchEnd']",
        "[class*='duels-end']",
    ]
    for sel in selectors:
        try:
            elem = await page.query_selector(sel)
            if elem:
                visible = await elem.is_visible()
                if visible:
                    print(f"  [duels-end] detected via selector: {sel}")
                    return True
        except Exception:
            continue
    return False
