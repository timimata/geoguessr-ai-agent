"""test_browser.py - checks on the browser layer that need no browser.

The pin placement maths is the part of the agent where a correct guess still
loses the round: project a coordinate to the wrong pixel and the pin lands in
another country. None of it needed a real page, and none of it was covered.

Page-driven helpers are exercised against a fake page, the same way
test_duels_api.py does, so selector handling and round detection are covered too.

Run: python test_browser.py
"""
import asyncio
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

config.QUIET = True

import browser

_failures: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    if not ok:
        _failures.append(label)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))


def check_true(label: str, got) -> None:
    check(label, bool(got), True)


def close(label: str, got: float, want: float, tol: float) -> None:
    ok = abs(got - want) <= tol
    if not ok:
        _failures.append(label)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: {got:.1f}" + ("" if ok else f"  (want {want:.1f} ±{tol})"))


WORLD = {"x": 0.0, "y": 0.0, "width": 1000.0, "height": 1000.0}


def test_world_mercator():
    print("world Mercator projection")
    # Null Island sits at the horizontal centre, and at the vertical centre too
    # because Web Mercator is symmetric about the equator.
    x, y = browser.latlon_to_world_mercator(0.0, 0.0, WORLD)
    close("equator/prime meridian is the centre (x)", x, 500.0, 0.5)
    close("equator/prime meridian is the centre (y)", y, 500.0, 0.5)

    # Longitude is linear, so ±90° lands at the quarter and three-quarter marks.
    close("lon -90 is a quarter across", browser.latlon_to_world_mercator(0, -90, WORLD)[0], 250.0, 0.5)
    close("lon +90 is three quarters across", browser.latlon_to_world_mercator(0, 90, WORLD)[0], 750.0, 0.5)

    # North is up: a northern latitude must produce a smaller y than a southern one.
    y_north = browser.latlon_to_world_mercator(60.0, 0.0, WORLD)[1]
    y_south = browser.latlon_to_world_mercator(-60.0, 0.0, WORLD)[1]
    check_true("north of the equator is above south", y_north < 500.0 < y_south)
    close("±60° are symmetric about the centre", y_north + y_south, 1000.0, 1.0)

    # The poles are clamped instead of running to infinity.
    for lat in (90.0, -90.0, 89.999):
        _, py = browser.latlon_to_world_mercator(lat, 0.0, WORLD)
        check_true(f"lat {lat} stays inside the box", 0.0 <= py <= 1000.0)

    # Everything must land inside the box, offsets included.
    offset = {"x": 120.0, "y": 40.0, "width": 300.0, "height": 200.0}
    for lat, lon in ((38.7, -9.1), (-33.9, 18.4), (35.7, 139.7), (64.1, -21.9)):
        px, py = browser.latlon_to_world_mercator(lat, lon, offset)
        check_true(f"({lat},{lon}) inside an offset box",
                   offset["x"] <= px <= offset["x"] + offset["width"]
                   and offset["y"] <= py <= offset["y"] + offset["height"])


def test_bounds_projection():
    print("projection inside known viewport bounds")
    # A viewport centred on Portugal. Its own centre must map to the box centre.
    bounds = {"north": 42.0, "south": 37.0, "west": -10.0, "east": -6.0}
    mid_lon = (bounds["west"] + bounds["east"]) / 2

    def merc_y(lat):
        s = math.sin(math.radians(lat))
        return 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)

    def inv_merc_y(y):
        e = math.exp((0.5 - y) * 4 * math.pi)
        return math.degrees(math.asin((e - 1) / (e + 1)))

    # The vertical centre of a Mercator viewport is not the mean of its latitudes.
    target = (merc_y(bounds["north"]) + merc_y(bounds["south"])) / 2
    mid_lat = inv_merc_y(target)
    check_true("the Mercator midpoint sits between the bounds",
               bounds["south"] < mid_lat < bounds["north"])

    x, y = browser.latlon_to_minimap_xy_bounds(mid_lat, mid_lon, WORLD, bounds)
    close("viewport centre maps to box centre (x)", x, 500.0, 1.0)
    close("viewport centre maps to box centre (y)", y, 500.0, 1.0)

    # Corners map to corners.
    x, y = browser.latlon_to_minimap_xy_bounds(bounds["north"], bounds["west"], WORLD, bounds)
    close("north-west corner x", x, 0.0, 0.5)
    close("north-west corner y", y, 0.0, 0.5)

    # A point outside the viewport is clamped to the edge, never off-box.
    x, y = browser.latlon_to_minimap_xy_bounds(0.0, 100.0, WORLD, bounds)
    check_true("a point outside the viewport is clamped", 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0)

    # Viewport crossing the antimeridian: Fiji must not land on the far side.
    wrap = {"north": 10.0, "south": -30.0, "west": 160.0, "east": -160.0}
    x_fiji, _ = browser.latlon_to_minimap_xy_bounds(-17.7, 178.0, WORLD, wrap)
    x_samoa, _ = browser.latlon_to_minimap_xy_bounds(-13.7, -172.0, WORLD, wrap)
    check_true("across the antimeridian, 178E is west of 172W", x_fiji < x_samoa)
    check_true("both stay inside the box", 0.0 <= x_fiji <= 1000.0 and 0.0 <= x_samoa <= 1000.0)


def test_minimap_dispatch():
    print("minimap projection dispatch")
    wide = {"x": 0.0, "y": 0.0, "width": 1200.0, "height": 800.0}
    check("a wide guess map uses world Mercator",
          browser.latlon_to_minimap_xy(35.7, 139.7, wide),
          browser.latlon_to_world_mercator(35.7, 139.7, wide))
    small = {"x": 10.0, "y": 10.0, "width": 300.0, "height": 200.0}
    x, y = browser.latlon_to_minimap_xy(35.7, 139.7, small)
    check_true("a small corner minimap stays inside its box",
               10.0 <= x <= 310.0 and 10.0 <= y <= 210.0)


def test_guess_freshness():
    print("duel guess freshness")
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    fresh = {"created": now.isoformat().replace("+00:00", "Z")}
    stale = {"created": (now - timedelta(hours=3)).isoformat().replace("+00:00", "Z")}
    check_true("a guess made now is recent", browser._guess_is_recent(fresh))
    check("a guess from three hours ago is not", browser._guess_is_recent(stale), False)
    check_true("a guess with no timestamp is assumed recent", browser._guess_is_recent({}))
    check_true("an unparseable timestamp is assumed recent",
               browser._guess_is_recent({"created": "not a date"}))
    check("the age limit is honoured",
          browser._guess_is_recent({"created": (now - timedelta(seconds=60)).isoformat()
                                    .replace("+00:00", "Z")}, max_age_s=30), False)


class FakePage:
    """Enough of a Playwright page for the selector helpers."""

    def __init__(self, present=(), broken=(), url="https://www.geoguessr.com/game/X"):
        self.present = set(present)
        self.broken = set(broken)
        self.url = url
        self.clicked = []
        self.tried = []

    async def query_selector(self, sel):
        self.tried.append(sel)
        if sel in self.broken:
            raise ValueError(f"unsupported selector: {sel}")
        return FakeElement(sel, self) if sel in self.present else None


class FakeElement:
    def __init__(self, sel, page):
        self.sel = sel
        self.page = page

    async def click(self):
        self.page.clicked.append(self.sel)

    async def is_visible(self):
        return True


async def test_selectors():
    print("selector handling")
    page = FakePage(present={"b"})
    el, sel = await browser.query_first(page, ["a", "b", "c"], "t")
    check("returns the first selector that matches", sel, "b")
    check("stops looking once it matches", page.tried, ["a", "b"])

    # A malformed selector must be skipped, not abort the round. This is the
    # class of bug that the old "button.button_variantPrimary__*" caused.
    page = FakePage(present={"good"}, broken={"bad[*"})
    el, sel = await browser.query_first(page, ["bad[*", "good"], "t")
    check("a malformed selector is skipped", sel, "good")

    page = FakePage()
    el, sel = await browser.query_first(page, ["x", "y"], "t")
    check("no match returns nothing", (el, sel), (None, ""))

    page = FakePage(present={"button:has-text('Play again')"})
    check_true("play again is found and clicked", await browser.click_play_again(page))
    check("and it clicked the right one", page.clicked, ["button:has-text('Play again')"])

    page = FakePage()
    check("play again reports failure when absent", await browser.click_play_again(page), False)

    page = FakePage(present={"button:has-text('Back to lobby')"})
    check_true("end of match is detected", await browser.is_duels_game_over(page))

    # A URL that is no longer an active duel also means the match is over.
    page = FakePage(url="https://www.geoguessr.com/me/profile")
    check_true("leaving the duel URL ends the match", await browser.is_duels_game_over(page))

    page = FakePage(present={"[class*='breakdown']"})
    check_true("the between-rounds panel is detected",
               await browser.is_results_overlay_visible(page))


def main() -> int:
    for fn in (test_world_mercator, test_bounds_projection, test_minimap_dispatch,
               test_guess_freshness):
        try:
            fn()
        except Exception as e:
            _failures.append(fn.__name__)
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
    try:
        asyncio.run(test_selectors())
    except Exception as e:
        _failures.append("test_selectors")
        print(f"  ERROR in test_selectors: {type(e).__name__}: {e}")
    print()
    if _failures:
        print(f"RESULT: {len(_failures)} failure(s): {', '.join(_failures)}")
        return 1
    print("RESULT: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
