import json
import math
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from storage import load_rounds, log_path

PROJECT_DIR = Path(__file__).parent
PORT = 8000

# GeoGuessr-style round score: 5000 at 0 km, decays with world-map scale.
# World map uses ~2000 km decay (approximation; official formula is map-specific).
def geoguessr_score(error_km: float | None) -> int:
    if error_km is None:
        return 0
    if error_km < 0.15:
        return 5000
    return int(round(5000 * math.exp(-float(error_km) / 2000.0)))


def group_games(rounds: list[dict]) -> list[list[dict]]:
    """Split the flat round log into games. A new game starts when round_num resets to 1."""
    games: list[list[dict]] = []
    current: list[dict] = []
    for r in rounds:
        rn = r.get("round")
        if rn == 1 and current:
            games.append(current)
            current = []
        current.append(r)
    if current:
        games.append(current)
    return games


def top_games(rounds: list[dict], n: int = 10) -> list[dict]:
    """Return the top-N completed games (5 rounds) ranked by total score desc."""
    games = group_games(rounds)
    scored = []
    for idx, g in enumerate(games, 1):
        if len(g) < 5:
            continue  # ignore partial games
        per_round = [geoguessr_score(r.get("error_km")) for r in g]
        total = sum(per_round)
        scored.append({
            "game_idx": idx,
            "total": total,
            "per_round": per_round,
            "countries": [(r.get("actual") or {}).get("country", "?") for r in g],
            "timestamp": g[0].get("timestamp", ""),
        })
    scored.sort(key=lambda x: x["total"], reverse=True)
    return scored[:n]


COUNTRY_ALIASES = {
    "usa": "united states",
    "us": "united states",
    "u.s.": "united states",
    "u.s.a.": "united states",
    "america": "united states",
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "britain": "united kingdom",
    "great britain": "united kingdom",
}


def normalize_country(s: str) -> str:
    s = (s or "").strip().lower()
    return COUNTRY_ALIASES.get(s, s)


def compute_stats() -> dict:
    try:
        rounds = load_rounds(force=True)
    except Exception:
        return {"total": 0, "rounds": []}
    if not rounds:
        return {"total": 0, "rounds": []}

    total = len(rounds)
    with_actual = [r for r in rounds if r.get("actual") and r.get("error_km") is not None]
    errors = [r["error_km"] for r in with_actual]

    country_hits = sum(1 for r in rounds if r.get("country_hit") is True)
    country_attempts = sum(1 for r in rounds if r.get("country_hit") is not None)

    def pct(a, b):
        return round(100 * a / b, 1) if b else 0

    stats = {
        "total": total,
        "with_actual": len(with_actual),
        "avg_error_km": round(sum(errors) / len(errors), 1) if errors else None,
        "median_error_km": round(sorted(errors)[len(errors) // 2], 1) if errors else None,
        "best_error_km": round(min(errors), 1) if errors else None,
        "worst_error_km": round(max(errors), 1) if errors else None,
        "country_hit_rate_pct": pct(country_hits, country_attempts),
        "country_hits": country_hits,
        "country_attempts": country_attempts,
        "top_guessed_countries": Counter(
            normalize_country((r.get("guess") or {}).get("country", "")) for r in rounds
        ).most_common(10),
        "worst_missed_countries": Counter(
            normalize_country((r.get("actual") or {}).get("country", "")) for r in rounds if r.get("country_hit") is False and r.get("actual")
        ).most_common(10),
        "recent": rounds[-10:][::-1],
        "top_games": top_games(rounds, 10),
    }
    return stats


HTML_TMPL = """<!doctype html>
<html><head><meta charset="utf-8"><title>GeoGuessr Bot Stats</title>
<meta http-equiv="refresh" content="10">
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif; background:#111; color:#eee; margin:2rem; }}
  h1 {{ margin-top:0; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:1rem; margin-bottom:2rem; }}
  .card {{ background:#1e1e2e; border-radius:8px; padding:1rem; }}
  .card .v {{ font-size:1.8rem; font-weight:bold; color:#a6e3a1; }}
  .card .l {{ color:#888; font-size:0.85rem; }}
  table {{ width:100%; border-collapse:collapse; background:#1e1e2e; border-radius:8px; overflow:hidden; }}
  th, td {{ padding:0.6rem 0.9rem; text-align:left; border-bottom:1px solid #333; font-size:0.9rem; }}
  th {{ background:#2a2a3e; }}
  tr:last-child td {{ border-bottom:none; }}
  .err-good {{ color:#a6e3a1; }}
  .err-ok {{ color:#f9e2af; }}
  .err-bad {{ color:#f38ba8; }}
  code {{ background:#2a2a3e; padding:2px 6px; border-radius:4px; font-size:0.85rem; }}
</style></head><body>
<h1>GeoGuessr Bot — Stats</h1>
<p style="color:#888">Auto-refreshes every 10s. Log: <code>{log_name}</code></p>

<div class="cards">
  <div class="card"><div class="l">Total rounds</div><div class="v">{total}</div></div>
  <div class="card"><div class="l">With actual location</div><div class="v">{with_actual}</div></div>
  <div class="card"><div class="l">Avg error (km)</div><div class="v">{avg_error_km}</div></div>
  <div class="card"><div class="l">Median error (km)</div><div class="v">{median_error_km}</div></div>
  <div class="card"><div class="l">Best (km)</div><div class="v">{best_error_km}</div></div>
  <div class="card"><div class="l">Worst (km)</div><div class="v">{worst_error_km}</div></div>
  <div class="card"><div class="l">Country hit rate</div><div class="v">{country_hit_rate_pct}%</div><div class="l">{country_hits}/{country_attempts}</div></div>
</div>

<h2>Top guessed countries</h2>
<table><tr><th>Country</th><th>Count</th></tr>
{top_countries_rows}
</table>

<h2 style="margin-top:2rem">Most missed countries</h2>
<table><tr><th>Country</th><th>Errors</th></tr>
{missed_countries_rows}
</table>

<h2 style="margin-top:2rem">🏆 Top 10 games (GeoGuessr score / 25000)</h2>
<table>
  <tr><th>#</th><th>Game</th><th>Score</th><th>Per-round</th><th>Countries</th><th>Started</th></tr>
  {top_games_rows}
</table>

<h2 style="margin-top:2rem">Last 10 rounds</h2>
<table>
  <tr><th>#</th><th>Timestamp</th><th>Guess</th><th>Actual</th><th>Error (km)</th><th>Conf</th></tr>
  {recent_rows}
</table>
</body></html>"""


def render_html() -> str:
    s = compute_stats()

    def fmt(v):
        return "–" if v is None else v

    top_rows = "".join(
        f"<tr><td>{c or '(empty)'}</td><td>{n}</td></tr>" for c, n in s.get("top_guessed_countries", [])
    ) or "<tr><td colspan=2>No data</td></tr>"

    missed_rows = "".join(
        f"<tr><td>{c or '(empty)'}</td><td>{n}</td></tr>" for c, n in s.get("worst_missed_countries", [])
    ) or "<tr><td colspan=2>No data</td></tr>"

    recent_parts = []
    for i, r in enumerate(s.get("recent", []), 1):
        g = r.get("guess") or {}
        a = r.get("actual") or {}
        err = r.get("error_km")
        if err is None:
            cls, err_s = "", "–"
        elif err < 500:
            cls, err_s = "err-good", f"{err:.0f}"
        elif err < 3000:
            cls, err_s = "err-ok", f"{err:.0f}"
        else:
            cls, err_s = "err-bad", f"{err:.0f}"

        try:
            guess_txt = f"{g.get('country','?')} ({float(g.get('latitude',0)):.2f}, {float(g.get('longitude',0)):.2f})"
        except Exception:
            guess_txt = str(g.get("country", "?"))

        if a:
            actual_txt = f"{float(a.get('latitude',0)):.2f}, {float(a.get('longitude',0)):.2f}"
        else:
            actual_txt = "–"

        recent_parts.append(
            f"<tr><td>{i}</td><td>{r.get('timestamp','')}</td>"
            f"<td>{guess_txt}</td><td>{actual_txt}</td>"
            f"<td class='{cls}'>{err_s}</td><td>{g.get('confidence','?')}</td></tr>"
        )
    recent_rows = "".join(recent_parts) or "<tr><td colspan=6>No rounds yet</td></tr>"

    top_game_parts = []
    for rank, tg in enumerate(s.get("top_games", []), 1):
        per = ", ".join(str(p) for p in tg["per_round"])
        countries = ", ".join(tg["countries"])
        total = tg["total"]
        if total >= 18000:
            cls = "err-good"
        elif total >= 10000:
            cls = "err-ok"
        else:
            cls = "err-bad"
        top_game_parts.append(
            f"<tr><td>{rank}</td><td>#{tg['game_idx']}</td>"
            f"<td class='{cls}'><b>{total}</b></td>"
            f"<td><code>{per}</code></td>"
            f"<td style='font-size:0.8rem;color:#bbb'>{countries}</td>"
            f"<td style='color:#888'>{tg['timestamp']}</td></tr>"
        )
    top_games_rows = "".join(top_game_parts) or "<tr><td colspan=6>No complete games yet</td></tr>"

    return HTML_TMPL.format(
        log_name=log_path().name,
        total=s.get("total", 0),
        with_actual=s.get("with_actual", 0),
        avg_error_km=fmt(s.get("avg_error_km")),
        median_error_km=fmt(s.get("median_error_km")),
        best_error_km=fmt(s.get("best_error_km")),
        worst_error_km=fmt(s.get("worst_error_km")),
        country_hit_rate_pct=s.get("country_hit_rate_pct", 0),
        country_hits=s.get("country_hits", 0),
        country_attempts=s.get("country_attempts", 0),
        top_countries_rows=top_rows,
        missed_countries_rows=missed_rows,
        recent_rows=recent_rows,
        top_games_rows=top_games_rows,
    )


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = render_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stats.json":
            body = json.dumps(compute_stats(), indent=2, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass  # quiet


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Stats server running on http://localhost:{PORT}/")
    server.serve_forever()
