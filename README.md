# GeoGuessr AI Agent

A Python agent that plays [GeoGuessr](https://www.geoguessr.com/) on its own. It drives a real
browser, screenshots the panorama, works out where the photo was taken, drops the pin and reads
the score back. Then it remembers the round, so the next one has more to go on.

Over 475 rounds of NMPZ world play it gets the country right 87.6% of the time, with a median
error of 148 km.

## Demo

<p align="center">
  <a href="https://www.youtube.com/watch?v=nGEwr3GJsPM">
    <img src="https://img.youtube.com/vi/nGEwr3GJsPM/maxresdefault.jpg" width="800">
  </a>
</p>

<p align="center">
  <i> screenshot → reasoning → pin → score.</i>
</p>

## How it works

A round is one screenshot in, one pin out.

The screenshot goes to a vision model, which answers with a country, a region, coordinates, a
confidence and its top three candidate countries. Before that call, the same image is embedded
with [StreetCLIP](https://huggingface.co/geolocal/StreetCLIP) and matched against every round the
agent has played before. A close match is worth showing the model: this looked like a road in
Slovenia last time. A distant one is noise and gets dropped.

The answer then goes through checks that need no model at all. Are the coordinates inside the
country it named? On land? Does the continent it declared match the country? If anything is
wrong, one follow-up call gets the whole list of problems at once and rewrites the answer.
Coordinates still outside the named country get snapped into it. A pin still in the sea gets
moved to the nearest land the agent has seen in that country.

That is the whole thing: at most two model calls, usually one.

The pin is placed on the minimap by projecting the coordinate through Web Mercator, and the
result is read from GeoGuessr's own API rather than scraped off the screen. If the round was any
good, its screenshot and true location go into the index.

## What isn't there any more

Earlier versions ran OCR over the image, read the compass with OpenCV, looked for the Google car,
loaded scraped country cheat sheets, and prefixed every request with 1500 tokens of GeoGuessr
tradecraft. All of it is still in the repo behind flags, all of it is off, and all of it was
switched off because it was measured and found wanting. Dropping the hand-written prompt rules
was the single biggest accuracy gain in the project. Turning off OCR cut three quarters of the
per-round latency and cost nothing, because a vision model can already read the signs in the
picture you just handed it.

The numbers behind each of those decisions sit in `config.py`, next to the flag.

## Setup

Needs Python 3.11+ and a GeoGuessr account.

```bash
git clone https://github.com/timimata/geoguessr-ai-agent.git
cd geoguessr-ai-agent
python -m venv venv
venv\Scripts\activate            # source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
playwright install chromium
python obter_fronteiras.py       # country borders, 14 MB, not committed
```

Copy the env template and fill it in:

```bash
cp .env.example .env
```

You need a key at minimum. The default endpoint is Google's Gemini, but anything that speaks the
OpenAI chat API works, including a local LM Studio server:

```
GEMINI_API_KEY=your-key
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
MODEL_IDS=gemini-3.1-flash-lite
MODE=solo
RAG_EMBEDDING=streetclip
```

A round costs about 1700 tokens, so 150 rounds run for roughly $0.08 on `gemini-3.1-flash-lite`.
Newer is not automatically better here: `gemini-3.5-flash-lite` costs 25% more and scored two
points worse on the same rounds.

### Building the index

The retrieval index is what the agent remembers, and a fresh clone has nothing in it. That is
fine, it plays blind until it has history. If you already have a `log.jsonl` and the matching
screenshots:

```bash
python indexador.py
```

StreetCLIP is a ViT-L/14 and takes about 2.5s an image on CPU, so a few thousand rounds is an
hour. With a CUDA build of PyTorch it is minutes.

## Running it

```bash
python bot.py
```

A Chrome window opens on the world map. Log in if you need to, pick NMPZ, press Play. The agent
takes over from the first panorama and keeps going, clicking Play Again between games, until it
runs out of games or you close the window.

Started without a terminal attached, from a scheduler say, it skips the Enter prompt and waits
for a panorama to appear instead.

To watch it:

```bash
python stats.py              # dashboard on localhost:8000, refreshes itself
python analyze_last_50.py    # quick summary in the terminal
python analyze_last_50.py 300
```

## Results

475 rounds of live NMPZ world play on `gemini-3.1-flash-lite`:

| | Before | After |
|---|---|---|
| Country correct | 49.0% | 87.6% |
| Median error | 856 km | 148 km |
| Within 200 km | 19.7% | 56% |
| Over 2000 km | 30.9% | 2.3% |
| Seconds per round | ~15 | 3.6 |
| Model calls per round | up to 7 | 1.01 |

Fourteen rounds landed within a kilometre. Best game was 24785 out of 25000.

What is still wrong is almost all neighbours: Canada and the United States, Moldova and Ukraine,
Romania and Bulgaria, Lithuania and Latvia. Nothing lands on the wrong continent any more.

## Measuring a change

This is the part worth stealing for another project. Every round the agent has played is on disk
with its screenshot and its true location, so the whole pipeline can be replayed offline with no
browser:

```bash
python benchmark.py --build --n 150 --seed 42   # freeze a sample
python benchmark.py --label baseline
python benchmark.py --label no_soil --off soil  # same rounds, one feature off
python benchmark.py --compare baseline no_soil
```

The index is told to ignore the round being tested, or it finds itself and scores 100%. Each run
records the answer at three stages, so the review call, the clamp and the pin are measured
separately without a second run. It also records which prompt hints actually fired: a feature
that is enabled and fires in zero rounds out of 150 is broken, not idle, and the report says so.
That check exists because a cross-module bug once disabled a hint silently and several benchmark
runs measured a degraded pipeline without complaining.

Two things worth knowing before copying the approach. Retrieval precision measured on its own does
not predict end-to-end value: a looser match threshold scored better in isolation and worse in the
pipeline, because extra references drag the model off answers it already had right. And the model
runs at temperature 0.2, so two identical runs differ by around 20 score points. Treat anything
smaller than that as a tie and decide on cost instead.

`rag_calibrate.py` answers a narrower question: how often does the nearest neighbour in the index
name the right country? It reads the stored vectors, so it takes seconds. That is how StreetCLIP
was picked over generic OpenCLIP, 78% against 46% on the same 1550 rounds.

## Tests

```bash
python run_tests.py          # about 45 seconds
python run_tests.py --quick  # skips the suites that load the heavy models
```

No browser, no model, no network. They cover the coordinate maths that decides where the pin
lands, polygon clamping, the round log and its cache, the duels result reader, and the
module-level state that splitting the code up made easy to get wrong.

## Layout

| Module | |
|---|---|
| `config.py` | Environment, paths, feature flags, prompts |
| `geo.py` | Country tables, borders, reverse geocoding, distances |
| `vision.py` | Cropping, the compass, biome colour, OCR |
| `llm.py` | The model call and the JSON that comes back |
| `rag.py` | The vector index and what is worth remembering |
| `storage.py` | The round log |
| `browser.py` | Playwright: capture, pin placement, reading results |
| `pipeline.py` | Screenshot in, pin out, no browser |
| `bot.py` | Entry point, the round and match loops |

| Tool | |
|---|---|
| `benchmark.py` | Replay logged rounds offline to score a change |
| `rag_calibrate.py` | How good the retrieval actually is |
| `indexador.py` | Build the index from the log |
| `stats.py` | Dashboard |
| `analyze_last_50.py`, `ab_compare.py` | Terminal summaries |
| `replay_errors.py` | Reopen the worst rounds |
| `migrate_log.py`, `compress_screenshots.py` | Storage housekeeping |
| `scraper_plonkit.py`, `obter_fronteiras.py` | Fetch the reference data |

## Duels

`MODE=duels` or `team-duels` plays multiplayer. The code reads match state from GeoGuessr's game
server and queues rounds that have not resolved yet, since a duel round only finishes once every
player has guessed. It is covered by tests against recorded payloads but has never run against a
live match, so treat it as untested.

## Notes

All of the above was measured against one model. `gemini-3.1-flash-lite` reads signage and
geolocates well without help, which is why the OCR and the cheat sheets earn nothing. A weaker or
text-blind local model would probably want several of those flags back on. Re-measure rather than
assuming.

`.env` holds your key and is gitignored, as are the screenshots, the round log and the browser
profile.

## Disclaimer

A personal experiment in applying vision models and retrieval to a single-player, non-competitive
game mode. Not for ranked or competitive multiplayer.
