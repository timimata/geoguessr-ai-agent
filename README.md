# GeoGuessr AI Agent

Autonomous Python agent that plays [GeoGuessr](https://www.geoguessr.com/) end-to-end, combining browser automation, computer vision, retrieval-augmented generation (RAG) and LLM reasoning to guess the location shown in a round.

## How it works

1. **Play** — [Playwright](https://playwright.dev/) (with stealth mode) drives a real browser through a GeoGuessr round and captures a screenshot.
2. **Perceive** — OpenCV/OCR extract visual clues from the screenshot: compass heading, car metadata, license plates.
3. **Retrieve** — a RAG pipeline built with [ChromaDB](https://www.trychroma.com/) and OpenCLIP embeddings finds visually similar past rounds, and country-specific hints scraped from [Plonkit](https://www.plonkit.net/) plus reverse geocoding data ground the search space.
4. **Reason** — a vision LLM combines all of the above into a coordinate guess, a confidence and a reasoning trace. Any OpenAI-compatible endpoint works; the default `OPENAI_BASE_URL` points at Google's Gemini compatibility layer, and local servers such as LM Studio work the same way.
5. **Act & learn** — the guess is submitted in the browser, the round result is logged, and later indexed back into the vector database to improve future retrieval.

## Tech stack

- **Automation:** Playwright, playwright-stealth
- **Computer vision:** OpenCV, Pillow
- **RAG / vector search:** ChromaDB, OpenCLIP embeddings
- **LLM reasoning:** the `openai` client against any OpenAI-compatible endpoint (Gemini by default)
- **Geo data:** reverse_geocoder, Shapely, GeoJSON country boundaries

## Project structure

The agent is split so that the part that decides where a photo was taken never
touches the browser, which is what makes a logged round replayable offline.

| Module | Purpose |
|---|---|
| `config.py` | Environment, paths, feature flags, prompts. Loaded first by everything else |
| `geo.py` | Country tables, boundary polygons, reverse geocoding, distance maths |
| `vision.py` | Cropping, the OpenCV compass, biome colour analysis, OCR |
| `llm.py` | The model client, the request it sends and the JSON it parses back |
| `rag.py` | The vector index, Plonkit cheat sheets, confusion pairs |
| `storage.py` | Reading and appending the round log |
| `browser.py` | Playwright: capturing the panorama, placing the pin, reading results |
| `pipeline.py` | The decision pipeline. Screenshot in, final pin out, no browser |
| `bot.py` | Entry point: the live round loop and the match loop |

| Tool | Purpose |
|---|---|
| `benchmark.py` | Replays logged rounds through the pipeline offline to score a change |
| `rag_calibrate.py` | Measures how often a retrieved neighbour names the right country |
| `indexador.py` | Builds the vector index from the round log |
| `migrate_log.py` | Converts `log.json` to the append-only `log.jsonl` |
| `compress_screenshots.py` | Re-encodes stored PNG rounds to JPEG (about 5x smaller) |
| `scraper_plonkit.py` | Scrapes per-country location hints from Plonkit |
| `obter_fronteiras.py` | Downloads `countries.geojson`, the boundaries used for clamping |
| `stats.py` | Serves an HTML performance dashboard on `http://localhost:8000/` |
| `analyze_last_50.py`, `ab_compare.py` | Accuracy and per-model comparisons over logged rounds |
| `replay_errors.py` | Lists and reopens the rounds the bot got most wrong |
| `run_tests.py` | Runs every offline suite; `--quick` skips the ones loading heavy models |
| `test_logic.py` | Decision logic: clamping, calibration, scoring, cross-module state |
| `test_browser.py` | Pin placement maths and selector handling, against a fake page |
| `test_duels_api.py` | The duels result reader, against canned game-server payloads |
| `metas/`, `metas_resumidas/` | Scraped and summarised per-country hint text |

## Setup

```bash
git clone https://github.com/timimata/geoguessr-ai-agent.git
cd geoguessr-ai-agent
python -m venv venv
venv\Scripts\activate      # or: source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
playwright install chromium
python obter_fronteiras.py     # downloads countries.geojson (14 MB, not committed)
```

If you have an NVIDIA GPU, install a CUDA build of PyTorch before the rest.
Without one, EasyOCR runs on CPU and becomes the slowest part of every round.

Copy `.env.example` to `.env` and fill in your own values:

```bash
cp .env.example .env
```

`GEMINI_API_KEY` (or `OPENAI_API_KEY`) holds the credential, `OPENAI_BASE_URL`
the endpoint, and `MODEL_IDS` a comma-separated list that the bot rotates through
round by round so two models can be compared on the same stream of locations.

> `.env` holds your real API keys and is listed in `.gitignore`, so it is never committed — only the blank `.env.example` template is tracked.

## Usage

```bash
python bot.py
```

`MODE` in `.env` selects the game type: `solo`, `duels` or `team-duels`. Solo reads
each round result from `/api/v3/games`; duels read the match state from the
GeoGuessr game server. A duel round only resolves once every player has guessed,
so rounds that are not readable straight away are queued and scored at the start
of the next round or at the end of the match.

## Decision pipeline

One round costs at most two model calls. The first call answers with a country,
coordinates, a confidence and its top-3 candidate countries. Deterministic checks
then look for contradictions: coordinates in the sea or in the wrong country, a
declared continent that does not match the country, OCR text naming a different
country, or a visual match that disagrees. If any fire, a single review call gets
the whole list at once and rewrites the answer. Coordinates still outside the
declared country are clamped to its polygon.

Reported confidence is replaced by the historical hit rate at that value, taken
from `log.json`. That calibrated probability decides whether to hedge: when the
model is torn between two countries, the pin moves toward the rival only if the
expected GeoGuessr score goes up.

## Measuring a change

Every logged round has a screenshot and a true location, so the pipeline can be
replayed without a browser. The retrieval index is told to ignore the round being
evaluated so it cannot find itself.

```bash
python benchmark.py --build --n 150 --seed 42   # freeze a sample
python benchmark.py --label baseline            # score the current pipeline
python benchmark.py --label no_soil --off soil  # same rounds, one feature off
python benchmark.py --compare baseline no_soil  # side by side, with win/loss rounds
python benchmark.py --calibration               # reported confidence vs real hit rate
```

Each run also records which prompt hints actually fired. A feature that is
enabled and fires in zero rounds is broken rather than idle, and the report
says so: that is how a cross-module aliasing bug had silently disabled the
confusion-pair hint across several runs.

Feature names come from `FEATURES` in `config.py`, and `FEATURES_OFF` in `.env`
disables the same switches during live play. Each run records the answer at three
stages, so the effect of the review call, the clamp and the hedge is read off the
same rounds without needing a second run.

Two flags are off by default because the benchmark measured them as losses on a
150-round set with `gemini-3.1-flash-lite`:

| Flag | Evidence for switching it off |
|---|---|
| `rich_system_prompt` | Dropping ~1500 tokens of hand-written rules took accuracy from 84.0% to 88.7% |
| `correction_rag_continent` | Fired 6 times in 150 rounds, changed the answer twice, both losses, one of them 22 km to 738 km |
| `hedge` | Fired 8 times for +21 score points in total, and flipped one correct country to wrong |
| `rag_car` | Its nearest neighbour named the right country 20.9% of the time, below the 21.9% you get by always answering "United States" |
| `metas`, `soil` | A dead tie (88.0% against 88.7%, one point of score), so decided on what they cost |

The last row is the honest case: those two are not harmful, they simply earn
nothing while costing tokens on every round. They were measured against
`gemini-3.1-flash-lite`, which geolocates well unaided. A weaker local model may
genuinely need the tips, so re-measure before assuming this carries over.

Turn any of these back on with evidence from a larger set, not from a hunch.

The through-line across all of them: this pipeline was losing accuracy to its own
context. Every hint that was measured turned out to be neutral or harmful, and
the one change that helped most was deleting text. Retrieval precision measured
on its own did not predict end-to-end value either, so measure the pipeline, not
the component.

The model runs at temperature 0.2, so two runs of the identical configuration
differ. Measured run to run on this set, the noise is around 20 score points and
half a point of country accuracy. Treat anything smaller than that as a tie and
decide on cost instead.

## How good is the retrieval?

`rag_calibrate.py` compares every indexed round against every other one using the
embeddings already stored, so it answers in seconds without running the encoder.
On 1979 rounds with the default OpenCLIP encoder:

| Nearest neighbour | Median cosine distance |
|---|---|
| Same country | 0.096 |
| Different country | 0.113 |

Those two distributions overlap almost completely, so no threshold separates
them. At the current setting every round gets a reference and only 43% of them
name the right country. That is why the prompt carries continent vetoes and
frequency warnings: they are patching a weak signal.

`RAG_EMBEDDING=streetclip` swaps in a CLIP checkpoint fine-tuned for geolocation
on Street View imagery. Each encoder keeps its own collections, so switching
means rebuilding:

```bash
RAG_EMBEDDING=streetclip python indexador.py
RAG_EMBEDDING=streetclip python rag_calibrate.py --ids-from geoguessr_rondas
```

The second command restricts the comparison to the rounds both indexes hold, so
the two encoders are judged on the same images. StreetCLIP is a ViT-L/14 and
takes about 2.5 s per image on CPU, so budget an hour for a full rebuild.

## Tests

```bash
python run_tests.py          # everything, about 45 seconds
python run_tests.py --quick  # skip the suites that load the country polygons and CLIP
```

They call the real functions rather than reimplementing them, and none of them
need a browser, a model or a network. What they cover: coordinate maths and pin
projection, polygon clamping and the open-water rescue, confidence parsing and
calibration, the round log including its cache, the duels result reader, and the
module-level state that the split into modules made easy to get wrong.

## Storage

The round log is JSON Lines (`log.jsonl`): appending a round is one write at the
end of the file, and reads come from a cache that re-parses only when the file
changes. It used to be a single JSON array rewritten in full every round, and
read several times per round on top of that. `python migrate_log.py --apply`
converts an existing `log.json`, leaving the original in place as a backup.

Rounds are saved as JPEG. `python compress_screenshots.py` reports what
converting the existing PNGs would save (about 5x on real panoramas) and only
writes when passed `--apply`.

## Disclaimer

Built as a personal experiment in agentic AI (computer vision + RAG + LLM reasoning) applied to a single-player, non-competitive game mode. Not intended for use in ranked or competitive multiplayer.
