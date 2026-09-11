# GeoGuessr AI Agent

Autonomous Python agent that plays [GeoGuessr](https://www.geoguessr.com/) end-to-end, combining browser automation, computer vision, retrieval-augmented generation (RAG) and LLM reasoning to guess the location shown in a round.

## How it works

1. **Play** — [Playwright](https://playwright.dev/) (with stealth mode) drives a real browser through a GeoGuessr round and captures a screenshot.
2. **Perceive** — OpenCV/OCR extract visual clues from the screenshot: compass heading, car metadata, license plates.
3. **Retrieve** — a RAG pipeline built with [ChromaDB](https://www.trychroma.com/) and OpenCLIP embeddings finds visually similar past rounds, and country-specific hints scraped from [Plonkit](https://www.plonkit.net/) plus reverse geocoding data ground the search space.
4. **Reason** — an LLM (OpenAI API) combines all of the above to produce a coordinate guess and reasoning trace.
5. **Act & learn** — the guess is submitted in the browser, the round result is logged, and later indexed back into the vector database to improve future retrieval.

## Tech stack

- **Automation:** Playwright, playwright-stealth
- **Computer vision:** OpenCV, Pillow
- **RAG / vector search:** ChromaDB, OpenCLIP embeddings
- **LLM reasoning:** OpenAI API
- **Geo data:** reverse_geocoder, Shapely, GeoJSON country boundaries

## Project structure

| File | Purpose |
|---|---|
| `bot.py` | Main agent loop — orchestrates browser automation, CV, RAG and LLM reasoning |
| `indexador.py` | Indexes past rounds (`log.json`) into ChromaDB for RAG |
| `scraper_plonkit.py` | Scrapes per-country geo-location hints from Plonkit |
| `obter_fronteiras.py` | Fetches/processes country boundary data |
| `stats.py`, `analyze_last_50.py`, `ab_compare.py` | Analyze bot performance (accuracy, distance error) over logged rounds |
| `replay_errors.py` | Replays and debugs rounds the bot got wrong |
| `metas/`, `metas_resumidas/` | Scraped and summarized per-country hint text |

## Setup

```bash
git clone https://github.com/timimata/geoguessr-ai-agent.git
cd geoguessr-ai-agent
python -m venv venv
venv\Scripts\activate      # or: source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
playwright install chromium
```

Copy `.env.example` to `.env` and fill in your own values:

```bash
cp .env.example .env
```

> `.env` holds your real API keys and is listed in `.gitignore`, so it is never committed — only the blank `.env.example` template is tracked.

## Usage

```bash
python bot.py
```

## Disclaimer

Built as a personal experiment in agentic AI (computer vision + RAG + LLM reasoning) applied to a single-player, non-competitive game mode. Not intended for use in ranked or competitive multiplayer.
