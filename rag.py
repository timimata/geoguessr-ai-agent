"""Retrieval over past rounds: the vector index, the Plonkit cheat sheets and
the confusion pairs derived from the bot's own mistakes."""
import json
import os
import threading
import time
from pathlib import Path

import chromadb
import numpy as np
from PIL import Image
from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

import config
from config import PROJECT_DIR, feature_on, say
from geo import continent_of
from storage import count_rounds_in_log, load_rounds

def compute_confusion_pairs(min_count: int = 3, max_error_km: float = 1500.0) -> list[tuple[str, str, int]]:
    """Return (guessed_wrong, actual_correct, count) for systematic near-miss errors.
    Only considers rounds where error < max_error_km (right region, wrong country).
    Called once at startup and cached."""
    data = load_rounds()
    from collections import Counter
    pairs: Counter = Counter()
    for r in data:
        if r.get("country_hit") is False and (r.get("error_km") or 9999) < max_error_km:
            guessed = (r.get("guess") or {}).get("country", "")
            actual = (r.get("actual") or {}).get("country", "")
            if guessed and actual and guessed != actual and guessed != "?":
                pairs[(guessed, actual)] += 1
    return [(g, a, c) for (g, a), c in pairs.most_common(20) if c >= min_count]


# Compute at import time so play_round doesn't pay the I/O cost every round.
_CONFUSION_PAIRS: list[tuple[str, str, int]] = []
_last_confusion_refresh_count: int = 0

def warm_up() -> None:
    """Load the retrieval model and open the index now, rather than during the
    first round. StreetCLIP is a ViT-L/14 and takes about 90 seconds to load on
    CPU, which is time the first round does not have."""
    started = time.time()
    coll, _ = get_chroma_collection()
    if coll is None:
        return
    try:
        # One throwaway query so the model weights are actually resident, not
        # merely downloaded: get_collection alone does not run a forward pass.
        coll.query(query_images=[np.zeros((64, 64, 3), dtype=np.uint8)], n_results=1,
                   include=[])
        say(f"[rag] {RAG_EMBEDDING} ready over {coll.count()} rounds "
            f"({time.time() - started:.0f}s)")
    except Exception as e:
        say(f"[rag] warm-up failed: {e}")


def confusion_pairs() -> list[tuple[str, str, int]]:
    """The current pairs. Call this instead of importing _CONFUSION_PAIRS: the
    list is rebound by _load_confusion_pairs(), so a `from rag import` binding in
    another module silently freezes at the empty startup value."""
    return _CONFUSION_PAIRS


def needs_confusion_refresh(current_count: int, every: int = 25) -> bool:
    """True once `every` new rounds have been logged since the last reload."""
    return current_count - _last_confusion_refresh_count >= every


def _load_confusion_pairs() -> None:
    global _CONFUSION_PAIRS, _last_confusion_refresh_count
    _CONFUSION_PAIRS = compute_confusion_pairs()
    _last_confusion_refresh_count = count_rounds_in_log()
    if _CONFUSION_PAIRS:
        say(f"[confusion] {len(_CONFUSION_PAIRS)} pairs loaded (log={_last_confusion_refresh_count}): "
              + ", ".join(f"{g}->{a}({c}x)" for g, a, c in _CONFUSION_PAIRS[:5]))

_chroma_client = None
_chroma_lock = threading.Lock()
_chroma_collection = None
_chroma_car_collection = None

def get_chroma_collection():
    global _chroma_client, _chroma_collection, _chroma_car_collection
    with _chroma_lock:
        return _get_chroma_collection_locked()


# --- embedding backend ----------------------------------------------------
# RAG_EMBEDDING picks which image encoder builds and queries the index.
#
#   openclip    ViT-B-32 trained on LAION. Generic: it encodes "green rolling
#               hills", not "this is Wales". rag_calibrate.py measured its
#               nearest neighbour naming the right country only ~41% of the time
#               on 1979 rounds, with the same-country and different-country
#               distance distributions almost fully overlapping.
#   streetclip  ViT-L/14 fine-tuned for geolocation on Street View imagery.
#               Bigger and slower, but trained for exactly this question.
#
# Each backend keeps its own collections, so switching does not mix vectors from
# two different spaces. Build the new ones with `python indexador.py`.
RAG_EMBEDDING = os.getenv("RAG_EMBEDDING", "openclip").strip().lower()
STREETCLIP_MODEL = os.getenv("STREETCLIP_MODEL", "geolocal/StreetCLIP")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vetores_db")
RAG_DISTANCE_THRESHOLD = float(
    os.getenv("RAG_DISTANCE_THRESHOLD")
    or {"openclip": 0.65, "streetclip": 0.10}.get(RAG_EMBEDDING, 0.65))


def collection_names() -> tuple[str, str]:
    """(scene collection, car-crop collection) for the active backend."""
    if RAG_EMBEDDING == "openclip":
        return "geoguessr_rondas", "geoguessr_carmeta"
    suffix = RAG_EMBEDDING.replace("-", "_")
    return f"geoguessr_rondas_{suffix}", f"geoguessr_carmeta_{suffix}"


class StreetCLIPEmbeddingFunction(EmbeddingFunction):
    """Chroma image embedding function backed by a HuggingFace CLIP checkpoint.

    Subclassing Chroma's EmbeddingFunction is what makes queries work: the base
    class supplies embed_query() and the validation wrapper. A plain class with
    __call__ is accepted on upsert but fails on query with "no attribute
    embed_query", which silently leaves retrieval returning nothing.

    The model is loaded lazily, so importing rag.py stays cheap for the analysis
    scripts that never touch the index.
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or STREETCLIP_MODEL
        self.device = device
        self._model = None
        self._processor = None

    @staticmethod
    def name() -> str:
        return "streetclip"

    def default_space(self) -> str:
        return "cosine"

    def get_config(self) -> dict:
        return {"model_name": self.model_name, "device": self.device}

    @staticmethod
    def build_from_config(config: dict) -> "StreetCLIPEmbeddingFunction":
        return StreetCLIPEmbeddingFunction(
            model_name=config.get("model_name"), device=config.get("device"))

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        say(f"[rag] loading {self.model_name} on {self.device}...")
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_name)

    def __call__(self, input):
        self._ensure()
        import torch
        images = []
        for item in input:
            if isinstance(item, Image.Image):
                images.append(item.convert("RGB"))
            else:
                images.append(Image.fromarray(np.asarray(item).astype("uint8")).convert("RGB"))
        with torch.no_grad():
            batch = self._processor(images=images, return_tensors="pt").to(self.device)
            feats = self._model.get_image_features(**batch)
            # transformers 5 returns an output object here; 4.x returned the tensor.
            if not isinstance(feats, torch.Tensor):
                feats = getattr(feats, "pooler_output", None)
                if feats is None:
                    raise RuntimeError("CLIP image features missing from model output")
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().tolist()


def build_embedding_function():
    if RAG_EMBEDDING == "streetclip":
        return StreetCLIPEmbeddingFunction()
    if RAG_EMBEDDING != "openclip":
        say(f"[rag] unknown RAG_EMBEDDING={RAG_EMBEDDING!r}, falling back to openclip")
    return OpenCLIPEmbeddingFunction()


def _get_chroma_collection_locked():
    global _chroma_client, _chroma_collection, _chroma_car_collection
    if _chroma_client is None:
        scene_name, car_name = collection_names()
        try:
            _chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
            embedding_function = build_embedding_function()
            _chroma_collection = _chroma_client.get_collection(
                name=scene_name, embedding_function=embedding_function)
            _chroma_car_collection = _chroma_client.get_collection(
                name=car_name, embedding_function=embedding_function)
        except Exception as e:
            say(f"[rag] no index for {RAG_EMBEDDING} ({scene_name}): {e}. "
                f"Build it with: python indexador.py")
            _chroma_collection = None
            _chroma_car_collection = None
    return _chroma_collection, _chroma_car_collection

# Maximum embedding distance for a retrieved round to be shown to the model.
# Measure it, do not guess it: `python rag_calibrate.py` replays indexed rounds
# against the index (excluding each round itself) and prints the separation
# between same-country and different-country matches per threshold.
#
# The value is encoder-specific, which is why it is a per-encoder default rather
# than one number. Measured on 1550 rounds:
#   openclip    no threshold separates anything. Same-country neighbours sit at a
#               median distance of 0.099 and different-country ones at 0.118, so
#               0.65 (show everything) loses nothing that a tighter value keeps.
#   streetclip  0.10. Retrieval precision alone does not predict end-to-end value:
#               0.15 scores better in isolation (78.5% precision, 99.5% coverage)
#               but measured worse in the pipeline, losing 11 countries and
#               gaining 2 over 150 rounds. The rounds it lost were ones the model
#               had right and the extra reference dragged to a neighbour
#               (Ireland to UK, Nepal to India, Peru to Ecuador). Showing more
#               references anchors the model away from its own better judgement.
_DEFAULT_THRESHOLDS = {"openclip": 0.65, "streetclip": 0.10}


def _default_threshold() -> float:
    return _DEFAULT_THRESHOLDS.get(RAG_EMBEDDING, 0.65)


# --- What gets remembered -------------------------------------------------
# One definition, used by both the live auto-index and indexador.py, which used
# to disagree (2000 km + country_hit here versus a flat 2500 km there, and two
# different "perfect" cutoffs). Keeping a round the bot got badly wrong teaches
# the index that a landscape looks like the wrong country.
RAG_INDEX_MAX_ERROR_KM = 2000.0   # keep a miss only if it was at least this close
RAG_PERFECT_MAX_ERROR_KM = 100.0  # flagged to the model as a high-quality reference


def is_rag_worthy(error_km: float | None, country_hit: bool | None) -> bool:
    if error_km is None:
        return False
    return bool(country_hit) or error_km < RAG_INDEX_MAX_ERROR_KM


def is_perfect_reference(error_km: float | None, country_hit: bool | None) -> bool:
    return (error_km is not None and error_km < RAG_PERFECT_MAX_ERROR_KM
            and bool(country_hit))


def build_rag_examples(current_image_path: Path, max_examples: int = 3,
                       exclude_ids: set[str] | None = None) -> tuple[str, list[str], str | None, float | None, dict[str, float]]:
    """exclude_ids: ChromaDB ids (screenshot stems) that must not be returned —
    used by benchmark.py so a held-out round never matches itself."""
    collection, collection_car = get_chroma_collection()
    if not collection:
        return "", [], None, None, {}
    
    try:
        img = Image.open(current_image_path).convert("RGB")
        img_array = np.array(img)
        w, h = img.size
        car_array = (np.array(img.crop((0, int(h * 0.70), w, h)))
                     if feature_on("rag_car") else None)

        # Fetch 4x more results than needed so the country-cap can enforce diversity.
        fetch_n = max(max_examples * 4, 12)
        with _chroma_lock:
            results = collection.query(
                query_images=[img_array],
                n_results=fetch_n,
                include=["distances", "metadatas"]
            )

        raw_matches = results['metadatas'][0] if results and results.get('metadatas') else []
        raw_distances = results['distances'][0] if results and results.get('distances') else []
        raw_ids = results['ids'][0] if results and results.get('ids') else [''] * len(raw_matches)

        # Apply distance filter then country-cap (max 2 results per country) to prevent
        # a single over-represented country (e.g. Philippines) from dominating the top-k.
        _country_seen: dict[str, int] = {}
        matches: list = []
        distances: list = []
        for rid, m, d in zip(raw_ids, raw_matches, raw_distances):
            if exclude_ids and rid in exclude_ids:
                continue
            if d > RAG_DISTANCE_THRESHOLD:
                continue
            c = m.get("country", "?")
            if _country_seen.get(c, 0) >= 2:
                continue
            _country_seen[c] = _country_seen.get(c, 0) + 1
            matches.append(m)
            distances.append(d)
            if len(matches) >= max_examples:
                break

        # Wrong-continent veto: if one country is on a completely different continent from
        # the other(s), it is likely a visual false-match (e.g. Philippines when others are SA).
        # Remove it from the match list so it doesn't bias the country guess.
        if len(matches) >= 2:
            _match_conts = [continent_of(m.get("country", "")) for m in matches]
            _valid_conts = [c for c in _match_conts if c]
            if len(_valid_conts) >= 2:
                from collections import Counter as _Counter
                _cont_counts = _Counter(_valid_conts)
                _majority_cont = _cont_counts.most_common(1)[0][0]
                _filtered_matches = []
                _filtered_distances = []
                for m, d, mc in zip(matches, distances, _match_conts):
                    if mc and mc != _majority_cont:
                        say(f"  [rag-veto] {m.get('country')} ({mc}) vetoed — majority continent is {_majority_cont}")
                    else:
                        _filtered_matches.append(m)
                        _filtered_distances.append(d)
                if _filtered_matches:
                    matches = _filtered_matches
                    distances = _filtered_distances

        countries = []
        top_country: str | None = None
        top_distance: float | None = None
        scores: dict[str, float] = {}

        lines = [
            "Reference: past Street View rounds whose image looked visually similar to the current "
            "one. These are NOT ground truth — two different countries with similar landscapes "
            "(e.g. UK vs New Zealand rolling hills, Ukraine vs Russia steppe, Indonesia vs "
            "Philippines tropics) often produce the same match. Treat this as one weak signal among "
            "many. Lower [dist] = more similar image, but similarity ≠ same country. Cross-check "
            "against concrete clues (OCR text, alphabet, plate format, signage language, unique "
            "bollards/poles). If the reference says country X but the image has text/signs pointing "
            "to country Y, trust the text.",
        ]
        rcount = 0
        for i, (match, dist) in enumerate(zip(matches, distances), 1):
            country = match.get("country", "?")
            if country != "?" and country not in countries:
                countries.append(country)
            if country != "?":
                # Weighted vote: closer match → bigger score. Small epsilon avoids /0.
                scores[country] = scores.get(country, 0.0) + 1.0 / max(float(dist), 0.01)
            if top_country is None and country != "?":
                top_country = country
                top_distance = float(dist)
            admin1 = match.get("admin1", "?")
            lat = match.get("lat", 0.0)
            lon = match.get("lon", 0.0)

            # Recompensa o bot se este for um Match histórico de precisão milimétrica
            is_perfect = match.get("is_perfect", False)
            perfect_tag = " (past guess <100km, country correct — high-quality reference)" if is_perfect else ""

            lines.append(f"{i}. {country} / {admin1} at ({lat:.3f}, {lon:.3f}){perfect_tag}. [dist: {dist:.2f}]")
            rcount += 1
            
        # Pesquisa separada do carro
        if collection_car and feature_on("rag_car"):
            with _chroma_lock:
                car_results = collection_car.query(
                    query_images=[car_array],
                    n_results=3 if exclude_ids else 1,
                    include=["distances", "metadatas"]
                )
            car_matches = car_results['metadatas'][0] if car_results and car_results.get('metadatas') else []
            car_distances = car_results['distances'][0] if car_results and car_results.get('distances') else []
            car_ids = car_results['ids'][0] if car_results and car_results.get('ids') else [''] * len(car_matches)
            if exclude_ids:
                keep = [i for i, cid in enumerate(car_ids)
                        if cid not in exclude_ids and cid.removesuffix('_car') not in exclude_ids]
                car_matches = [car_matches[i] for i in keep]
                car_distances = [car_distances[i] for i in keep]
            if car_matches and car_distances:
                match = car_matches[0]
                dist = car_distances[0]
                if dist > RAG_DISTANCE_THRESHOLD:
                    say(f"  [rag] Rejeitado car meta ({match.get('country')}): distância {dist:.3f} > {RAG_DISTANCE_THRESHOLD}")
                else:
                    country = match.get("country", "?")
                    if country != "?" and country not in countries:
                        countries.append(country)
                    if country != "?":
                        scores[country] = scores.get(country, 0.0) + 1.0 / max(float(dist), 0.01)
                    admin1 = match.get("admin1", "?")
                    reasoning = match.get("reasoning", "").replace("\n", " ")[:80]
                    # Car Meta também avisa se é um hit perfeito na base de dados
                    is_perfect = match.get("is_perfect", False)
                    perfect_tag = " (past guess <100km, country correct — high-quality reference)" if is_perfect else ""
                    lines.append(f"\nCar Meta Match: Similar vehicle/camera structure found in {country} / {admin1}{perfect_tag}. Past clue: {reasoning} [dist: {dist:.2f}]")

        if rcount == 0 and len(countries) == 0:
            return "", [], None, None, {}

        if scores:
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            score_str = ", ".join(f"{c}: {s:.2f}" for c, s in ranked)
            lines.append(f"\nWeighted RAG country scores (higher = more visual evidence): {score_str}.")

        # Frequency penalty: warn when a RAG country appears very rarely in recent play.
        # A country with few recent occurrences may be over-represented visually in the
        # index (e.g. Philippines has tropical images that match many SA scenes).
        if countries:
            try:
                _recent = load_rounds()[-100:]
                _recent_counts: dict[str, int] = {}
                for _r in _recent:
                    _c = (_r.get("actual") or {}).get("country", "")
                    if _c:
                        _recent_counts[_c] = _recent_counts.get(_c, 0) + 1
                _rare = []
                for _c in countries:
                    _freq = _recent_counts.get(_c, 0)
                    if _freq <= 1:
                        _rare.append(f"{_c} (appeared {_freq}x in last 100 rounds)")
                if _rare:
                    lines.append(
                        f"\nRAG FREQUENCY WARNING: The following RAG-suggested countries "
                        f"are rare in recent play and may be false visual matches — "
                        f"require strong confirming evidence before choosing them: "
                        + "; ".join(_rare)
                    )
            except Exception:
                pass

        return "\n".join(lines), countries, top_country, top_distance, scores
    except Exception as e:
        say(f"  [rag error] {e}")
        return "", [], None, None, {}

def get_country_metas(countries: list[str]) -> str:
    # Passa a procurar primeiro na sub-pasta de Resumos para poupar contexto
    meta_dir = PROJECT_DIR / "metas_resumidas"
    fallback_dir = PROJECT_DIR / "metas"
    
    if not fallback_dir.exists():
        return ""
    
    # Country name aliases for meta file lookup (handles naming variants)
    _META_LOOKUP_ALIASES = {
        "türkiye": "turkey",
        "turkiye": "turkey",
        "russian federation": "russian federation",  # explicit file exists
        "korea, republic of": "south korea",
        "bolivia, plurinational state of": "bolivia",
        "venezuela, bolivarian republic of": "venezuela",
        "taiwan, province of china": "taiwan",
        "iran, islamic republic of": "iran",
    }
    tips = []
    for country in countries:
        country_lc = country.lower()
        # Apply meta lookup alias (e.g. Türkiye → turkey.txt)
        lookup_name = _META_LOOKUP_ALIASES.get(country_lc, country_lc)
        safe_name = lookup_name.replace(" ", "_")
        c_file_space = f"{lookup_name}.txt"
        c_file_underscore = f"{safe_name}.txt"

        # 1. Tentamos obter o resumo concentrado
        p = meta_dir / c_file_space
        if not p.exists(): p = meta_dir / c_file_underscore

        # 2. Fazemos Fallback para o artigo enorme se o resumo não estiver feito
        if not p.exists(): p = fallback_dir / c_file_space
        if not p.exists(): p = fallback_dir / c_file_underscore

        if p.exists():
            tips.append(f"--- META TIPS FOR {country.upper()} ---")
            tips.append(p.read_text(encoding="utf-8").strip())
    
    return "\n\n".join(tips) if tips else ""


# ===========================================================================
# Live round (browser side)
# ===========================================================================
def index_round_into_rag(shot_path: Path, actual_entry: dict | None, reasoning: str,
                         error_km: float | None, country_hit: bool | None) -> None:
    """Continuous learning: store this round's image + true location in ChromaDB.
    Rounds the bot got badly wrong are skipped so they don't pollute retrieval."""
    if not actual_entry or error_km is None:
        return
    try:
        coll, coll_car = get_chroma_collection()
        if not coll:
            return
        if not is_rag_worthy(error_km, country_hit):
            say(f"  [rag-skip] not indexing: wrong country + {error_km:.0f}km error")
            return
        start_idx = time.time()
        img = Image.open(shot_path).convert("RGB")
        img_array = np.array(img)
        w, h = img.size
        car_array = (np.array(img.crop((0, int(h * 0.70), w, h)))
                     if feature_on("rag_car") else None)
        meta = [{
            "country": actual_entry.get("country", "?"),
            "admin1": actual_entry.get("admin1", "?"),
            "lat": float(actual_entry.get("latitude", 0)),
            "lon": float(actual_entry.get("longitude", 0)),
            "reasoning": reasoning,
            "is_perfect": is_perfect_reference(error_km, country_hit),
        }]
        with _chroma_lock:
            coll.upsert(ids=[shot_path.stem], images=[img_array], metadatas=meta)
            if coll_car is not None and car_array is not None:
                coll_car.upsert(ids=[f"{shot_path.stem}_car"], images=[car_array], metadatas=meta)
        say(f"  [auto-index] ronda guardada na memoria ({time.time()-start_idx:.1f}s)")
    except Exception as e:
        say(f"  [auto-index erro] {e}")
