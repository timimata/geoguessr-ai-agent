"""
replay_errors.py — Lista os piores erros do log para revisão manual e adição ao RAG.

Uso:
    python replay_errors.py                  # mostra os 60 piores erros (>3000km)
    python replay_errors.py --threshold 5000 # só erros >5000km
    python replay_errors.py --add           # modo interativo: abre cada screenshot e pergunta se deve adicionar ao RAG
    python replay_errors.py --top 30        # limitar a 30 entradas
"""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()
LOG_FILE = PROJECT_DIR / "log.json"


def load_log() -> list[dict]:
    if not LOG_FILE.exists():
        print("log.json não encontrado.")
        sys.exit(1)
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def find_worst_errors(data: list[dict], threshold: float, top: int) -> list[dict]:
    errors = [r for r in data if r.get("error_km") and r["error_km"] >= threshold]
    errors.sort(key=lambda r: r["error_km"], reverse=True)
    return errors[:top]


def get_actual_country(r: dict) -> str:
    actual = r.get("actual") or {}
    return actual.get("country") or actual.get("cc") or "?"


def print_summary(errors: list[dict]) -> None:
    actual_counts = Counter(get_actual_country(r) for r in errors)
    guess_counts = Counter(r["guess"]["country"] for r in errors)

    print(f"\n{'='*70}")
    print(f"  {len(errors)} RONDAS COM ERRO ELEVADO")
    print(f"{'='*70}")
    print(f"\nPaíses REAIS mais confundidos:")
    for c, n in actual_counts.most_common(10):
        print(f"  {c:<35} {n}x")
    print(f"\nPaíses MAIS ADIVINHADOS (errado):")
    for c, n in guess_counts.most_common(10):
        print(f"  {c:<35} {n}x")

    print(f"\n{'='*70}")
    print(f"  LISTA COMPLETA (ordenada por erro desc)")
    print(f"{'='*70}")
    print(f"{'#':<4} {'Actual':<30} {'Guessed':<30} {'Error':>8}  {'Conf':>5}  Screenshot")
    print(f"{'-'*130}")
    for i, r in enumerate(errors, 1):
        actual = get_actual_country(r)
        guessed = r["guess"]["country"]
        err = r["error_km"]
        conf = r["guess"].get("confidence", 0)
        ts = r.get("timestamp", "?")
        screenshot = r.get("screenshot", "")
        print(f"{i:<4} {actual:<30} {guessed:<30} {err:>8.0f}  {conf:>5.2f}  {screenshot}")


def open_screenshot(screenshot_path: str) -> None:
    full_path = PROJECT_DIR / screenshot_path
    if not full_path.exists():
        print(f"  [aviso] screenshot não encontrado: {full_path}")
        return
    try:
        if sys.platform == "win32":
            os.startfile(str(full_path))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(full_path)])
        else:
            subprocess.run(["xdg-open", str(full_path)])
        print(f"  Aberto: {full_path.name}")
    except Exception as e:
        print(f"  Erro ao abrir: {e}")


def add_to_rag(r: dict) -> bool:
    """Tenta adicionar a ronda ao RAG com o país correto."""
    try:
        import chromadb
        from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
        import numpy as np
        import cv2

        screenshot = r.get("screenshot", "")
        if not screenshot:
            print("  Sem screenshot — não é possível adicionar ao RAG.")
            return False

        img_path = PROJECT_DIR / screenshot
        if not img_path.exists():
            print(f"  Screenshot não encontrado: {img_path}")
            return False

        actual = r.get("actual") or {}
        country = actual.get("country") or get_actual_country(r)
        admin1 = actual.get("admin1", "?")
        lat = actual.get("latitude", 0.0)
        lon = actual.get("longitude", 0.0)
        ts = r.get("timestamp", "unknown")

        client = chromadb.PersistentClient(path=str(PROJECT_DIR / "vetores_db"))
        ef = OpenCLIPEmbeddingFunction()
        collection = client.get_or_create_collection(
            name="geoguessr_rondas",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )

        from PIL import Image as PILImage
        pil_img = PILImage.open(img_path).convert("RGB")
        img_array = np.array(pil_img)  # uint8 HxWx3, same format as bot

        doc_id = f"replay_{ts}_{country.replace(' ','_')}"

        collection.upsert(
            ids=[doc_id],
            images=[img_array],
            metadatas=[{
                "country": country,
                "admin1": admin1,
                "lat": lat,
                "lon": lon,
                "timestamp": ts,
                "source": "replay_correction",
                "is_perfect": False,
                "error_km": 0.0,
            }]
        )
        print(f"  [RAG] Adicionado: {country}/{admin1} ({lat:.3f},{lon:.3f}) — id={doc_id}")
        return True

    except Exception as e:
        print(f"  [RAG erro] {e}")
        import traceback
        traceback.print_exc()
        return False


def interactive_mode(errors: list[dict]) -> None:
    added = 0
    skipped = 0
    print(f"\nMODO INTERATIVO — {len(errors)} rondas para rever")
    print("Comandos: [a]=adicionar ao RAG  [s]=saltar  [q]=sair\n")

    for i, r in enumerate(errors, 1):
        actual = get_actual_country(r)
        guessed = r["guess"]["country"]
        err = r["error_km"]
        conf = r["guess"].get("confidence", 0)
        screenshot = r.get("screenshot", "")
        ts = r.get("timestamp", "?")

        print(f"\n[{i}/{len(errors)}] {actual} -> adivinhado: {guessed}  |  {err:.0f}km  |  conf={conf:.2f}  |  {ts}")

        open_screenshot(screenshot)

        while True:
            cmd = input("  > [a]dicionar / [s]altar / [q]sair: ").strip().lower()
            if cmd in ("a", ""):
                if add_to_rag(r):
                    added += 1
                break
            elif cmd == "s":
                skipped += 1
                break
            elif cmd == "q":
                print(f"\nSaída — adicionadas: {added}, saltadas: {skipped}")
                return
            else:
                print("  Comando inválido. Use 'a', 's' ou 'q'.")

    print(f"\nFim — adicionadas: {added}, saltadas: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay dos piores erros do bot GeoGuessr")
    parser.add_argument("--threshold", type=float, default=3000,
                        help="Threshold mínimo de erro em km (default: 3000)")
    parser.add_argument("--top", type=int, default=80,
                        help="Máximo de rondas a mostrar (default: 80)")
    parser.add_argument("--add", action="store_true",
                        help="Modo interativo: revisar e adicionar ao RAG")
    parser.add_argument("--recent", type=int, default=0,
                        help="Limitar às últimas N rondas do log (0=todas)")
    parser.add_argument("--country", type=str, default="",
                        help="Filtrar por país real específico (ex: 'South Africa')")
    args = parser.parse_args()

    data = load_log()
    if args.recent > 0:
        data = data[-args.recent:]
        print(f"Usando as últimas {len(data)} rondas do log.")
    else:
        print(f"Log total: {len(data)} rondas.")

    errors = find_worst_errors(data, args.threshold, args.top)

    if args.country:
        errors = [r for r in errors if args.country.lower() in get_actual_country(r).lower()]
        print(f"Filtrado para '{args.country}': {len(errors)} rondas.")

    if not errors:
        print(f"Nenhuma ronda com erro >= {args.threshold}km encontrada.")
        return

    print_summary(errors)

    if args.add:
        interactive_mode(errors)


if __name__ == "__main__":
    main()
