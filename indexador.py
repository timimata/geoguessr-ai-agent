"""indexador.py - (re)build the vector index from the round log.

Which encoder is used comes from RAG_EMBEDDING (see rag.py); each backend gets
its own collections, so switching encoders means running this again rather than
mixing two vector spaces.

    python indexador.py                      # fill in whatever is missing
    RAG_EMBEDDING=streetclip python indexador.py
    python indexador.py --limit 300          # stop after 300 new rounds
"""
import argparse
import sys
import time
from pathlib import Path

import chromadb
import numpy as np
from PIL import Image

# Single source of truth for what is worth remembering, shared with the live
# auto-index so the two can never drift apart again.
from config import feature_on
from rag import (RAG_EMBEDDING, VECTOR_DB_PATH, build_embedding_function,
                 collection_names, is_perfect_reference, is_rag_worthy)
from storage import load_rounds, log_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after N new rounds")
    args = ap.parse_args()

    scene_name, car_name = collection_names()
    print(f"Encoder: {RAG_EMBEDDING} -> {scene_name} / {car_name}")
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    embedding_function = build_embedding_function()

    collection = client.get_or_create_collection(
        name=scene_name, embedding_function=embedding_function)
    # The car-crop search is off by default (it measured worse than always
    # answering "United States"), and indexing those crops doubles the run.
    index_car = feature_on("rag_car")
    collection_car = (client.get_or_create_collection(
        name=car_name, embedding_function=embedding_function) if index_car else None)
    if not index_car:
        print("Car-crop index desativado (FEATURES['rag_car']).")
    
    rounds = load_rounds()
    if not rounds:
        print(f"{log_path().name} está vazio ou não existe.")
        return
    print(f"A processar histórico de {len(rounds)} rondas...")
    
    existing_ids = set(collection.get(include=[])['ids'])
    print(f"A base de dados já tem {len(existing_ids)} imagens (paisagem). A saltar o que já está indexado para ser mais rápido...")
    
    count = 0
    started = time.time()
    for r in rounds:
        if args.limit and count >= args.limit:
            print(f"A parar em {count} rondas novas (--limit).")
            break
        if r.get("screenshot") and r.get("actual"):
            # Mesma política do auto-index em rag.py (is_rag_worthy): guardar uma
            # ronda que o bot errou por muito ensina o índice que uma paisagem se
            # parece com o país errado.
            if not is_rag_worthy(r.get("error_km"), r.get("country_hit")):
                continue

            img_path = Path(r["screenshot"])
            
            if img_path.stem in existing_ids:
                continue # Salta a imagem se já estiver no ChromaDB!
                
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_array = np.array(img)
                    
                    # Car Meta Crop (últimos 30% da altura)
                    w, h = img.size
                    car_array = (np.array(img.crop((0, int(h * 0.70), w, h)))
                                 if index_car else None)

                    metadatas = [{
                        "country": r["actual"].get("country", "?"),
                        "admin1": r["actual"].get("admin1", "?"),
                        "lat": float(r["actual"].get("latitude", 0)),
                        "lon": float(r["actual"].get("longitude", 0)),
                        "reasoning": str(r.get("guess", {}).get("reasoning", "")),
                        "is_perfect": is_perfect_reference(r.get("error_km"), r.get("country_hit"))
                    }]

                    # Usamos o nome do ficheiro (ex: round_20231024_1200.png) como ID único
                    collection.upsert(
                        ids=[img_path.stem],
                        images=[img_array],
                        metadatas=metadatas
                    )
                    
                    # Id único para o car meta, usamos um sufixo
                    if index_car:
                        collection_car.upsert(
                            ids=[f"{img_path.stem}_car"],
                            images=[car_array],
                            metadatas=metadatas,
                        )
                    count += 1
                    
                    if count % 10 == 0:
                        rate = count / max(time.time() - started, 1e-6)
                        print(f"  ... {count} indexadas ({rate:.1f}/s) ...", flush=True)
                except Exception as e:
                    print(f"Erro na ronda {r.get('round', '?')}: {e}")
                    
    print(f"Indexação concluída! {count} imagens guardadas na base de dados (vetores_db).")

if __name__ == "__main__":
    main()
