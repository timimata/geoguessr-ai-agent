import json
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from pathlib import Path
from PIL import Image
import numpy as np

def main():
    print("A iniciar o ChromaDB e o modelo CLIP...")
    client = chromadb.PersistentClient(path="./vetores_db")
    embedding_function = OpenCLIPEmbeddingFunction()
    
    collection = client.get_or_create_collection(
        name="geoguessr_rondas",
        embedding_function=embedding_function
    )
    
    collection_car = client.get_or_create_collection(
        name="geoguessr_carmeta",
        embedding_function=embedding_function
    )
    
    log_path = Path("log.json")
    if not log_path.exists():
        print("log.json não encontrado.")
        return
        
    rounds = json.loads(log_path.read_text(encoding="utf-8"))
    print(f"A processar histórico de {len(rounds)} rondas...")
    
    existing_ids = set(collection.get(include=[])['ids'])
    print(f"A base de dados já tem {len(existing_ids)} imagens (paisagem). A saltar o que já está indexado para ser mais rápido...")
    
    count = 0
    for r in rounds:
        if r.get("screenshot") and r.get("actual"):
            # Apenas memorizamos rondas em que o erro foi inferior a 2500km
            # Erros maiores que isso significam que o bot não tinha ideia/foi atirado para a sorte (ex: Rússia vs Argentina), e isso polui o RAG.
            if r.get("error_km", 9999) > 2500:
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
                    car_box = (0, int(h * 0.70), w, h)
                    car_img = img.crop(car_box)
                    car_array = np.array(car_img)

                    metadatas = [{
                        "country": r["actual"].get("country", "?"),
                        "admin1": r["actual"].get("admin1", "?"),
                        "lat": float(r["actual"].get("latitude", 0)),
                        "lon": float(r["actual"].get("longitude", 0)),
                        "reasoning": str(r.get("guess", {}).get("reasoning", "")),
                        "is_perfect": r.get("error_km", 9999) < 250 and bool(r.get("country_hit"))
                    }]

                    # Usamos o nome do ficheiro (ex: round_20231024_1200.png) como ID único
                    collection.upsert(
                        ids=[img_path.stem],
                        images=[img_array],
                        metadatas=metadatas
                    )
                    
                    # Id único para o car meta, usamos um sufixo
                    collection_car.upsert(
                        ids=[f"{img_path.stem}_car"],
                        images=[car_array],
                        metadatas=metadatas
                    )
                    count += 1
                    
                    if count % 10 == 0:
                        print(f"  ... [Progresso] {count} imagens indexadas até agora ...")
                except Exception as e:
                    print(f"Erro na ronda {r.get('round', '?')}: {e}")
                    
    print(f"Indexação concluída! {count} imagens guardadas na base de dados (vetores_db).")

if __name__ == "__main__":
    main()
