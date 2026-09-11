import os
from pathlib import Path

def clean_metas():
    meta_dir = Path("metas")
    if not meta_dir.exists():
        return

    # Mapear ficheiros. Dar prioridade aos que têm "PLONKIT SCRAPED" no conteúdo.
    # Vamos agrupar os ficheiros pelo nome limpo (ex: "united states", "costa rica")
    files_by_country = {}
    
    for f in meta_dir.glob("*.txt"):
        name = f.stem.lower().replace("-", " ").replace("_", " ")
        if name not in files_by_country:
            files_by_country[name] = []
        files_by_country[name].append(f)
        
    deleted = 0
    for country, files in files_by_country.items():
        if len(files) > 1:
            # Se houver mais de um ficheiro para o mesmo país (ex: costa_rica.txt e costa rica.txt)
            # Encontrar qual tem a versão do PLONKIT, ou se não houver, a maior versão
            best_file = None
            plonkit_found = False
            for f in files:
                content = f.read_text(encoding="utf-8")
                if "PLONKIT SCRAPED" in content:
                    best_file = f
                    plonkit_found = True
                    break
            
            # Se não houver versão Plonkit, escolher o ficheiro com espaço como principal
            if not best_file:
                best_file = next((f for f in files if " " in f.name), files[0])
                
            # Apagar os outros
            for f in files:
                if f != best_file:
                    f.unlink()
                    deleted += 1
                    
            # Renomear o melhor ficheiro para ter espaços (formato universal)
            new_path = meta_dir / f"{country}.txt"
            if best_file.name != new_path.name:
                best_file.rename(new_path)
                
    print(f"Limpeza concluída! {deleted} duplicações eliminadas. Ficas com apenas 1 ficheiro definitivo por país.")

if __name__ == "__main__":
    clean_metas()