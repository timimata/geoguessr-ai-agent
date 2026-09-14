import json
import urllib.request
from pathlib import Path

DEST = Path(__file__).parent / "countries.geojson"


def download_countries_geojson(force: bool = False):
    dest = DEST
    if dest.exists() and not force:
        print(f"O ficheiro {dest.name} já existe ({dest.stat().st_size / 1e6:.1f} MB).")
        print("Use --force para voltar a descarregar.")
        return

    # Usar um set público, leve e gratuito com polígonos de fronteira
    url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    print("A transferir ficheiro de Fronteiras dos Países (GeoJSON)...")
    
    try:
        req = urllib.request.urlopen(url)
        data = req.read()
        dest.write_bytes(data)
        
        # Testar se o JSON é válido
        geojson = json.loads(data)
        print(f"Sucesso! Descarregadas as fronteiras de {len(geojson.get('features', []))} países.")
    except Exception as e:
        print(f"Erro ao descarregar as fronteiras: {e}")

if __name__ == "__main__":
    import sys
    download_countries_geojson(force="--force" in sys.argv)