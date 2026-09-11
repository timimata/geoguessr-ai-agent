import json
import urllib.request
from pathlib import Path

def download_countries_geojson():
    dest = Path("countries.geojson")
    if dest.exists():
        print("O ficheiro countries.geojson já existe.")
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
    download_countries_geojson()