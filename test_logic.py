import sys
import json
import numpy as np
from pathlib import Path
from shapely.geometry import Point, shape
from shapely.ops import nearest_points

import bot

def test_clamping():
    print("--- TESTE DE CLAMPING (SHAPELY) ---")
    if not bot._COUNTRY_POLYGONS:
        print("Erro: Polígonos de países não carregados.")
        return

    # Um palpite (guess) de um país, mas as coordenadas caem no oceano ao largo.
    # Exemplo prático: Portugal. Palpite dado (lat 38.0, lon -12.0) - No meio do mar.
    # Coordenadas corretas aproximadas para o centro seriam (lat 39.5, lon -8.0).
    test_country_iso = "PT"
    ocean_lat, ocean_lon = 38.0, -12.0
    
    print(f"Tentando dar clamping ao palpite em {test_country_iso} com as coordenadas ({ocean_lat}, {ocean_lon})...")
    
    point_guess = Point(ocean_lon, ocean_lat)
    target_geom = None
    
    # Procurar a geometria do país usando o ISO
    for feature in bot._COUNTRY_POLYGONS:
        props = feature.get("properties", {})
        iso2 = props.get("ISO_A2", "").upper()
        if iso2 == test_country_iso:
            target_geom = shape(feature["geometry"])
            break
            
    if not target_geom:
        print(f"País com ISO '{test_country_iso}' não encontrado nos dados geojson.")
    else:
        if not target_geom.contains(point_guess):
            pol_pt, _ = nearest_points(target_geom, point_guess)
            clamped_lon, clamped_lat = pol_pt.x, pol_pt.y
            print(f"Sucesso: Ponto movido de ({ocean_lat:.4f}, {ocean_lon:.4f}) para dentro de {test_country_iso}: ({clamped_lat:.4f}, {clamped_lon:.4f}).")
        else:
            print(f"O ponto original já estava dentro de {test_country_iso}.")

def test_rag():
    print("\n--- TESTE DE RAG (CHROMADB) ---")
    # Usa um screenshot existente na pasta screenshots se houver
    screenshots_dir = Path("screenshots")
    if not screenshots_dir.exists() or not any(screenshots_dir.iterdir()):
        print("Aviso: Nenhuma imagem em screenshots/ para testar.")
        return
        
    test_img = next(screenshots_dir.glob("*.png"), None)
    if not test_img:
        print("Nenhum ficheiro .png encontrado para teste RAG.")
        return
        
    print(f"A testar RAG com a imagem {test_img} usando threshold de 0.8...")
    result_text, countries, top_c, top_d, _scores = bot.build_rag_examples(test_img, max_examples=3)
    
    print("\n[Resultados do RAG]")
    if result_text:
        print(result_text)
        print(f"-> Países listados nas dicas: {countries}")
    else:
        print("Nenhuma tip encontrada (provavelmente todas ficaram foram rejeitadas pelo limite de distância ou base de dados vazia).")

if __name__ == "__main__":
    bot.WORLD_BORDERS = "countries.geojson"
    try:
        with open(bot.WORLD_BORDERS, 'r', encoding='utf-8') as f:
            data = json.load(f)
            bot._COUNTRY_POLYGONS = data.get("features", [])
    except Exception as e:
        print(f"Erro ao carregar {bot.WORLD_BORDERS}: {e}")

    test_clamping()
    test_rag()
