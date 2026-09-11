import asyncio
import base64
import json
import math
import os
import random
import hashlib
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path

import reverse_geocoder as rg
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw
from playwright.async_api import async_playwright, Page
from playwright_stealth import Stealth
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
import numpy as np
import cv2

# Sistema de Fronteiras (Clamping Pós-Decisão)
try:
    from shapely.geometry import shape, Point
    from shapely.ops import nearest_points
    from pathlib import Path
    import json
    
    # Carregar o GeoJSON uma vez
    border_file = Path(__file__).parent / "countries.geojson"
    try:
        with open(border_file, "r", encoding="utf-8") as f:
            WORLD_BORDERS = json.load(f)
            # Dicionário de nome rápido para indexar as features (Geometrias dos países)
            _COUNTRY_POLYGONS = {}
            for feat in WORLD_BORDERS.get("features", []):
                props = feat["properties"]
                name = props.get("name", "").lower()
                iso_a3 = props.get("ISO3166-1-Alpha-3", "").lower()
                iso_a2 = props.get("ISO3166-1-Alpha-2", "").lower()

                poly = shape(feat["geometry"])
                if name: _COUNTRY_POLYGONS[name] = poly
                if iso_a3 and iso_a3 != "-99": _COUNTRY_POLYGONS[iso_a3] = poly
                if iso_a2 and iso_a2 != "-99": _COUNTRY_POLYGONS[iso_a2] = poly

            # Aliases para nomes alternativos mais usados pelo LLM/reverse_geocoder
            _COUNTRY_POLYGONS["united states"] = _COUNTRY_POLYGONS.get("united states of america")
            _COUNTRY_POLYGONS["usa"] = _COUNTRY_POLYGONS.get("united states of america")
            _COUNTRY_POLYGONS["uk"] = _COUNTRY_POLYGONS.get("united kingdom")
            _COUNTRY_POLYGONS["russian federation"] = _COUNTRY_POLYGONS.get("russia")
            _COUNTRY_POLYGONS["czechia"] = _COUNTRY_POLYGONS.get("czech republic") or _COUNTRY_POLYGONS.get("czechia")
            # Türkiye — GeoJSON may store as "turkey" or "türkiye"
            _COUNTRY_POLYGONS["türkiye"] = _COUNTRY_POLYGONS.get("turkey") or _COUNTRY_POLYGONS.get("türkiye")
            _COUNTRY_POLYGONS["turkey"] = _COUNTRY_POLYGONS.get("turkey") or _COUNTRY_POLYGONS.get("türkiye")
            print(f"[geo] loaded {len(_COUNTRY_POLYGONS)} country polygon keys from countries.geojson")
            
    except Exception as e:
        print(f"Erro ao ler countries.geojson: {e}")
        WORLD_BORDERS = None
except ImportError:
    print("Aviso: biblioteca 'shapely' em falta. O Country Clamping não estará ativo.")
    WORLD_BORDERS = None

_COUNTRY_ALIASES = {
    "usa": "united states", "us": "united states", "u.s.": "united states",
    "u.s.a.": "united states", "america": "united states",
    "united states of america": "united states",
    "uk": "united kingdom", "u.k.": "united kingdom", "britain": "united kingdom",
    "great britain": "united kingdom", "england": "united kingdom",
    "south korea": "korea, republic of",
    "korea, republic of": "Korea, Republic of",  # preserve lowercase 'of'
    "north korea": "korea, democratic people's republic of",
    "turkey": "türkiye", "turkiye": "türkiye",  # normalize to Unicode name used by reverse_geocoder
    "russia": "russian federation", "iran": "iran, islamic republic of",
    "taiwan": "taiwan, province of china", "bolivia": "bolivia, plurinational state of",
    "bolivia, plurinational state of": "Bolivia, Plurinational State of",  # preserve lowercase 'of'
    "venezuela": "venezuela, bolivarian republic of",
    "czechia": "czech republic",
    "ivory coast": "côte d'ivoire", "cote d'ivoire": "côte d'ivoire",
    "east timor": "timor-leste",
    "the netherlands": "netherlands", "holland": "netherlands",
    "burma": "myanmar",
    "cape verde": "cabo verde",
    "swaziland": "eswatini",
    "palestinian territory": "palestine", "palestinian territories": "palestine",
    "vatican": "vatican city", "the vatican": "vatican city",
    "macedonia": "north macedonia",
    "congo-kinshasa": "democratic republic of the congo", "dr congo": "democratic republic of the congo",
    "congo-brazzaville": "republic of the congo",
}

# Country → continent, for sanity checks (RAG consensus vs guess). Uses the
# aliased country name (lowercase). Keep in sync with reverse_geocoder output.
_COUNTRY_CONTINENT: dict[str, str] = {
    # Europe
    **{c: "Europe" for c in [
        "united kingdom","ireland","france","germany","spain","portugal","italy","netherlands",
        "belgium","switzerland","austria","poland","czechia","czech republic","slovakia","hungary",
        "romania","bulgaria","greece","serbia","croatia","slovenia","bosnia and herzegovina",
        "albania","north macedonia","montenegro","kosovo","ukraine","belarus","moldova","lithuania",
        "latvia","estonia","finland","sweden","norway","denmark","iceland","luxembourg","malta",
        "cyprus","andorra","monaco","liechtenstein","san marino","vatican city","russian federation",
    ]},
    # Asia
    **{c: "Asia" for c in [
        "china","japan","korea, republic of","korea, democratic people's republic of","mongolia",
        "taiwan, province of china","hong kong","macau","singapore","malaysia","indonesia","thailand",
        "vietnam","laos","cambodia","myanmar","philippines","brunei darussalam","timor-leste",
        "india","pakistan","bangladesh","nepal","bhutan","sri lanka","maldives","afghanistan",
        "kazakhstan","uzbekistan","kyrgyzstan","tajikistan","turkmenistan",
        "iran, islamic republic of","iraq","syria","jordan","lebanon","israel","palestine",
        "saudi arabia","yemen","oman","united arab emirates","qatar","bahrain","kuwait",
        "armenia","azerbaijan","georgia",
        "türkiye",  # Unicode name used by reverse_geocoder
    ]},
    # Africa
    **{c: "Africa" for c in [
        "morocco","algeria","tunisia","libya","egypt","sudan","south sudan","ethiopia","eritrea",
        "djibouti","somalia","kenya","uganda","tanzania","rwanda","burundi","south africa",
        "namibia","botswana","zimbabwe","zambia","malawi","mozambique","angola","lesotho","eswatini",
        "madagascar","mauritius","comoros","seychelles","nigeria","ghana","côte d'ivoire",
        "senegal","gambia","mali","burkina faso","niger","chad","cameroon","central african republic",
        "gabon","equatorial guinea","sao tome and principe","republic of the congo",
        "democratic republic of the congo","benin","togo","liberia","sierra leone","guinea",
        "guinea-bissau","cabo verde","mauritania",
    ]},
    # North America
    **{c: "North America" for c in [
        "united states","canada","mexico","guatemala","belize","honduras","el salvador","nicaragua",
        "costa rica","panama","cuba","jamaica","haiti","dominican republic","puerto rico","bahamas",
        "trinidad and tobago","barbados",
    ]},
    # South America
    **{c: "South America" for c in [
        "brazil","argentina","chile","uruguay","paraguay","bolivia, plurinational state of",
        "peru","ecuador","colombia","venezuela, bolivarian republic of","guyana","suriname",
    ]},
    # Oceania
    **{c: "Oceania" for c in [
        "australia","new zealand","papua new guinea","fiji","solomon islands","vanuatu","samoa",
        "tonga","micronesia","palau","kiribati","marshall islands","nauru","tuvalu",
    ]},
}


def continent_of(country: str) -> str | None:
    if not country:
        return None
    raw = country.strip().lower()
    c = _COUNTRY_ALIASES.get(raw, raw)
    result = _COUNTRY_CONTINENT.get(c)
    if result is None:
        # Try the aliased value lowercased (handles "Türkiye" → "türkiye" lookup)
        result = _COUNTRY_CONTINENT.get(c.lower())
    return result

# Country centroids derived once from loaded polygons.
_COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {}
if WORLD_BORDERS:
    try:
        _seen_polys = set()
        for _name, _poly in list(_COUNTRY_POLYGONS.items()):
            if _poly is None or id(_poly) in _seen_polys:
                continue
            _seen_polys.add(id(_poly))
            try:
                _pt = _poly.representative_point()
                _COUNTRY_CENTROIDS[_name] = (_pt.y, _pt.x)
            except Exception:
                pass
    except Exception:
        pass

# Manual region/capital centroids for frequent GeoGuessr regions.
# Key format: (country_lower_alias_normalized, region_lower)
_REGION_CENTROIDS: dict[tuple[str, str], tuple[float, float]] = {
    ("austria", "vienna"): (48.21, 16.37),
    ("france", "paris"): (48.86, 2.35), ("france", "île-de-france"): (48.86, 2.35),
    ("germany", "berlin"): (52.52, 13.40), ("germany", "bavaria"): (48.78, 11.50),
    ("italy", "rome"): (41.90, 12.50), ("italy", "lazio"): (41.90, 12.50),
    ("italy", "lombardy"): (45.46, 9.19),
    ("spain", "madrid"): (40.42, -3.70), ("spain", "catalonia"): (41.58, 1.62),
    ("portugal", "lisbon"): (38.72, -9.14), ("portugal", "porto"): (41.15, -8.61),
    ("united kingdom", "london"): (51.51, -0.13), ("united kingdom", "england"): (52.50, -1.50),
    ("united kingdom", "scotland"): (56.49, -4.20), ("united kingdom", "wales"): (52.10, -3.78),
    ("ireland", "dublin"): (53.35, -6.26),
    ("netherlands", "amsterdam"): (52.37, 4.90), ("netherlands", "north holland"): (52.50, 4.80),
    ("belgium", "brussels"): (50.85, 4.35),
    ("switzerland", "zurich"): (47.38, 8.54), ("switzerland", "geneva"): (46.20, 6.15),
    ("poland", "warsaw"): (52.23, 21.01), ("poland", "masovian"): (52.23, 21.01),
    ("czechia", "prague"): (50.08, 14.43), ("czech republic", "prague"): (50.08, 14.43),
    ("hungary", "budapest"): (47.50, 19.04),
    ("greece", "athens"): (37.98, 23.73),
    ("turkey", "istanbul"): (41.01, 28.97), ("turkey", "ankara"): (39.93, 32.87),
    ("russia", "moscow"): (55.75, 37.62), ("russian federation", "moscow"): (55.75, 37.62),
    ("japan", "tokyo"): (35.68, 139.65), ("japan", "osaka"): (34.69, 135.50),
    ("japan", "okayama"): (34.66, 133.93), ("japan", "hokkaido"): (43.06, 141.35),
    ("south korea", "seoul"): (37.57, 126.98),
    ("china", "beijing"): (39.90, 116.40), ("china", "shanghai"): (31.23, 121.47),
    ("thailand", "bangkok"): (13.75, 100.50),
    ("indonesia", "jakarta"): (-6.21, 106.85),
    ("malaysia", "kuala lumpur"): (3.14, 101.69),
    ("singapore", "singapore"): (1.35, 103.82),
    ("philippines", "manila"): (14.60, 120.98),
    ("india", "delhi"): (28.61, 77.21), ("india", "mumbai"): (19.08, 72.88),
    ("australia", "sydney"): (-33.87, 151.21), ("australia", "new south wales"): (-33.87, 151.21),
    ("australia", "victoria"): (-37.81, 144.96), ("australia", "queensland"): (-27.47, 153.03),
    ("new zealand", "auckland"): (-36.85, 174.76), ("new zealand", "wellington"): (-41.29, 174.78),
    ("canada", "toronto"): (43.65, -79.38), ("canada", "ontario"): (43.65, -79.38),
    ("canada", "quebec"): (46.81, -71.21), ("canada", "british columbia"): (49.28, -123.12),
    ("united states", "new york"): (40.71, -74.01), ("united states", "california"): (36.78, -119.42),
    ("united states", "texas"): (31.97, -99.90), ("united states", "florida"): (27.66, -81.52),
    ("united states", "nevada"): (38.80, -116.42), ("united states", "arizona"): (34.05, -111.09),
    ("mexico", "mexico city"): (19.43, -99.13),
    ("brazil", "são paulo"): (-23.55, -46.63), ("brazil", "rio de janeiro"): (-22.91, -43.20),
    ("argentina", "buenos aires"): (-34.61, -58.38),
    ("south africa", "johannesburg"): (-26.20, 28.04), ("south africa", "cape town"): (-33.92, 18.42),
    ("south africa", "gauteng"): (-26.27, 28.11), ("south africa", "western cape"): (-33.92, 18.42),
    ("south africa", "kwazulu-natal"): (-29.86, 31.02), ("south africa", "eastern cape"): (-33.04, 27.87),
    ("south africa", "limpopo"): (-23.40, 29.45), ("south africa", "mpumalanga"): (-25.47, 30.98),
    ("south africa", "north-west"): (-26.69, 25.28), ("south africa", "free state"): (-29.12, 26.21),
    ("kenya", "nairobi"): (-1.29, 36.82),
    ("egypt", "cairo"): (30.04, 31.24),
    # Russia (frequent GeoGuessr regions)
    ("russia", "saint petersburg"): (59.93, 30.33), ("russian federation", "saint petersburg"): (59.93, 30.33),
    ("russia", "moskovskaya"): (55.71, 37.02), ("russian federation", "moskovskaya"): (55.71, 37.02),
    ("russia", "perm"): (58.01, 56.25), ("russian federation", "perm"): (58.01, 56.25),
    ("russia", "sverdlovsk"): (56.84, 60.60), ("russian federation", "sverdlovsk"): (56.84, 60.60),
    ("russia", "khabarovsk krai"): (48.48, 135.08), ("russian federation", "khabarovsk krai"): (48.48, 135.08),
    ("russia", "altai krai"): (52.63, 82.25), ("russian federation", "altai krai"): (52.63, 82.25),
    ("russia", "vladimir"): (56.13, 40.41), ("russian federation", "vladimir"): (56.13, 40.41),
    ("russia", "krasnodar"): (45.04, 38.98), ("russian federation", "krasnodar"): (45.04, 38.98),
    ("russia", "novosibirsk"): (55.00, 82.92), ("russian federation", "novosibirsk"): (55.00, 82.92),
    ("russia", "chelyabinsk"): (55.16, 61.40), ("russian federation", "chelyabinsk"): (55.16, 61.40),
    ("russia", "tatarstan"): (55.79, 49.12), ("russian federation", "tatarstan"): (55.79, 49.12),
    # Ukraine
    ("ukraine", "kyiv"): (50.45, 30.52), ("ukraine", "lviv"): (49.84, 24.03),
    ("ukraine", "odesa"): (46.48, 30.73), ("ukraine", "kharkiv"): (49.99, 36.23),
    # Central Asia
    ("kyrgyzstan", "bishkek"): (42.87, 74.59), ("kyrgyzstan", "chuy"): (42.88, 74.59),
    ("kyrgyzstan", "osh"): (40.53, 72.79), ("kyrgyzstan", "batken"): (40.06, 70.82),
    ("kazakhstan", "almaty"): (43.22, 76.85), ("kazakhstan", "astana"): (51.17, 71.45),
    ("kazakhstan", "nur-sultan"): (51.17, 71.45),
    ("uzbekistan", "tashkent"): (41.30, 69.24),
    ("mongolia", "ulaanbaatar"): (47.88, 106.91),
    # Europe extras
    ("france", "auvergne-rhône-alpes"): (45.50, 4.60), ("france", "auvergne"): (45.75, 3.11),
    ("france", "brittany"): (48.20, -2.93), ("france", "normandy"): (49.18, 0.37),
    ("france", "nouvelle-aquitaine"): (45.50, 0.00), ("france", "poitou-charentes"): (46.10, -0.10),
    ("france", "occitanie"): (43.60, 2.00), ("france", "provence-alpes-côte d'azur"): (43.93, 6.06),
    ("france", "corsica"): (42.04, 9.01), ("france", "hauts-de-france"): (50.30, 2.80),
    ("france", "grand est"): (48.70, 5.40), ("france", "pays de la loire"): (47.50, -0.80),
    ("france", "centre-val de loire"): (47.50, 1.60), ("france", "bourgogne-franche-comté"): (47.20, 4.70),
    ("germany", "baden-württemberg"): (48.66, 9.35), ("germany", "saxony"): (51.10, 13.20),
    ("germany", "north rhine-westphalia"): (51.43, 7.66), ("germany", "hesse"): (50.65, 9.16),
    ("germany", "lower saxony"): (52.64, 9.85), ("germany", "rhineland-palatinate"): (50.12, 7.31),
    ("germany", "thuringia"): (50.86, 11.05), ("germany", "schleswig-holstein"): (54.22, 9.70),
    ("italy", "tuscany"): (43.77, 11.26), ("italy", "campania"): (40.84, 14.25),
    ("italy", "sicily"): (37.60, 14.02), ("italy", "sardinia"): (40.12, 9.01),
    ("italy", "veneto"): (45.44, 12.32), ("italy", "piedmont"): (45.07, 7.69),
    ("italy", "emilia-romagna"): (44.49, 11.34), ("italy", "apulia"): (40.79, 17.10),
    ("italy", "calabria"): (38.91, 16.59),
    ("spain", "andalusia"): (37.54, -4.73), ("spain", "galicia"): (42.76, -7.98),
    ("spain", "basque country"): (43.00, -2.60), ("spain", "valencia"): (39.47, -0.38),
    ("spain", "aragón"): (41.60, -0.89), ("spain", "castile and leon"): (41.75, -4.77),
    ("spain", "castilla-la mancha"): (39.27, -3.10),
    ("united kingdom", "northern ireland"): (54.60, -6.50),
    ("ireland", "connaught"): (53.63, -8.63), ("ireland", "leinster"): (53.35, -6.26),
    ("ireland", "munster"): (52.26, -8.62), ("ireland", "ulster"): (54.60, -6.92),
    ("norway", "vestland"): (60.70, 5.92), ("norway", "sogn og fjordane"): (61.50, 6.70),
    ("norway", "hordaland"): (60.39, 5.32), ("norway", "buskerud"): (60.20, 9.00),
    ("norway", "oslo"): (59.91, 10.75), ("norway", "trøndelag"): (63.47, 10.92),
    ("sweden", "stockholm"): (59.33, 18.07), ("sweden", "västra götaland"): (58.25, 12.32),
    ("sweden", "skåne"): (55.99, 13.59), ("sweden", "vaestra goetaland"): (58.25, 12.32),
    ("finland", "helsinki"): (60.17, 24.94), ("finland", "uusimaa"): (60.34, 25.01),
    ("denmark", "copenhagen"): (55.68, 12.57), ("denmark", "zealand"): (55.46, 11.75),
    ("estonia", "tallinn"): (59.44, 24.75), ("estonia", "harju"): (59.33, 25.28),
    ("latvia", "riga"): (56.95, 24.11),
    ("lithuania", "vilnius"): (54.69, 25.28),
    ("romania", "bucharest"): (44.43, 26.10), ("romania", "harghita"): (46.36, 25.80),
    ("romania", "transylvania"): (46.17, 24.07), ("romania", "cluj"): (46.77, 23.60),
    ("bulgaria", "sofia"): (42.70, 23.32), ("bulgaria", "pernik"): (42.60, 23.04),
    ("bulgaria", "vidin"): (43.99, 22.87), ("bulgaria", "plovdiv"): (42.14, 24.75),
    ("czechia", "bohemia"): (49.80, 14.50), ("czechia", "moravia"): (49.44, 17.12),
    ("czech republic", "bohemia"): (49.80, 14.50), ("czech republic", "moravia"): (49.44, 17.12),
    ("slovakia", "bratislava"): (48.15, 17.11),
    ("slovenia", "ljubljana"): (46.06, 14.51),
    ("croatia", "zagreb"): (45.81, 15.98), ("croatia", "grad zagreb"): (45.82, 16.00),
    ("croatia", "istarska"): (45.27, 13.93), ("croatia", "dalmatia"): (43.51, 16.44),
    ("croatia", "licko-senjska"): (44.55, 15.37),
    ("serbia", "belgrade"): (44.80, 20.47),
    ("albania", "tirana"): (41.33, 19.82),
    ("hungary", "pest"): (47.50, 19.08),
    ("austria", "tyrol"): (47.25, 11.40), ("austria", "salzburg"): (47.80, 13.04),
    ("austria", "upper austria"): (48.30, 14.29), ("austria", "lower austria"): (48.11, 15.80),
    ("austria", "vorarlberg"): (47.24, 9.93), ("austria", "styria"): (47.36, 14.27),
    ("austria", "carinthia"): (46.72, 14.18),
    # North America — states
    ("united states", "washington"): (47.40, -120.74), ("united states", "oregon"): (43.80, -120.55),
    ("united states", "idaho"): (44.07, -114.74), ("united states", "montana"): (46.88, -110.36),
    ("united states", "wyoming"): (43.08, -107.29), ("united states", "colorado"): (39.06, -105.31),
    ("united states", "utah"): (39.32, -111.09), ("united states", "new mexico"): (34.40, -106.11),
    ("united states", "kansas"): (38.53, -98.68), ("united states", "nebraska"): (41.13, -98.27),
    ("united states", "south dakota"): (44.44, -100.24), ("united states", "north dakota"): (47.53, -100.30),
    ("united states", "minnesota"): (46.73, -94.69), ("united states", "iowa"): (42.03, -93.58),
    ("united states", "missouri"): (38.46, -92.29), ("united states", "oklahoma"): (35.47, -97.52),
    ("united states", "arkansas"): (34.97, -92.37), ("united states", "louisiana"): (31.17, -91.87),
    ("united states", "mississippi"): (32.74, -89.68), ("united states", "alabama"): (32.80, -86.79),
    ("united states", "georgia"): (32.64, -83.44), ("united states", "tennessee"): (35.86, -86.66),
    ("united states", "kentucky"): (37.83, -84.27), ("united states", "illinois"): (40.35, -89.05),
    ("united states", "indiana"): (39.77, -86.15), ("united states", "ohio"): (40.37, -82.73),
    ("united states", "michigan"): (44.35, -85.41), ("united states", "wisconsin"): (44.27, -89.62),
    ("united states", "west virginia"): (38.60, -80.45), ("united states", "virginia"): (37.77, -78.17),
    ("united states", "north carolina"): (35.55, -79.39), ("united states", "south carolina"): (33.86, -80.95),
    ("united states", "pennsylvania"): (40.59, -77.20), ("united states", "maryland"): (39.06, -76.80),
    ("united states", "rhode island"): (41.68, -71.51), ("united states", "massachusetts"): (42.23, -71.53),
    ("united states", "connecticut"): (41.59, -72.76), ("united states", "new hampshire"): (43.45, -71.56),
    ("united states", "maine"): (44.69, -69.38), ("united states", "vermont"): (44.04, -72.71),
    ("united states", "alaska"): (64.20, -150.00), ("united states", "hawaii"): (20.80, -156.33),
    # Canada
    ("canada", "alberta"): (53.93, -116.58), ("canada", "saskatchewan"): (52.94, -106.45),
    ("canada", "manitoba"): (53.76, -98.81), ("canada", "nova scotia"): (44.68, -63.74),
    ("canada", "new brunswick"): (46.57, -66.46), ("canada", "newfoundland and labrador"): (48.99, -55.66),
    # Mexico
    ("mexico", "jalisco"): (20.66, -103.35), ("mexico", "chiapas"): (16.76, -93.12),
    ("mexico", "hidalgo"): (20.12, -98.74), ("mexico", "nuevo león"): (25.59, -99.99),
    ("mexico", "yucatán"): (20.85, -89.06), ("mexico", "oaxaca"): (17.07, -96.72),
    # South America
    ("brazil", "goias"): (-15.93, -49.00), ("brazil", "goiás"): (-15.93, -49.00),
    ("brazil", "maranhao"): (-5.41, -44.33), ("brazil", "maranhão"): (-5.41, -44.33),
    ("brazil", "minas gerais"): (-19.92, -43.94), ("brazil", "bahia"): (-12.97, -38.51),
    ("brazil", "paraná"): (-25.25, -52.02), ("brazil", "rio grande do sul"): (-30.03, -51.23),
    ("argentina", "rio negro"): (-40.50, -66.64), ("argentina", "córdoba"): (-31.42, -64.18),
    ("argentina", "mendoza"): (-32.89, -68.84), ("argentina", "buenos aires province"): (-36.50, -60.00),
    ("chile", "santiago"): (-33.45, -70.67), ("chile", "santiago metropolitan"): (-33.64, -70.72),
    ("chile", "valparaiso"): (-33.05, -71.62),
    ("ecuador", "pichincha"): (-0.32, -78.54), ("ecuador", "guayas"): (-2.14, -79.94),
    ("colombia", "bogota"): (4.71, -74.07), ("colombia", "boyaca"): (5.63, -73.17),
    ("peru", "lima"): (-12.05, -77.04),
    # Asia — Japan prefectures
    ("japan", "kumamoto"): (32.79, 130.74), ("japan", "miyazaki"): (31.91, 131.42),
    ("japan", "kagoshima"): (31.56, 130.56), ("japan", "fukuoka"): (33.59, 130.40),
    ("japan", "hiroshima"): (34.39, 132.46), ("japan", "aichi"): (35.18, 136.91),
    ("japan", "kyoto"): (35.02, 135.75), ("japan", "kanagawa"): (35.45, 139.64),
    ("japan", "saitama"): (35.86, 139.64), ("japan", "chiba"): (35.61, 140.12),
    ("japan", "nagano"): (36.65, 138.18), ("japan", "niigata"): (37.90, 139.02),
    # SE Asia
    ("malaysia", "kuala lumpur"): (3.14, 101.69), ("malaysia", "penang"): (5.38, 100.40),
    ("malaysia", "johor"): (1.49, 103.75), ("malaysia", "selangor"): (3.07, 101.52),
    ("indonesia", "jambi"): (-1.61, 103.61), ("indonesia", "java"): (-7.61, 110.71),
    ("indonesia", "bali"): (-8.34, 115.09), ("indonesia", "sumatra"): (-0.59, 101.34),
    ("philippines", "luzon"): (16.67, 121.24), ("philippines", "soccsksargen"): (6.50, 124.85),
    ("philippines", "cebu"): (10.32, 123.90), ("philippines", "davao"): (7.07, 125.61),
    ("taiwan", "kaohsiung"): (22.63, 120.30), ("taiwan, province of china", "kaohsiung"): (22.63, 120.30),
    ("taiwan", "taipei"): (25.03, 121.57),
    ("hong kong", "new territories"): (22.42, 114.11),
    # India states
    ("india", "maharashtra"): (19.60, 75.55), ("india", "tamil nadu"): (11.13, 78.66),
    ("india", "karnataka"): (15.32, 75.71), ("india", "west bengal"): (22.99, 87.86),
    ("india", "uttar pradesh"): (26.85, 80.95), ("india", "punjab"): (31.15, 75.34),
    ("india", "gujarat"): (22.26, 71.19), ("india", "rajasthan"): (27.02, 74.22),
    ("india", "kerala"): (10.85, 76.27),
    # Oceania
    ("new zealand", "waikato"): (-37.78, 175.28), ("new zealand", "canterbury"): (-43.75, 171.30),
    ("new zealand", "otago"): (-45.24, 170.20), ("new zealand", "northland"): (-35.73, 174.32),
    ("australia", "western australia"): (-27.67, 121.63), ("australia", "south australia"): (-30.00, 136.21),
    ("australia", "tasmania"): (-41.45, 145.97), ("australia", "northern territory"): (-19.49, 132.55),
    # Africa extras
    ("morocco", "casablanca"): (33.57, -7.59),
    ("nigeria", "lagos"): (6.52, 3.38),
    ("ghana", "greater accra"): (5.60, -0.19),
    ("tanzania", "dar es salaam"): (-6.79, 39.21),
    ("uganda", "kampala"): (0.35, 32.58),
}


def region_coord_hint(country: str, region: str) -> str:
    """Return a short '{Region} is around (lat, lon)' hint if we know it."""
    c = _COUNTRY_ALIASES.get((country or "").strip().lower(), (country or "").strip().lower())
    r = (region or "").strip().lower()
    if c and r:
        pt = _REGION_CENTROIDS.get((c, r))
        if pt:
            return f"{region}, {country} is roughly at ({pt[0]:.2f}, {pt[1]:.2f})."
    if c:
        pt = _COUNTRY_CENTROIDS.get(c)
        if pt:
            return f"{country} is roughly centered at ({pt[0]:.2f}, {pt[1]:.2f})."
    return ""


def country_bbox_hint(country: str) -> str:
    """Return 'your lat must be in [A,B] and lon in [C,D]' bbox for a country, if known."""
    if not country or not _COUNTRY_POLYGONS:
        return ""
    c = _COUNTRY_ALIASES.get(country.strip().lower(), country.strip().lower())
    poly = _COUNTRY_POLYGONS.get(c)
    if poly is None:
        return ""
    try:
        lon_min, lat_min, lon_max, lat_max = poly.bounds
    except Exception:
        return ""
    return (
        f"Valid {country} coordinates MUST satisfy: "
        f"latitude ∈ [{lat_min:.2f}, {lat_max:.2f}], "
        f"longitude ∈ [{lon_min:.2f}, {lon_max:.2f}]."
    )

ISO2_TO_NAME: dict[str, str] = {}
try:
    import pycountry  # optional — nicer names
    for c in pycountry.countries:
        ISO2_TO_NAME[c.alpha_2] = c.name
except Exception:
    pass


def lookup_country(lat: float, lon: float) -> tuple[str, str, str]:
    """Return (country_name, admin1, cc) for given coords, offline."""
    res = rg.search((lat, lon), mode=1)
    if not res:
        return ("", "", "")
    r = res[0]
    cc = r.get("cc", "")
    name = ISO2_TO_NAME.get(cc, cc)
    return (name, r.get("admin1", ""), cc)


def distance_to_nearest_land_km(lat: float, lon: float) -> float:
    """Haversine distance (km) from (lat,lon) to nearest populated place.
    Large values (>80km) strongly suggest the point is in the ocean/open water."""
    res = rg.search((lat, lon), mode=1)
    if not res:
        return 9999.0
    r = res[0]
    try:
        nlat = float(r.get("lat"))
        nlon = float(r.get("lon"))
    except Exception:
        return 9999.0
    R = 6371.0
    p1, p2 = math.radians(lat), math.radians(nlat)
    dp = math.radians(nlat - lat)
    dl = math.radians(nlon - lon)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

load_dotenv()

# ANSI color codes — enable virtual terminal on Windows so PowerShell renders them.
if os.name == "nt":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GREY = "\033[90m"


_round_started_at: float = 0.0


def _ts() -> str:
    """Seconds since current round started, formatted as +SS.Ss."""
    if _round_started_at <= 0:
        return "      "
    return f"+{time.time() - _round_started_at:5.1f}s"


def log(tag: str, msg: str, color: str = "") -> None:
    """Pretty per-round log line: timestamp | tag | message."""
    c = color
    end = _C.RESET if color else ""
    print(f"  {_C.GREY}{_ts()}{_C.RESET} {c}{tag:<10}{end} {msg}")


OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
OPENAI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or "missing-key"
# MODE controls behaviour:
#   solo        — original behaviour (no overlays, no UI mask, __geoMap-based pin
#                  placement, 5-round + Play-Again loop, 600s API timeout).
#   duels       — 1v1 NMPZ multiplayer: canvas-stability detection, UI mask for
#                  HP/avatars/minimap (2 players), Mercator+zoom click for pin
#                  placement, 25s API timeout, auto-continue between matches.
#   team-duels  — 2v2 or 1v3 NMPZ multiplayer: same as duels but the UI mask is
#                  wider to cover all 4 player avatars + 3 opponent "has guessed"
#                  notifications.
MODE = os.getenv("MODE", "solo").lower().strip()
IS_MULTIPLAYER = MODE in ("duels", "team-duels")
# MODEL_IDS: comma-separated list for A/B testing (round-robin per round).
# Falls back to single MODEL_ID if MODEL_IDS not set.
_model_ids_env = os.getenv("MODEL_IDS", "").strip()
if _model_ids_env:
    MODEL_IDS = [m.strip() for m in _model_ids_env.split(",") if m.strip()]
else:
    MODEL_IDS = [os.getenv("MODEL_ID", "gemini-3.1-flash-lite")]
MODEL_ID = MODEL_IDS[0]  # kept for compatibility

PROJECT_DIR = Path(__file__).parent
SCREENSHOTS_DIR = PROJECT_DIR / "screenshots"
USER_DATA_DIR = PROJECT_DIR / "user_data"
LOG_FILE = PROJECT_DIR / "log.json"
SCREENSHOTS_DIR.mkdir(exist_ok=True)
USER_DATA_DIR.mkdir(exist_ok=True)

_CLIENT_TIMEOUT = 25.0 if IS_MULTIPLAYER else 600.0
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY, timeout=_CLIENT_TIMEOUT)

SYSTEM_PROMPT = """You are a GeoGuessr expert. Before answering, silently identify these concrete signals in the image (do NOT include them in output, only use them to decide):

1. LANGUAGE & SCRIPT on any signs, storefronts, license plates (Latin/Cyrillic/Arabic/CJK/Thai/Devanagari/Greek/Hebrew). Specific words narrow country fast.
2. DRIVING SIDE — cars drive on left (UK, Japan, Australia, India, South Africa, Ireland, Thailand, Indonesia, Malaysia) vs right (most of world).
3. LICENSE PLATE color/shape — EU yellow-rear, US white with state name, Brazil Mercosur, Japan white+green, etc. EU plates have a BLUE LEFT STRIP with a 2-letter country code — if readable, it is definitive: FR=France, DE=Germany, RO=Romania, IT=Italy, ES=Spain, AT=Austria, NL=Netherlands, BE=Belgium, PL=Poland, CZ=Czechia, HU=Hungary, SK=Slovakia, SI=Slovenia, HR=Croatia, BG=Bulgaria, GR=Greece, PT=Portugal, GB=United Kingdom, SE=Sweden, NO=Norway, FI=Finland, DK=Denmark, IE=Ireland, LT=Lithuania, LV=Latvia, EE=Estonia, LU=Luxembourg, CH=Switzerland.
4. ROAD MARKINGS — yellow center line (US, Canada, Mexico, Japan) vs white (Europe, most others); double solid, dashed patterns.
5. UTILITY POLES — wooden US-style, concrete European, distinctive Japanese/Thai shapes.
6. VEGETATION & CLIMATE — palms (tropical), eucalyptus (Australia/Iberia), conifers (Nordic/Canada/Russia), savanna, desert, tundra.
7. ARCHITECTURE — roof tiles (Mediterranean orange, Nordic dark), wooden houses (Scandinavia/Russia), adobe (Latin America), etc.
8. ROAD QUALITY & VEHICLES — old cars and poor roads suggest developing countries; specific car models are regional.
9. GOOGLE CAR / CAMERA — sometimes visible (snorkel in Kenya/Senegal, pickup truck in Mongolia, etc.).
10. HEMISPHERE & SHADOWS — Compare the shadows to the OpenCV compass direction. If shadows point North, the sun is South, meaning you are in the Northern Hemisphere. If shadows point South, you are in the Southern Hemisphere.

CRITICAL NMPZ ANTI-ERROR RULES (APPLY THESE TO RESOLVE CLOSE TIES):
- CANADA vs USA: Unpainted wooden utility poles, yellow center lines, speed signs reading "MAX" = Canada. "SPEED LIMIT" = USA.
- BRAZIL vs ARGENTINA/MEXICO: Very red/orange dirt, ladder-style concrete poles (postes de betão furado), black backs of traffic signs = Brazil.
- RUSSIA vs UKRAINE: Endless birch trees, poor road quality, A-frame concrete poles, black painted pole bases = Russia. Ukraine often has a visible red car antenna.
- SPAIN vs MEXICO: Spain has flawless European-style paved roads, thin continuous white line on edge, lots of olives/eucalyptus, European plates. Mexico has rougher roads, "SCT" signs, concrete power poles, and yellow center lines.
- FRANCE vs SPAIN/ITALY: French roads have thick dashed white edge lines. Spanish asphalt is pristine with very thin continuous edge lines. Italian signs have black/dark grey backs.
- AUSTRALIA vs SOUTH AFRICA: Both drive on LEFT. South Africa has YELLOW outer edge lines. Australia has WHITE outer edge lines and eucalyptus trees.
- UK vs IRELAND: Both drive on LEFT. UK has YELLOW rear license plates. Ireland has WHITE rear license plates and yellow diamond warning signs.
- SCANDINAVIA: Norway has YELLOW center lines. Sweden has short-dashed WHITE edge lines. Denmark is totally flat with red-and-white road delineator posts.
- COLOMBIA: If you see a black-and-white striped cross on the back of road signs, it is Colombia. Yellow license plates are common.
- JAPAN: Drives on LEFT, low camera angle, yellow and black striped utility poles, curve mirrors at intersections.
- GERMANY vs POLAND/AUSTRIA: Germany often has blurred houses, short-dashed white edge lines, and bollards with vertical black lines. Poland has bollards with a thick horizontal red band and uses holey concrete poles. Austria uses bollards with black "hats" or caps.
- DO NOT DEFAULT TO USA: USA is the most common country in GeoGuessr but "generic-looking" does NOT mean USA. Specific traps:
  * Patagonia (southern Argentina, lat -45 to -55): barren windswept steppe identical to Montana/Wyoming — but Spanish signs = Argentina.
  * Romanian/Italian/Eastern European countryside: open fields, old farmhouses ≠ USA. Check for Cyrillic, EU plates, European road signs.
  * Central Anatolian plateau (Turkey): flat dry plateau ≠ US Midwest. Latin text with Ğ/İ/Ş/Ü/Ö = Türkiye.
  * UAE/Middle East desert roads: straight desert highway ≠ US Southwest. Check for Arabic script, right-hand traffic.
  * Northeast Brazil (caatinga): dry scrubland with sparse trees ≠ US South. Portuguese text confirms Brazil.
  * Northern Chile coast: arid rocky desert ≠ US Southwest. Spanish text, Pacific Ocean cliffs.
  * South African highveld: flat golden grassland ≠ US Midwest. Left-hand traffic + yellow road lines = South Africa.
  Only guess USA when you see: English-only signs, "SPEED LIMIT" text, US state highway shields, or American-specific infrastructure.
- DO NOT CONFUSE THESE PAIRS:
  * South Africa vs Spain/Europe: South Africa has LEFT-HAND traffic, YELLOW outer road lines, bird poles, green N-road signs. No European country drives on the left.
  * New Zealand vs UK/Sweden: NZ has lat -36 to -47 (southern hemisphere), distinctive red-top bollards, white outer lines, possum guards on poles.
  * Chile vs Romania/Ukraine: Chile has Andes visible east + Pacific west, Spanish text, all-white or all-yellow road lines. Romania has holey poles + Romanian Cyrillic-like text.
  * Cambodia vs Brazil: Cambodia has Khmer script (curly angular letters), white+blue plates. Brazil has Portuguese text, ladder-style concrete poles.
  * Sweden vs Finland: Sweden = SHORT white edge dashes + Swedish text (väg, gata, och). Finland = LONGER white edge dashes + Finnish text (double vowels: aamu, talo).
  * Poland vs Romania/Serbia/Croatia: Poland has bollards with a thick HORIZONTAL RED BAND near the top. Romania has holey concrete poles. Serbia/Croatia lack red-band bollards.

Then output ONE JSON object only. No text before or after, no code fences, no chain-of-thought.

Rules:
- "continent": the continent your answer is in. Must be one of: "Europe", "Asia", "Africa", "North America", "South America", "Oceania", "Antarctica". This MUST match your country — e.g. Philippines → "Asia", Colombia → "South America", USA → "North America". Declaring this forces you to double-check your continent before committing.
- "country": single country name (no lists, no alternatives, no hyphens joining multiple).
- "region": single region/state/province, or "" if unsure. Never JSON or reasoning here.
- "latitude": in [-90, 90]. NEGATIVE south of equator (Argentina, Australia, S.Africa, NZ, Chile).
- "longitude": in [-180, 180]. NEGATIVE for Americas (USA, Canada, Brazil, Argentina, Chile, Mexico). POSITIVE for Europe/Africa/Asia/Oceania. Double-check the sign.
- Do NOT default to the capital. Pick coordinates inside the region you chose, matching the landscape (urban vs rural, coastal vs inland).
- "confidence": 0-1. Calibration rules: use 0.92-0.95 ONLY when you have a plate code + a script/language clue + a unique infrastructure clue all agreeing (landmark-level certainty). Use 0.75-0.88 for one strong specific clue (e.g. plate code alone, or unique bollard type alone). Use 0.55-0.70 for educated guess based on landscape/vegetation with no text. NEVER use ≥0.90 for generic scenes (open road, forest, fields) without specific identifiers. A score of 0.95 should be extremely rare — if you feel 0.95, ask yourself: what would make this NOT that country? If you can't answer, you are overconfident.
- "reasoning": ≤150 chars plain text. Stating the STRONGEST clues. If you relied on the provided Meta Tips or Car Meta, specifically mention it. (e.g. "Meta tips mention double yellow center lines and yellow rear plates -> UK").
"""

USER_PROMPT = (
    "Identify where this Street View was captured. "
    "Remember: Western hemisphere (Americas) = NEGATIVE longitude. "
    "Eastern hemisphere (Europe, Africa, Asia, Oceania) = POSITIVE longitude. "
    "Southern hemisphere = NEGATIVE latitude. "
    "Return only the JSON."
)


@dataclass
class Guess:
    country: str
    region: str
    latitude: float
    longitude: float
    confidence: float
    reasoning: str
    continent: str = ""
    candidates: list[tuple[str, float]] = field(default_factory=list)


async def human_delay(a: float = 0.4, b: float = 1.2) -> None:
    await asyncio.sleep(random.uniform(a, b))


async def human_mouse_move(page: Page, x: float, y: float, steps: int = 25) -> None:
    await page.mouse.move(x, y, steps=steps)


def mask_duels_ui(png_bytes: bytes) -> bytes:
    """Black out the GeoGuessr multiplayer UI overlays so they don't pollute OCR.
    The .widget-scene-canvas spans the full window and the HP bars, avatars,
    compass, minimap, Google logo, and 'X has guessed' notifications are
    HTML positioned ON TOP — they end up in the screenshot. In team-duels
    (2v2 / 1v3) there are 4 avatars and 3 opponents that can fire 'has guessed'
    notifications, so we mask a slightly bigger area than in 1v1 duels.
    Street View text (signs, plates) is almost always in the middle band of
    the panorama, which we leave untouched.
    """
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    is_team = MODE == "team-duels"
    # Top strip taller in team-duels to cover 4 stacked avatars.
    top_h = 0.22 if is_team else 0.18
    # Center notification band wider in team-duels (3 opponents → more toasts).
    notif_x0, notif_x1 = (0.15, 0.85) if is_team else (0.20, 0.80)
    notif_y0, notif_y1 = (top_h, 0.34 if is_team else 0.30)
    regions = [
        (0,                 0,                  w,             int(top_h * h)),
        (int(notif_x0 * w), int(notif_y0 * h),  int(notif_x1 * w), int(notif_y1 * h)),
        (0,                 int(0.85 * h),      int(0.12 * w), h),               # bottom-left controls/Google logo
        (int(0.76 * w),     int(0.66 * h),      w,             h),               # bottom-right minimap
    ]
    for (x0, y0, x1, y1) in regions:
        draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def encode_image(png_bytes: bytes, max_side: int = 1568) -> str:
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def crop_car_meta(png_bytes: bytes) -> str:
    """Extrai a parte inferior do ecrã (25%) onde o carro do Google aparece e converte para base64."""
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    w, h = img.size
    crop_box = (0, int(h * 0.70), w, h) # Últimos 30% da imagem inferior
    cropped = img.crop(crop_box)
    buf = BytesIO()
    cropped.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

COMPASS_INVERT = False  # Flip to True if the detected direction is 180° off.
COMPASS_DEBUG = True    # Save annotated crops to compass_debug/ for inspection.


def crop_compass_meta(png_bytes: bytes) -> str:
    """
    Detects the GeoGuessr compass needle direction via OpenCV.
    Strategy:
      1. Crop bottom-left corner (compass zone in NMPZ).
      2. Find compass circle (pivot) via Hough.
      3. Build red mask (the N-marker / red needle tip).
      4. Use the red pixel FARTHEST from pivot as the needle tip — robust against
         needles where red covers both halves (centroid bug).
      5. Compute bearing.
    """
    img_pil = Image.open(BytesIO(png_bytes)).convert("RGB")
    w, h = img_pil.size
    # A bússola nova do GeoGuessr fica no canto SUPERIOR no MEIO do ecrã
    crop_box = (int(w * 0.35), 0, int(w * 0.65), int(h * 0.20))
    compass_crop = img_pil.crop(crop_box)

    img_cv = cv2.cvtColor(np.array(compass_crop), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    # Red mask
    lower_red1 = np.array([0, 80, 60])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 80, 60])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    if int(np.count_nonzero(red_mask)) < 10:
        return ""

    # Find the compass circle (pivot)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=30, minRadius=10, maxRadius=60,
    )
    if circles is None:
        return ""

    with np.errstate(over="ignore"):
        circles = np.around(circles).astype(np.int64)
        # Pick the circle whose interior contains the most red pixels
        best_circle = None
        best_red = -1
        for i in circles[0, :]:
            cx_c, cy_c, r_c = int(i[0]), int(i[1]), int(i[2])
            y0, y1 = max(0, cy_c - r_c), min(red_mask.shape[0], cy_c + r_c)
            x0, x1 = max(0, cx_c - r_c), min(red_mask.shape[1], cx_c + r_c)
            sub = red_mask[y0:y1, x0:x1]
            red_in = int(np.count_nonzero(sub))
            if red_in > best_red:
                best_red = red_in
                best_circle = (cx_c, cy_c, r_c)
        if best_circle is None or best_red < 5:
            return ""
        cx, cy, r = best_circle

        # Mask red pixels within the circle radius only
        ys, xs = np.nonzero(red_mask)
        if len(xs) == 0:
            return ""
        dxs = xs.astype(np.int64) - cx
        dys = ys.astype(np.int64) - cy
        dists_sq = dxs * dxs + dys * dys
        inside = dists_sq <= (r * r)
        if not np.any(inside):
            return ""
        dxs_in = dxs[inside]
        dys_in = dys[inside]
        dists_in = dists_sq[inside]
        # Farthest red pixel from pivot = needle tip
        tip_idx = int(np.argmax(dists_in))
        tip_dx = int(dxs_in[tip_idx])
        tip_dy = int(dys_in[tip_idx])

        rads = math.atan2(tip_dy, tip_dx)
        degs = math.degrees(rads)
        facing_deg = (-degs - 90) % 360
        if COMPASS_INVERT:
            facing_deg = (facing_deg + 180) % 360

        dirs = ["North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"]
        idx = int(round(facing_deg / 45.0)) % 8
        direction = dirs[idx]

        if COMPASS_DEBUG:
            try:
                dbg_dir = PROJECT_DIR / "compass_debug"
                dbg_dir.mkdir(exist_ok=True)
                dbg = img_cv.copy()
                cv2.circle(dbg, (cx, cy), r, (0, 255, 0), 1)
                cv2.circle(dbg, (cx, cy), 2, (0, 255, 0), -1)
                tip_abs = (cx + tip_dx, cy + tip_dy)
                cv2.line(dbg, (cx, cy), tip_abs, (255, 255, 0), 2)
                cv2.circle(dbg, tip_abs, 3, (0, 0, 255), -1)
                cv2.putText(dbg, f"{direction} ({facing_deg:.0f}deg)",
                            (2, dbg.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 255, 255), 1, cv2.LINE_AA)
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(dbg_dir / f"compass_{ts}_{direction.replace('-', '')}.png"), dbg)
            except Exception:
                pass

        return direction


def ask_model(
    png_bytes: bytes,
    extra_user_msgs: list[dict] | None = None,
    model_id: str | None = None,
    car_bytes: bytes | None = None,
    compass_bytes: bytes | None = None,
) -> Guess:
    b64 = encode_image(png_bytes)
    
    content = [
        {"type": "text", "text": "MAIN VIEW:\n" + USER_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    ]

    if car_bytes:
        car_b64 = encode_image(car_bytes)
        content.extend([
            {"type": "text", "text": "CAR META CROP (Bottom section of the screen, look for roof racks, antennas, tape, snorkel, mirrors, camera halo):"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{car_b64}"}}
        ])

    if compass_bytes:
        compass_b64 = encode_image(compass_bytes)
        content.extend([
            {"type": "text", "text": "COMPASS METRICS CROP (Bottom left section, shows exact compass direction):"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{compass_b64}"}}
        ])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": content,
        },
    ]
    if extra_user_msgs:
        messages.extend(extra_user_msgs)
    resp = client.chat.completions.create(
        model=model_id or MODEL_ID,
        messages=messages,
        temperature=0.2,
        max_tokens=400,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "geo_guess",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "continent": {"type": "string"},
                        "country": {"type": "string"},
                        "region": {"type": "string"},
                        "latitude": {"type": "number"},
                        "longitude": {"type": "number"},
                        "confidence": {"type": "number"},
                        "reasoning": {"type": "string"},
                        "candidates": {
                            "type": "array",
                            "description": "Top-3 country hypotheses with confidence 0-1, including the primary. Ordered most to least likely.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "country": {"type": "string"},
                                    "confidence": {"type": "number"},
                                },
                                "required": ["country", "confidence"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["continent", "country", "region", "latitude", "longitude", "confidence", "reasoning", "candidates"],
                    "additionalProperties": False,
                },
            },
        },
    )
    raw = resp.choices[0].message.content or ""
    thinking = getattr(resp.choices[0].message, "reasoning_content", None) or ""
    if thinking:
        short = thinking.replace("\n", " ").strip()[:220]
        print(f"  [thinking] {short}{'…' if len(thinking) > 220 else ''}")
        
    if not raw.strip() and thinking.strip():
        # O modelo (ex: Qwen) pode colocar acidentalmente o JSON final dentro do bloco de raciocínio
        raw = thinking

    if not raw.strip():
        raise ValueError("empty model output (check LM Studio logs — image may have been rejected)")
    
    return parse_guess(raw)


def parse_guess(raw: str) -> Guess:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in model output: {raw[:200]}")
    data = json.loads(match.group(0))
    lat = float(data["latitude"])
    lon = float(data["longitude"])
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"latitude out of range: {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"longitude out of range: {lon}")
    cands_raw = data.get("candidates") or []
    candidates: list[tuple[str, float]] = []
    if isinstance(cands_raw, list):
        for c in cands_raw[:5]:
            if isinstance(c, dict):
                cn = str(c.get("country", "")).strip()
                try:
                    cc = float(c.get("confidence", 0.0))
                except (TypeError, ValueError):
                    cc = 0.0
                if cn:
                    candidates.append((cn, cc))
    return Guess(
        continent=str(data.get("continent", "")),
        country=str(data.get("country", "")),
        region=str(data.get("region", "")),
        latitude=lat,
        longitude=lon,
        confidence=min(float(data.get("confidence", 0.0)), 1.0) if float(data.get("confidence", 0.0)) <= 1.0 else float(data.get("confidence", 0.0)) / 100.0,
        reasoning=str(data.get("reasoning", "")),
        candidates=candidates,
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


async def capture_streetview(page: Page) -> tuple[bytes, dict]:
    # GeoGuessr uses a canvas for Street View; try to screenshot just that canvas.
    viewport = page.viewport_size or {"width": 1366, "height": 850}
    selectors = [
        ".widget-scene-canvas",
        "canvas.mapsConsumerUiSceneCoreScene__canvas",
        "div.game-layout__panorama",
        "div[class*='panorama']",
        "canvas",
    ]
    for sel in selectors:
        for el in await page.query_selector_all(sel):
            box = await el.bounding_box()
            if not box:
                continue
            # Clamp to viewport and ensure the element is actually visible.
            x = max(0.0, box["x"])
            y = max(0.0, box["y"])
            w = min(box["width"], viewport["width"] - x)
            h = min(box["height"], viewport["height"] - y)
            if w < 300 or h < 200:
                continue
            clip = {"x": x, "y": y, "width": w, "height": h}
            try:
                png = await page.screenshot(clip=clip, type="png")
                return png, clip
            except Exception:
                continue
    # Fallback: viewport screenshot.
    png = await page.screenshot(type="png", full_page=False)
    return png, {"x": 0, "y": 0, "width": viewport["width"], "height": viewport["height"]}


async def find_minimap(page: Page, debug: bool = False) -> dict | None:
    # The GeoGuessr minimap is a Google Maps inside a container. We need the
    # on-screen rect of the actual map tiles, not the container (which may
    # include title/padding) nor stray overlay canvases.
    info = await page.evaluate(
        """() => {
            const container = document.querySelector("[data-qa='guess-map'], .guess-map, div[class*='guess-map']");
            if (!container) return null;
            const cRect = container.getBoundingClientRect();
            const canvases = [...container.querySelectorAll('canvas')].map(c => {
                const r = c.getBoundingClientRect();
                return {x: r.x, y: r.y, w: r.width, h: r.height, cls: c.className || ''};
            });
            // Google Maps tiles container (gm-style).
            const tileDiv = container.querySelector(".gm-style > div:first-child") || container.querySelector(".gm-style");
            let tile = null;
            if (tileDiv) {
                const r = tileDiv.getBoundingClientRect();
                tile = {x: r.x, y: r.y, w: r.width, h: r.height};
            }
            return {container: {x: cRect.x, y: cRect.y, w: cRect.width, h: cRect.height}, canvases, tile};
        }"""
    )
    if debug:
        print(f"  [minimap debug] {info}")
    if not info:
        return None
    # Prefer the .gm-style tile div (actual map area).
    if info.get("tile") and info["tile"]["w"] > 100 and info["tile"]["h"] > 100:
        t = info["tile"]
        return {"x": t["x"], "y": t["y"], "width": t["w"], "height": t["h"]}
    # Else biggest canvas inside container.
    canvases = [c for c in info["canvases"] if c["w"] > 100 and c["h"] > 100]
    if canvases:
        c = max(canvases, key=lambda c: c["w"] * c["h"])
        return {"x": c["x"], "y": c["y"], "width": c["w"], "height": c["h"]}
    # Fallback: container.
    c = info["container"]
    return {"x": c["x"], "y": c["y"], "width": c["w"], "height": c["h"]}


def latlon_to_minimap_xy_bounds(
    lat: float, lon: float, box: dict, bounds: dict
) -> tuple[float, float]:
    # Web Mercator within known geographic bounds of the map viewport.
    def merc_y(lat_):
        lat_c = max(-85.0511, min(85.0511, lat_))
        s = math.sin(math.radians(lat_c))
        return 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)

    north, south = bounds["north"], bounds["south"]
    west, east = bounds["west"], bounds["east"]
    my_top, my_bot = merc_y(north), merc_y(south)
    my = merc_y(lat)
    # Longitude may wrap the antimeridian; normalize to the same side as west.
    lon_n = lon
    if east < west:  # viewport crosses 180°
        if lon_n < west:
            lon_n += 360.0
        east_n = east + 360.0
        x_frac = (lon_n - west) / (east_n - west)
    else:
        x_frac = (lon_n - west) / (east - west)
    y_frac = (my - my_top) / (my_bot - my_top)
    x = box["x"] + max(0.0, min(1.0, x_frac)) * box["width"]
    y = box["y"] + max(0.0, min(1.0, y_frac)) * box["height"]
    return x, y


def latlon_to_world_mercator(lat: float, lon: float, box: dict) -> tuple[float, float]:
    """Direct Web Mercator projection assuming the box shows the full world
    (zoom ~1). Reliable for the Duels guess map which defaults to world view
    and doesn't require Google Maps internal state."""
    lat_c = max(-85.0511, min(85.0511, lat))
    s = math.sin(math.radians(lat_c))
    merc_y = 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)
    x_frac = (lon + 180) / 360
    x = box["x"] + max(0.0, min(1.0, x_frac)) * box["width"]
    y = box["y"] + max(0.0, min(1.0, merc_y)) * box["height"]
    return x, y


def latlon_to_minimap_xy(lat: float, lon: float, box: dict) -> tuple[float, float]:
    # If the box looks like a full-screen guess map (wide enough), use direct
    # Web Mercator projection — works when Google Maps internal state isn't
    # available (e.g., Duels where __geoMap isn't reliably captured).
    if box["width"] > 600:
        return latlon_to_world_mercator(lat, lon, box)
    # Otherwise empirical calibration tuned for small corner minimaps.
    LON_MIN, LON_MAX = -124.34, 156.80
    MERC_Y_MIN, MERC_Y_MAX = 0.2163, 0.6713
    lat_clamped = max(-85.0511, min(85.0511, lat))
    sin_lat = math.sin(math.radians(lat_clamped))
    merc_y = 0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)
    x_frac = (lon - LON_MIN) / (LON_MAX - LON_MIN)
    y_frac = (merc_y - MERC_Y_MIN) / (MERC_Y_MAX - MERC_Y_MIN)
    x = box["x"] + max(0.0, min(1.0, x_frac)) * box["width"]
    y = box["y"] + max(0.0, min(1.0, y_frac)) * box["height"]
    return x, y


async def read_map_bounds(page: Page) -> dict | None:
    # Uses the actually-visible guess map (in Duels there are multiple Map
    # instances on the page; we want the one inside the guess-map container).
    try:
        data = await page.evaluate(
            """() => {
                const map = (typeof window.__getGuessMap === 'function')
                    ? window.__getGuessMap()
                    : window.__geoMap;
                if (!map || typeof map.getBounds !== 'function') return null;
                const b = map.getBounds();
                if (!b) return null;
                const ne = b.getNorthEast(), sw = b.getSouthWest();
                return {
                    north: ne.lat(), south: sw.lat(),
                    east: ne.lng(), west: sw.lng(),
                };
            }"""
        )
        return data
    except Exception:
        return None


async def expand_minimap(page: Page) -> None:
    # Click the maximize/expand button (arrow icon) to make the minimap full-screen.
    for sel in [
        "button[data-qa='guess-map-expand']",
        "button[aria-label*='Expand' i]",
        "button[aria-label*='fullscreen' i]",
        "button[class*='expand']",
        "[class*='guess-map'] button[class*='arrow']",
    ]:
        btn = await page.query_selector(sel)
        if btn:
            try:
                await btn.click()
                await human_delay(0.4, 0.8)
                return
            except Exception:
                continue


async def place_guess(page: Page, guess: Guess) -> bool:
    minimap = await find_minimap(page)
    if not minimap:
        # Sometimes the minimap appears with a small delay (post-render or
        # transition state). Retry a few times before giving up.
        for attempt in range(8):
            await asyncio.sleep(0.5)
            minimap = await find_minimap(page)
            if minimap:
                print(f"  minimap found after {attempt+1} retries")
                break
    if not minimap:
        print("  minimap not found after retries — skipping click")
        return False

    # Hover minimap so GeoGuessr expands it to its interactive (bigger) size.
    hover_x = minimap["x"] + minimap["width"] / 2
    hover_y = minimap["y"] + minimap["height"] / 2
    await human_mouse_move(page, hover_x, hover_y, steps=20)
    await human_delay(0.3, 0.5)
    # Re-measure after hover-expand.
    minimap = await find_minimap(page, debug=True) or minimap

    # TRUQUE DE MESTRE: centrar/fazer zoom programaticamente. Tenta zoom 10 (preciso);
    # se as bounds não contiverem o alvo (pan falhou ou o minimap é global), reduz
    # o zoom para 7 e depois 5. Evita drifts catastróficos (>1000 km).
    async def _try_center(zoom: int) -> dict | None:
        try:
            await page.evaluate(
                """([lat, lng, z]) => {
                    const map = (typeof window.__getGuessMap === 'function')
                        ? window.__getGuessMap()
                        : window.__geoMap;
                    if (map && typeof map.setCenter === 'function') {
                        map.setZoom(z);
                        map.setCenter({lat: lat, lng: lng});
                    }
                }""",
                [guess.latitude, guess.longitude, zoom]
            )
            await human_delay(0.2, 0.4)
            return await read_map_bounds(page)
        except Exception as e:
            print(f"  [map debug] pan@z{zoom} failed: {e}")
            return None

    # In duels, the guess map can be at any zoom/center (e.g., remembered
    # from last round, or default not exactly world view). Reset it to a
    # known world view first via mouse-wheel zoom-out (15 steps is enough
    # for any starting zoom), then do Mercator click + zoom-in for precision.
    if IS_MULTIPLAYER and minimap["width"] > 600:
        center_x = minimap["x"] + minimap["width"] / 2
        center_y = minimap["y"] + minimap["height"] / 2
        # 1. Zoom out to world view from wherever the map currently is.
        await page.mouse.move(center_x, center_y)
        await human_delay(0.1, 0.2)
        print(f"  [map] duels — resetting to world view (zoom out)")
        for _ in range(15):
            try:
                await page.mouse.wheel(0, 240)  # positive = zoom out
            except Exception:
                break
            await asyncio.sleep(0.08)
        await human_delay(0.4, 0.6)
        # 2. Compute target pixel assuming world Mercator (-180..180, ±85).
        x, y = latlon_to_world_mercator(guess.latitude, guess.longitude, minimap)
        print(f"  [map] Mercator target ({x:.0f}, {y:.0f}) for ({guess.latitude:.2f}, {guess.longitude:.2f})")
        # 3. Move cursor to target so subsequent zoom anchors on it.
        await page.mouse.move(x, y)
        await human_delay(0.25, 0.4)
        # 4. Zoom in for precision (cursor stays over same lat/lon while
        #    Google Maps zooms centred on cursor).
        ZOOM_STEPS = 7
        for i in range(ZOOM_STEPS):
            try:
                await page.mouse.wheel(0, -240)  # negative = zoom in
            except Exception as e:
                print(f"  [map] zoom wheel step {i+1} failed: {e}")
                break
            await asyncio.sleep(0.18)
        await human_delay(0.35, 0.55)
        print(f"  [map] zoomed in — clicking at ({x:.0f}, {y:.0f})")
    else:
        bounds = None
        for z in (10, 7, 5):
            b = await _try_center(z)
            if not b:
                continue
            in_lat = b["south"] <= guess.latitude <= b["north"]
            lon_ok = (b["west"] <= guess.longitude <= b["east"]) if b["east"] >= b["west"] else (
                guess.longitude >= b["west"] or guess.longitude <= b["east"]
            )
            if in_lat and lon_ok:
                bounds = b
                if z != 10:
                    print(f"  [map] target outside zoom-10 viewport; using zoom {z}")
                break
        if bounds:
            x, y = latlon_to_minimap_xy_bounds(guess.latitude, guess.longitude, minimap, bounds)
        else:
            print("  map bounds: unavailable — using empirical fallback")
            x, y = latlon_to_minimap_xy(guess.latitude, guess.longitude, minimap)
    x = max(minimap["x"] + 2, min(minimap["x"] + minimap["width"] - 2, x))
    y = max(minimap["y"] + 2, min(minimap["y"] + minimap["height"] - 2, y))

    await human_mouse_move(page, x, y, steps=30)
    await human_delay(0.1, 0.2)
    await page.mouse.click(x, y)
    await human_delay(0.2, 0.4)

    # Confirm guess button.
    for sel in [
        "button[data-qa='perform-guess']",
        "button.button_variantPrimary__*",
        "button:has-text('Guess')",
        "button:has-text('Place your pin')",
    ]:
        btn = await page.query_selector(sel)
        if btn:
            await btn.click()
            return True
    await page.keyboard.press("Space")
    return True


async def read_round_from_api(page: Page, round_index: int) -> dict | None:
    # Returns {actual: (lat,lng), placed: (lat,lng) or None} from GeoGuessr's API.
    url = page.url
    m = re.search(r"/(?:game|results)/([A-Za-z0-9]+)", url)
    if not m:
        return None
    token = m.group(1)
    try:
        data = await page.evaluate(
            """async (token) => {
                const r = await fetch(`/api/v3/games/${token}`, {credentials: 'include'});
                if (!r.ok) return null;
                return await r.json();
            }""",
            token,
        )
    except Exception:
        return None
    if not data or "rounds" not in data:
        return None
    rounds = data["rounds"]
    if round_index - 1 >= len(rounds):
        return None
    r = rounds[round_index - 1]
    try:
        actual = (float(r["lat"]), float(r["lng"]))
    except (KeyError, TypeError, ValueError):
        return None
    placed = None
    try:
        player = data.get("player") or {}
        guesses = player.get("guesses") or []
        if round_index - 1 < len(guesses):
            g = guesses[round_index - 1]
            placed = (float(g["lat"]), float(g["lng"]))
    except (KeyError, TypeError, ValueError):
        pass
    return {"actual": actual, "placed": placed}


def append_log(entry: dict) -> None:
    data = []
    if LOG_FILE.exists():
        try:
            data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = []
    data.append(entry)
    LOG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def recent_wrong_countries(n: int = 8) -> list[str]:
    """Return the distinct countries the bot guessed wrong in the last `n` rounds.
    Used to warn the prompt against repeating a recent mistake (same country gets
    picked again even after a miss — session-level adaptive penalty)."""
    if not LOG_FILE.exists():
        return []
    try:
        data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for r in reversed(data[-n:]):
        if r.get("country_hit") is False:
            g = (r.get("guess") or {}).get("country", "")
            if g and g != "?" and g not in seen:
                seen.add(g)
                out.append(g)
    return out


def analyze_ground_color(pil_img) -> str | None:
    """Analyse dominant soil/ground color in the middle band of the image.
    Returns a descriptive hint string for the LLM, or None if inconclusive.
    Useful for distinguishing South American countries by soil color."""
    try:
        import cv2
        w, h = pil_img.size
        # Middle third of the image — avoids sky and car/UI chrome
        band = pil_img.crop((0, int(h * 0.35), w, int(h * 0.68)))
        hsv = cv2.cvtColor(np.array(band), cv2.COLOR_RGB2HSV)
        # Ignore very dark pixels (shadows, asphalt) — V < 50
        bright = hsv[:, :, 2] > 50
        if bright.sum() < 500:
            return None
        hues = hsv[:, :, 0][bright].astype(float)   # OpenCV H: 0-179
        sats = hsv[:, :, 1][bright].astype(float)   # S: 0-255
        mean_hue = float(np.mean(hues))
        mean_sat = float(np.mean(sats))
        # Red/orange soil (terra roxa): H<20 or H>160, S>70
        if (mean_hue < 20 or mean_hue > 160) and mean_sat > 70:
            return (
                "BIOME DETECTED — RED/ORANGE SOIL. This biome spans multiple continents. "
                "DO NOT default to Brazil. Discriminate using the cues below.\n"
                "\n"
                "Candidate regions (with discriminating signals):\n"
                "- Brazil terra roxa (São Paulo, Paraná, Minas Gerais, Mato Grosso): "
                "coffee/sugarcane/banana plantations, Portuguese signs, white Mercosul "
                "plates with blue top stripe (post-2018), 'PARE' stop signs (NOT 'STOP' "
                "or 'ALTO'), right-hand drive.\n"
                "- Australia red center (NT, WA, SA outback): eucalyptus and spinifex "
                "tussocks, gum trees, yellow diamond 'kangaroo crossing' signs, English "
                "signs, AU plates with state name (NSW, QLD, WA), left-hand drive, "
                "Toyota LandCruisers / Hilux dominant.\n"
                "- US Southwest red rock (Arizona, New Mexico, southern Utah, SW "
                "Colorado): saguaro/cholla cacti (Arizona only), mesa/butte landforms, "
                "English signs, US-style green directional signage, state-specific "
                "plates ('ARIZONA - GRAND CANYON STATE'), right-hand drive, double yellow "
                "centerlines.\n"
                "- Madagascar: red laterite, baobab trees, rice paddies, French + "
                "Malagasy script, very rough dirt roads, zebu cattle, right-hand drive.\n"
                "- West African Sahel (Mali, Niger, Burkina Faso, parts of Senegal, "
                "Chad): drier red-orange tone, baobabs, French signs, mud-brick "
                "architecture, mosques, Toyota Hilux pickups, right-hand drive.\n"
                "- India red soil (Karnataka, Tamil Nadu, Andhra Pradesh, parts of "
                "Odisha): coconut palms, banana, Dravidian scripts (Tamil/Kannada/Telugu "
                "— distinctive curly characters), IN plates (white-on-black private, "
                "yellow-on-black commercial, state code prefix like 'TN' / 'KA'), "
                "left-hand drive.\n"
                "- Ethiopian highlands: red soil with eucalyptus plantations, Amharic "
                "script (Ge'ez characters — visually unmistakable), thatched rondavels, "
                "right-hand drive.\n"
                "- Southern Africa (Botswana, Zimbabwe, parts of South Africa, "
                "Mozambique): red soil mixed with savanna grass, baobabs, English/"
                "Portuguese, left-hand drive (RSA/Bots/Zim/Moz).\n"
                "\n"
                "DISCRIMINATE via: script on signs (Latin/Arabic/Dravidian/Amharic/"
                "Khmer), license plate format and color, vegetation specifics "
                "(saguaro = SW US only; baobab = Africa/Madagascar; eucalyptus + dry "
                "= Australia/Ethiopia/Iberia), driving side, road signage style, stop "
                "sign text. Color alone is NOT enough."
            )
        # Yellow-brown dry land: H 20-35, moderate sat
        if 20 <= mean_hue <= 35 and 40 < mean_sat <= 110:
            return (
                "BIOME DETECTED — DRY YELLOW-BROWN GROUND (semi-arid grassland or "
                "steppe). This biome occurs on every continent. DO NOT default to "
                "Argentina. Discriminate via the cues below.\n"
                "\n"
                "Candidate regions:\n"
                "- Argentine Pampas: flat grassland, eucalyptus windbreaks, Spanish "
                "signs ('Camino', 'Estancia'), white Mercosul plates with blue top "
                "stripe, cattle on electric fences, 'PARE' stop signs.\n"
                "- Uruguay countryside: same Pampas biome, Spanish but different town "
                "names, older white plates or newer Mercosul, less density than Argentina.\n"
                "- Chile arid north (Atacama region) / Peru coastal desert: extreme "
                "arid, Andes visible to the east, Spanish, distinctive plates (Chile "
                "white plates with red 'CL' block; Peru solid white).\n"
                "- Spain meseta (Castilla, Aragón, Extremadura): olive groves and "
                "almond trees, white-stuccoed buildings with red roof tiles, Spanish "
                "signs, EU plates with blue strip and 'E', right-hand drive.\n"
                "- Portugal interior (Alentejo, Beira): cork oaks, olive trees, "
                "Portuguese signs, EU plates with yellow strip and 'P' on the RIGHT "
                "side (unlike Spain), right-hand drive.\n"
                "- Turkey Anatolian plateau: mosques on the horizon, Turkish in Latin "
                "script with ı/ğ/ş/ç/ö/ü, TR plates (white with red TR block on left), "
                "distinctive blue road signs.\n"
                "- Central Asia (Kazakhstan, Mongolia, Kyrgyzstan, Uzbekistan): vast "
                "steppe, mostly empty horizon, Cyrillic + national scripts (Mongolia "
                "uses traditional script + Cyrillic), gers/yurts, Lada / Korean used "
                "cars common, very low-quality roads.\n"
                "- Australia outback: red-yellow earth, spinifex, eucalyptus, AU plates, "
                "left-hand drive, dead-flat highways with single yellow line.\n"
                "- US Great Plains, Wyoming, Nevada, Montana, Idaho: vast plains, "
                "wooden barns, US state plates, English signs, large double yellow "
                "centerlines on rural roads, white mailboxes on poles by the road.\n"
                "- Mexico north (Sonora, Chihuahua, Coahuila, Durango): cactus "
                "(saguaro, organ pipe, prickly pear), Spanish, MX plates with state-"
                "specific colorful designs, 'TOPE' speed bumps, 'ALTO' stop signs "
                "(NOT 'PARE'), Pemex gas stations.\n"
                "- North Africa (Morocco, Tunisia, Algeria, Mauritania, Libya): mix "
                "of French and Arabic signs, palm groves, mosques, Moroccan plates "
                "white with Arabic numerals, right-hand drive.\n"
                "- South African Karoo: similar dry steppe, English + Afrikaans "
                "bilingual signs, ZA plates (white-on-black or province-coded), "
                "left-hand drive.\n"
                "- Iran plateau: Persian (Farsi) script, Iranian plates, mosques.\n"
                "- Northern China (Inner Mongolia, Gansu, Xinjiang): Chinese characters "
                "and/or Uyghur Arabic, distinctive Chinese plates blue with white text.\n"
                "\n"
                "DISCRIMINATE via: script (Latin/Arabic/Cyrillic/Persian/Chinese/Hebrew), "
                "vegetation specifics (saguaro = SW US/N Mexico ONLY; olive/almond "
                "= Iberia/Mediterranean; spinifex+gum = Australia; cork oak = Portugal/W "
                "Spain), plate format and country code, road sign style, driving side, "
                "stop sign text ('PARE' / 'ALTO' / 'STOP' / 'ARRÊT')."
            )
        # Dense green vegetation: H 35-90, high sat
        if 35 <= mean_hue <= 90 and mean_sat > 90:
            return (
                "BIOME DETECTED — DENSE GREEN TROPICAL VEGETATION. This is the highest-"
                "confusion biome — it spans Latin America, SE Asia, tropical Africa, "
                "the Caribbean, and Pacific islands. DO NOT default to Brazil. The fact "
                "that you see lush greenery alone gives essentially zero country signal. "
                "Discriminate using the cues below.\n"
                "\n"
                "LATIN AMERICA tropical:\n"
                "- Brazil (Amazon, Atlantic Forest, Mata Atlântica, Cerrado): "
                "Portuguese, Mercosul plates (white + blue top stripe), 'PARE' stop "
                "signs, octagonal Brazilian road signs, right-hand drive, distinctive "
                "yellow taxis in cities.\n"
                "- Colombia / Ecuador / Peru / Bolivia lowlands: Spanish, white CO/EC/"
                "PE plates (Peru has 'PERÚ' on plate; Colombia has 'COLOMBIA'), 'PARE' "
                "or 'ALTO' stop signs, mountainous backdrop common.\n"
                "- Costa Rica / Panama: brightly-painted small houses, Spanish, "
                "distinctive yellow CR plates, 'Alto' stop signs, surprising bilingual "
                "English in tourist areas.\n"
                "- Guatemala, Honduras, El Salvador, Nicaragua: Spanish, Mesoamerican "
                "(Mayan) ethnic influence visible, poorer rural roads, distinctive "
                "'pollero' chicken buses (re-used US schoolbuses, brightly painted).\n"
                "- Southern Mexico (Chiapas, Tabasco, Veracruz, Yucatán, Quintana Roo): "
                "Spanish, 'TOPE' speed bumps EVERYWHERE, 'ALTO' stop signs (NOT 'PARE'), "
                "MX state-coded colorful plates, Volkswagen Beetles still common in "
                "older towns, distinctive Mayan ruins or pyramids occasionally visible.\n"
                "- Cuba, Dominican Republic, Puerto Rico: Spanish, distinctive Cuban "
                "American classic cars (Cuba), Caribbean architecture.\n"
                "- Haiti: French + Haitian Creole, very poor roads, distinctive "
                "tap-tap painted vans.\n"
                "\n"
                "SE ASIA tropical:\n"
                "- Cambodia: Khmer script (rounded, distinctive: ខ្មែរ), tuk-tuks, "
                "dilapidated French colonial buildings, red-orange laterite dirt "
                "roads, right-hand drive.\n"
                "- Vietnam: Vietnamese (Latin alphabet with MANY diacritics — ơ, ư, "
                "đ, tone marks ̀ ́ ̃ ̉ ̣), motorbike swarms, distinctive narrow tall "
                "shophouses, yellow-on-blue or white VN plates, right-hand drive.\n"
                "- Thailand: Thai script (curly, no spaces between words: ภาษาไทย), "
                "distinctive ornate Buddhist temple roofs (wat), yellow/red taxis, "
                "LEFT-HAND drive, distinctive guardrails painted red-and-white.\n"
                "- Laos: Lao script (similar to Thai but more rounded), French signs "
                "occasionally, very rural, right-hand drive.\n"
                "- Myanmar: Burmese script (round circular characters: မြန်မာ), "
                "right-hand drive but RHD cars (anomaly), pagodas.\n"
                "- Indonesia: Indonesian/Bahasa in Latin ('Jalan' = street), mosques "
                "with distinctive domes, scooter density, ID plates with red letters "
                "on white background and area code prefix, LEFT-HAND drive.\n"
                "- Malaysia: bilingual Malay (Latin) + English on signs, distinctive "
                "yellow regulatory signs, mosques + Chinese temples mixed, LEFT-HAND "
                "drive, MY plates with black-on-white reverse format.\n"
                "- Singapore: English-dominant signs, very clean, LHD, distinctive SG "
                "plates white front yellow rear.\n"
                "- Philippines: English + Tagalog/Filipino signs, distinctive 'jeepney' "
                "and 'tricycle' (motorcycle+sidecar) vehicles, RIGHT-HAND drive, white "
                "plates with blue text, basketball courts in every rural village.\n"
                "- Sri Lanka: Sinhala script (rounded loops: සිංහල) + Tamil, LEFT-HAND "
                "drive, white plates.\n"
                "- Bangladesh: Bengali script (distinctive horizontal bar above "
                "characters: বাংলা), rickshaws, LEFT-HAND drive, very dense.\n"
                "- Southern India (Kerala, Tamil Nadu): Dravidian scripts, coconut "
                "palms, backwaters, LEFT-HAND drive, IN plates.\n"
                "\n"
                "TROPICAL AFRICA:\n"
                "- DRC, Congo, Cameroon, Gabon, CAR: French signs, distinctive "
                "African vernacular architecture, banana plantations, very poor "
                "roads, RIGHT-HAND drive (most). Cameroon split English/French.\n"
                "- Uganda, Kenya (west), Tanzania (highlands), Rwanda, Burundi: "
                "English (Uganda/Kenya/Rwanda) or Swahili+English, distinctive matatu "
                "minibuses (Kenya), Rwanda very clean by African standards, LEFT-HAND "
                "drive (Kenya/Uganda/Tanzania/Rwanda — ex-British).\n"
                "- Madagascar: French + Malagasy, distinctive zebu cattle, baobabs, "
                "RIGHT-HAND drive.\n"
                "- Mozambique: Portuguese (only African country with Portuguese as "
                "national + tropical green biome), LEFT-HAND drive, distinct from "
                "Brazilian Portuguese accent in voice but visually distinguishable "
                "from Brazil by plates (MZ tall white plates) and architecture.\n"
                "- Ghana, Ivory Coast, Sierra Leone, Liberia, Nigeria south: English "
                "(Ghana/SL/Liberia/Nigeria) or French (Ivory Coast), RIGHT-HAND drive, "
                "distinctive West African market scenes.\n"
                "\n"
                "PACIFIC / CARIBBEAN:\n"
                "- Hawaii (US state): English, US plates, RIGHT-HAND drive, distinctive "
                "Hawaiian volcanic landscape.\n"
                "- Fiji, Samoa, Tonga, Vanuatu: English + native language, LEFT-HAND "
                "drive (Fiji), very small island feel.\n"
                "- French Polynesia, New Caledonia: French signs.\n"
                "\n"
                "DISCRIMINATE FIRST BY SCRIPT — it is the highest-information signal:\n"
                "- Khmer (rounded sub-script characters) → Cambodia\n"
                "- Thai (curly, no spaces) → Thailand\n"
                "- Lao (similar to Thai, rounder) → Laos\n"
                "- Vietnamese (Latin with stacked tone marks) → Vietnam\n"
                "- Burmese (perfect circles) → Myanmar\n"
                "- Bengali (horizontal top bar) → Bangladesh\n"
                "- Sinhala (loopy) → Sri Lanka\n"
                "- Tamil/Kannada/Telugu (Dravidian curly) → South India / Sri Lanka\n"
                "- Chinese characters → mainland China / Taiwan / Hong Kong / Singapore\n"
                "- Arabic → North Africa / Middle East (rare in tropical biome)\n"
                "- Portuguese (with ã, õ, ç) → Brazil OR Mozambique (distinguish via "
                "plate format + driving side: Brazil RHD, Mozambique LHD)\n"
                "- Spanish (with ñ, ¿, ¡) → Latin America (further discriminate by "
                "stop-sign text: 'ALTO' = Mexico/Central America; 'PARE' = South "
                "America)\n"
                "- French → DRC/Gabon/Madagascar/Cameroon/Ivory Coast/Haiti (Africa "
                "+ Haiti only — French Guiana also possible)\n"
                "- English-only → Caribbean / Philippines (with Tagalog) / former "
                "British Africa (with native language) / Hawaii / Australia (rare for "
                "lush tropical) / NZ north.\n"
                "\n"
                "SECONDARY DISCRIMINATORS:\n"
                "- Driving side: LHD = Cambodia/Vietnam/Indonesia/Philippines/most "
                "Latin America/most tropical Africa (DRC, Madagascar, etc.); "
                "RHD = Thailand/Malaysia/Sri Lanka/Bangladesh/UK ex-colonies (Kenya, "
                "Uganda, Tanzania, Mozambique, S. India).\n"
                "- Plate format: distinctive per country.\n"
                "- Stop-sign text: 'PARE' (Brazil/most SA), 'ALTO' (Mex/CR/El Sal/"
                "Honduras/Nicaragua/Guatemala), 'STOP' (most others), 'ARRÊT' (Quebec).\n"
                "- Vehicle types: jeepney/tricycle = Philippines; tuk-tuk + rough "
                "roads = Cambodia; motorbike swarms + narrow shophouses = Vietnam; "
                "matatu = Kenya; chicken bus = Central America; classic cars = Cuba.\n"
                "\n"
                "Lush green vegetation alone tells you the LATITUDE BAND. It does NOT "
                "tell you the country. Use the discriminators above before deciding."
            )
    except Exception:
        pass
    return None


def compute_confusion_pairs(min_count: int = 3, max_error_km: float = 1500.0) -> list[tuple[str, str, int]]:
    """Return (guessed_wrong, actual_correct, count) for systematic near-miss errors.
    Only considers rounds where error < max_error_km (right region, wrong country).
    Called once at startup and cached."""
    if not LOG_FILE.exists():
        return []
    try:
        data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
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

def _load_confusion_pairs() -> None:
    global _CONFUSION_PAIRS, _last_confusion_refresh_count
    _CONFUSION_PAIRS = compute_confusion_pairs()
    _last_confusion_refresh_count = count_rounds_in_log()
    if _CONFUSION_PAIRS:
        print(f"[confusion] {len(_CONFUSION_PAIRS)} pairs loaded (log={_last_confusion_refresh_count}): "
              + ", ".join(f"{g}->{a}({c}x)" for g, a, c in _CONFUSION_PAIRS[:5]))


_ocr_reader = None
_ocr_reader_latin = None
_ocr_reader_cjk = None
try:
    import easyocr
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Reader principal: inglês + cirílico (russo). Cada script só mistura com EN.
        _ocr_reader = easyocr.Reader(['en', 'ru'], gpu=True, verbose=False)
        # Reader auxiliar: línguas latinas romanas (PT/ES/FR) + japonês juntos não funciona,
        # então dois readers separados wrapped em try/except.
        try:
            _ocr_reader_latin = easyocr.Reader(['en', 'pt', 'es', 'fr'], gpu=True, verbose=False)
        except Exception:
            pass
        try:
            _ocr_reader_cjk = easyocr.Reader(['ja', 'en'], gpu=True, verbose=False)
        except Exception:
            pass
except ImportError:
    pass

def extract_text_from_image(png_bytes: bytes) -> str:
    if _ocr_reader is None:
        return ""
    try:
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(png_bytes)).convert("RGB")
        w, h = img.size
        # Cortar os 15% do topo e os 20% de baixo do ecrã para limpar o lixo do GeoGuessr (HUD, avatars, score)
        # E cortar os 20% da direita para ignorar o minimapa e a bússola lateral
        crop_box = (0, int(h * 0.15), int(w * 0.80), int(h * 0.80))
        img_cropped = img.crop(crop_box)

        # Multi-crop: run OCR on several windows so small/peripheral text is caught.
        # UPDATE: Ignorar o extremo direito/sul onde aparece a bússola/minimapa do GeoGuessr
        crops = [
            img_cropped,  # full middle band
            img.crop((0, int(h * 0.55), int(w * 0.80), int(h * 0.90))),  # bottom strip sem o minimapa
        ]
        seen: set[str] = set()
        out: list[str] = []
        readers = [r for r in (_ocr_reader, _ocr_reader_latin, _ocr_reader_cjk) if r is not None]
        for c in crops:
            arr = np.array(c)
            for reader in readers:
                try:
                    res = reader.readtext(arr, detail=0)
                except Exception:
                    continue
                for t in res:
                    s = t.strip()
                    if len(s) < 3:
                        continue
                    key = s.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(s)
        return " | ".join(out)
    except Exception as e:
        print(f"  [ocr error] {e}")
        return ""


# Unicode-block → candidate countries. Used to constrain the guess when a
# non-Latin script is detected in the OCR output.
_SCRIPT_COUNTRIES: dict[str, tuple[str, ...]] = {
    "cyrillic": ("Russia", "Ukraine", "Belarus", "Bulgaria", "Serbia", "North Macedonia",
                 "Kazakhstan", "Kyrgyzstan", "Mongolia", "Tajikistan"),
    "greek":    ("Greece", "Cyprus"),
    "hebrew":   ("Israel",),
    "arabic":   ("Saudi Arabia", "Egypt", "United Arab Emirates", "Jordan", "Iraq", "Syria",
                 "Lebanon", "Tunisia", "Morocco", "Algeria", "Libya", "Qatar", "Kuwait",
                 "Oman", "Yemen", "Sudan"),
    "thai":     ("Thailand",),
    "lao":      ("Laos",),
    "khmer":    ("Cambodia",),
    "devanagari": ("India", "Nepal"),
    "bengali":  ("Bangladesh", "India"),
    "tamil":    ("India", "Sri Lanka"),
    "sinhala":  ("Sri Lanka",),
    "burmese":  ("Myanmar",),
    "chinese":  ("China", "Taiwan", "Hong Kong", "Macau", "Singapore"),
    "japanese": ("Japan",),
    "korean":   ("South Korea", "North Korea"),
    "georgian": ("Georgia",),
    "armenian": ("Armenia",),
    "ethiopic": ("Ethiopia", "Eritrea"),
}


def detect_script(text: str) -> str | None:
    """Return the dominant non-Latin script in `text`, or None if only Latin/digits/punct."""
    if not text:
        return None
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        # Script ranges (coarse but sufficient for GeoGuessr)
        if   0x0400 <= cp <= 0x04FF: counts["cyrillic"]   = counts.get("cyrillic", 0) + 1
        elif 0x0370 <= cp <= 0x03FF: counts["greek"]      = counts.get("greek", 0) + 1
        elif 0x0590 <= cp <= 0x05FF: counts["hebrew"]     = counts.get("hebrew", 0) + 1
        elif 0x0600 <= cp <= 0x06FF: counts["arabic"]     = counts.get("arabic", 0) + 1
        elif 0x0E00 <= cp <= 0x0E7F: counts["thai"]       = counts.get("thai", 0) + 1
        elif 0x0E80 <= cp <= 0x0EFF: counts["lao"]        = counts.get("lao", 0) + 1
        elif 0x1780 <= cp <= 0x17FF: counts["khmer"]      = counts.get("khmer", 0) + 1
        elif 0x0900 <= cp <= 0x097F: counts["devanagari"] = counts.get("devanagari", 0) + 1
        elif 0x0980 <= cp <= 0x09FF: counts["bengali"]    = counts.get("bengali", 0) + 1
        elif 0x0B80 <= cp <= 0x0BFF: counts["tamil"]      = counts.get("tamil", 0) + 1
        elif 0x0D80 <= cp <= 0x0DFF: counts["sinhala"]    = counts.get("sinhala", 0) + 1
        elif 0x1000 <= cp <= 0x109F: counts["burmese"]    = counts.get("burmese", 0) + 1
        elif 0x4E00 <= cp <= 0x9FFF: counts["chinese"]    = counts.get("chinese", 0) + 1
        elif 0x3040 <= cp <= 0x30FF: counts["japanese"]   = counts.get("japanese", 0) + 1
        elif 0xAC00 <= cp <= 0xD7AF: counts["korean"]     = counts.get("korean", 0) + 1
        elif 0x10A0 <= cp <= 0x10FF: counts["georgian"]   = counts.get("georgian", 0) + 1
        elif 0x0530 <= cp <= 0x058F: counts["armenian"]   = counts.get("armenian", 0) + 1
        elif 0x1200 <= cp <= 0x137F: counts["ethiopic"]   = counts.get("ethiopic", 0) + 1
    if not counts:
        return None
    # At least 3 chars of the script to consider it real (avoid single-char OCR noise).
    top_script, top_count = max(counts.items(), key=lambda kv: kv[1])
    if top_count < 3:
        return None
    return top_script


# Literal country names and unique long tokens that, if present verbatim in OCR,
# identify the country with high confidence. Values are the canonical country names.
_OCR_COUNTRY_TOKENS: dict[str, str] = {
    "kyrgyzstan": "Kyrgyzstan", "кыргызстан": "Kyrgyzstan",
    "deutschland": "Germany", "germany": "Germany",
    "österreich": "Austria", "austria": "Austria",
    "schweiz": "Switzerland", "suisse": "Switzerland",
    "polska": "Poland", "poland": "Poland",
    "česko": "Czechia", "ceska": "Czechia", "czechia": "Czechia",
    "magyarország": "Hungary", "hungary": "Hungary",
    "românia": "Romania", "romania": "Romania",
    "hrvatska": "Croatia", "croatia": "Croatia",
    "slovenija": "Slovenia", "slovensko": "Slovakia",
    "nederland": "Netherlands", "netherlands": "Netherlands",
    "belgië": "Belgium", "belgique": "Belgium", "belgium": "Belgium",
    "sverige": "Sweden", "sweden": "Sweden",
    "norge": "Norway", "norway": "Norway",
    "suomi": "Finland", "finland": "Finland",
    "danmark": "Denmark", "denmark": "Denmark",
    "ísland": "Iceland", "iceland": "Iceland",
    "eesti": "Estonia", "estonia": "Estonia",
    "latvija": "Latvia", "latvia": "Latvia",
    "lietuva": "Lithuania", "lithuania": "Lithuania",
    "україна": "Ukraine", "ukraine": "Ukraine",
    "беларусь": "Belarus", "belarus": "Belarus",
    "россия": "Russia",
    "қазақстан": "Kazakhstan", "kazakhstan": "Kazakhstan",
    "türkiye": "Türkiye", "turkiye": "Türkiye", "turkey": "Türkiye",
    "hellas": "Greece", "greece": "Greece", "ελλάδα": "Greece",
    "portugal": "Portugal", "españa": "Spain", "spain": "Spain",
    "france": "France", "italia": "Italy", "italy": "Italy",
    "brasil": "Brazil", "brazil": "Brazil",
    "argentina": "Argentina", "uruguay": "Uruguay",
    "chile": "Chile", "paraguay": "Paraguay", "bolivia": "Bolivia",
    "perú": "Peru", "peru": "Peru", "ecuador": "Ecuador",
    "colombia": "Colombia", "venezuela": "Venezuela",
    "méxico": "Mexico", "mexico": "Mexico",
    "nippon": "Japan", "日本": "Japan",
    "한국": "South Korea", "대한민국": "South Korea",
    "ประเทศไทย": "Thailand", "thailand": "Thailand",
    "malaysia": "Malaysia", "indonesia": "Indonesia",
    "pilipinas": "Philippines", "philippines": "Philippines",
    "việt nam": "Vietnam", "vietnam": "Vietnam",
    "भारत": "India", "india": "India",
    "australia": "Australia", "aotearoa": "New Zealand",
    "south africa": "South Africa", "suid-afrika": "South Africa",
    "kenya": "Kenya", "nigeria": "Nigeria", "ghana": "Ghana",
}


# EU license plate 2-letter country codes. Only matched as whole words (word boundary)
# to avoid false positives from random abbreviations.
_EU_PLATE_CODES: dict[str, str] = {
    "FR": "France", "DE": "Germany", "RO": "Romania", "IT": "Italy", "ES": "Spain",
    "AT": "Austria", "NL": "Netherlands", "BE": "Belgium", "PL": "Poland", "CZ": "Czechia",
    "HU": "Hungary", "SK": "Slovakia", "SI": "Slovenia", "HR": "Croatia", "BG": "Bulgaria",
    "GR": "Greece", "PT": "Portugal", "GB": "United Kingdom", "SE": "Sweden", "NO": "Norway",
    "FI": "Finland", "DK": "Denmark", "IE": "Ireland", "LT": "Lithuania", "LV": "Latvia",
    "EE": "Estonia", "LU": "Luxembourg", "CH": "Switzerland", "TR": "Turkey",
}


def detect_country_from_ocr(text: str) -> str | None:
    """Return a country name if a verbatim country-identifying token appears in OCR text."""
    if not text:
        return None
    t = text.lower()
    for tok, country in _OCR_COUNTRY_TOKENS.items():
        if tok in t:
            return country
    # Check for EU license plate codes as isolated uppercase tokens (e.g. "FR" on blue strip)
    import re
    for code, country in _EU_PLATE_CODES.items():
        if re.search(rf'\b{code}\b', text):
            return country
    return None

_chroma_client = None
_chroma_collection = None
_chroma_car_collection = None

def get_chroma_collection():
    global _chroma_client, _chroma_collection, _chroma_car_collection
    if _chroma_client is None:
        try:
            _chroma_client = chromadb.PersistentClient(path="./vetores_db")
            embedding_function = OpenCLIPEmbeddingFunction()
            _chroma_collection = _chroma_client.get_collection(
                name="geoguessr_rondas",
                embedding_function=embedding_function
            )
            _chroma_car_collection = _chroma_client.get_collection(
                name="geoguessr_carmeta",
                embedding_function=embedding_function
            )
        except Exception:
            _chroma_collection = None
            _chroma_car_collection = None
    return _chroma_collection, _chroma_car_collection

# CLIP distance threshold for RAG matches. Calibrated on 344 indexed rounds:
# correct-country matches sit at p75≈0.13, max≈0.20; wrong-country matches
# overlap but reach >0.20 often. 0.20 keeps most useful matches, filters noise.
# The old value 0.8 accepted everything (CLIP distances never reach 0.8 in practice).
RAG_DISTANCE_THRESHOLD = 0.65


def build_rag_examples(current_image_path: Path, max_examples: int = 3) -> tuple[str, list[str], str | None, float | None, dict[str, float]]:
    collection, collection_car = get_chroma_collection()
    if not collection:
        return "", [], None, None, {}
    
    try:
        img = Image.open(current_image_path).convert("RGB")
        img_array = np.array(img)
        w, h = img.size
        car_box = (0, int(h * 0.70), w, h)
        car_array = np.array(img.crop(car_box))

        # Fetch 4x more results than needed so the country-cap can enforce diversity.
        fetch_n = max(max_examples * 4, 12)
        results = collection.query(
            query_images=[img_array],
            n_results=fetch_n,
            include=["distances", "metadatas"]
        )

        raw_matches = results['metadatas'][0] if results and results.get('metadatas') else []
        raw_distances = results['distances'][0] if results and results.get('distances') else []

        # Apply distance filter then country-cap (max 2 results per country) to prevent
        # a single over-represented country (e.g. Philippines) from dominating the top-k.
        _country_seen: dict[str, int] = {}
        matches: list = []
        distances: list = []
        for m, d in zip(raw_matches, raw_distances):
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
                        print(f"  [rag-veto] {m.get('country')} ({mc}) vetoed — majority continent is {_majority_cont}")
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
        if collection_car:
            car_results = collection_car.query(
                query_images=[car_array],
                n_results=1,
                include=["distances", "metadatas"]
            )
            car_matches = car_results['metadatas'][0] if car_results and car_results.get('metadatas') else []
            car_distances = car_results['distances'][0] if car_results and car_results.get('distances') else []
            if car_matches and car_distances:
                match = car_matches[0]
                dist = car_distances[0]
                if dist > RAG_DISTANCE_THRESHOLD:
                    print(f"  [rag] Rejeitado car meta ({match.get('country')}): distância {dist:.3f} > {RAG_DISTANCE_THRESHOLD}")
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
        if countries and LOG_FILE.exists():
            try:
                _log_data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
                _recent = _log_data[-100:]  # last 100 rounds
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
        print(f"  [rag error] {e}")
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


def count_rounds_in_log() -> int:
    if not LOG_FILE.exists():
        return 0
    try:
        return len(json.loads(LOG_FILE.read_text(encoding="utf-8")))
    except Exception:
        return 0


async def click_play_again(page: Page) -> bool:
    for sel in [
        "button[data-qa='play-again-button']",
        "button:has-text('Play again')",
        "button:has-text('PLAY AGAIN')",
        "button:has-text('Jogar novamente')",
    ]:
        btn = await page.query_selector(sel)
        if btn:
            try:
                await btn.click()
                return True
            except Exception:
                continue
    return False


_last_canvas_sig: str | None = None


async def _canvas_signature(page: Page) -> str | None:
    """Visual hash of a small central region of the panorama canvas. Robust to
    GeoGuessr's internal JS structure — changes whenever the visible panorama
    content changes."""
    try:
        canvas = await page.query_selector(".widget-scene-canvas, canvas.mapsConsumerUiSceneCoreScene__canvas")
        if not canvas:
            return None
        box = await canvas.bounding_box()
        if not box or box["width"] < 400:
            return None
        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2
        sample = await page.screenshot(
            clip={"x": max(0.0, cx - 100), "y": max(0.0, cy - 100), "width": 200, "height": 200},
            type="png",
        )
        return hashlib.md5(sample).hexdigest()
    except Exception:
        return None


async def wait_for_round(page: Page, timeout_s: int = 120) -> bool:
    global _last_canvas_sig
    start = time.time()
    deadline = start + timeout_s
    pause_lo, pause_hi = (1.5, 2.0) if IS_MULTIPLAYER else (3.0, 4.0)

    # In duels, between rounds the canvas first shows a results overlay (world
    # map with player pins), THEN the new panorama. We must NOT capture during
    # the overlay. Strategy:
    #  1. Enforce a 5s minimum delay (results overlay typically lasts ~5-7s)
    #  2. Then take two hashes 1.5s apart — if equal (stable) AND different
    #     from last round, we have a fully-loaded new panorama
    #  3. If still transitioning after 25s, give up and proceed anyway
    is_duels_subsequent = IS_MULTIPLAYER and _last_canvas_sig is not None
    waited_for_canvas_msg = False

    if is_duels_subsequent:
        # Minimum wait to let the results overlay clear.
        await asyncio.sleep(5.0)

    stability_deadline = start + 25.0 if is_duels_subsequent else None

    while time.time() < deadline:
        canvas = await page.query_selector(".widget-scene-canvas, canvas.mapsConsumerUiSceneCoreScene__canvas")
        if not canvas:
            if not waited_for_canvas_msg:
                print("  [wait] waiting for panorama canvas to appear...")
                waited_for_canvas_msg = True
            await asyncio.sleep(1.0)
            continue
        box = await canvas.bounding_box()
        if not box or box["width"] < 400:
            await asyncio.sleep(1.0)
            continue

        if is_duels_subsequent and stability_deadline and time.time() < stability_deadline:
            # First: if the results breakdown panel is still visible (DOM check),
            # we are guaranteed to be in a between-rounds state — keep waiting
            # regardless of canvas hash.
            if await is_results_overlay_visible(page):
                await asyncio.sleep(0.8)
                continue
            sig_a = await _canvas_signature(page)
            if not sig_a:
                await asyncio.sleep(0.8)
                continue
            await asyncio.sleep(1.5)
            # Re-check overlay after the sleep — it can appear/disappear.
            if await is_results_overlay_visible(page):
                await asyncio.sleep(0.8)
                continue
            sig_b = await _canvas_signature(page)
            if not sig_b:
                await asyncio.sleep(0.8)
                continue
            # Both must be equal (canvas stable = fully rendered) AND different
            # from last round's panorama.
            if sig_a == sig_b and sig_a != _last_canvas_sig:
                print(f"  [canvas] stable new content (detected after {time.time()-start:.1f}s)")
                _last_canvas_sig = sig_a
                await human_delay(pause_lo, pause_hi)
                return True
            # Either still changing (overlay/transition) or still the previous
            # panorama. Keep polling.
            await asyncio.sleep(0.8)
            continue

        if is_duels_subsequent and stability_deadline and time.time() >= stability_deadline:
            print(f"  [canvas] stable new content not confirmed within 25s — proceeding anyway")
        sig = await _canvas_signature(page)
        if sig:
            _last_canvas_sig = sig
        await human_delay(pause_lo, pause_hi)
        return True

    print(f"  [wait] timed out after {timeout_s}s")
    return False


async def is_results_overlay_visible(page: Page) -> bool:
    """Detect the inter-round results / breakdown panel that appears between
    duels rounds. The panel typically shows 'Round X', score breakdown, and a
    world map — capturing during it pollutes OCR with leaderboard text and
    map labels."""
    for sel in [
        "div:has-text('Game Breakdown')",
        "button:has-text('WATCH REPLAY')",
        "button:has-text('Watch replay')",
        "[data-qa='round-result']",
        "[data-qa='duels-round-result']",
        "[class*='round-result']",
        "[class*='roundResult']",
        "[class*='breakdown']",
        "[class*='Breakdown']",
    ]:
        try:
            elem = await page.query_selector(sel)
            if elem and await elem.is_visible():
                return True
        except Exception:
            continue
    return False


async def is_duels_game_over(page: Page) -> bool:
    # Signal 1: URL no longer in an active duel/game.
    try:
        url = page.url or ""
        in_active = any(p in url for p in (
            "/duels/",
            "/team-duels/",
            "/game/",
            "/live-challenge/",
            "/battle-royale/",
        ))
        if url and not in_active:
            print(f"  [duels-end] URL no longer in active duel: {url}")
            return True
    except Exception:
        pass

    # Signal 2: end-of-match-specific UI elements. Avoid generic 'Continue' /
    # 'Play again' since those may also appear between rounds.
    selectors = [
        "button:has-text('Back to lobby')",
        "button:has-text('Return to lobby')",
        "button:has-text('Find a new game')",
        "button:has-text('Find new game')",
        "button:has-text('Leave game')",
        "div:has-text('Victory')",
        "div:has-text('Defeat')",
        "div:has-text('You won')",
        "div:has-text('You lost')",
        "div:has-text('VICTORY')",
        "div:has-text('DEFEAT')",
        "[data-qa='duels-end']",
        "[data-qa='game-end']",
        "[data-qa='match-end']",
        "[data-qa='duel-finished']",
        "[class*='gameOver']",
        "[class*='matchEnd']",
        "[class*='duels-end']",
    ]
    for sel in selectors:
        try:
            elem = await page.query_selector(sel)
            if elem:
                visible = await elem.is_visible()
                if visible:
                    print(f"  [duels-end] detected via selector: {sel}")
                    return True
        except Exception:
            continue
    return False


async def play_round(page: Page, round_num: int) -> None:
    global _round_started_at
    _round_started_at = time.time()
    bar = "═" * 64
    print(f"\n{_C.CYAN}{bar}")
    print(f"  ROUND {round_num}  ·  mode={MODE}  ·  model={MODEL_ID}")
    print(f"{bar}{_C.RESET}")
    if not await wait_for_round(page):
        print(f"  {_C.RED}✗ timed out waiting for panorama{_C.RESET}")
        return

    png, _ = await capture_streetview(page)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shot_path = SCREENSHOTS_DIR / f"round_{ts}.png"
    shot_path.write_bytes(png)
    
    # Gerar os recortes detalhados para alimentar no prompt e aumentar a resolução visual do carro / bússola
    import io
    pil_img = Image.open(io.BytesIO(png)).convert("RGB")
    w_img, h_img = pil_img.size
    
    # 1. Car Meta (disabled for NMPZ)
    car_bytes = None
    
    # 2. Compass Meta
    # Bússola está no canto SUPERIOR no MEIO do ecrã
    compass_box = (int(w_img * 0.35), 0, int(w_img * 0.65), int(h_img * 0.20))
    compass_cropped = pil_img.crop(compass_box)
    compass_io = io.BytesIO()
    compass_cropped.save(compass_io, format="JPEG", quality=85)
    compass_bytes = compass_io.getvalue()

    # Round-robin model selection across MODEL_IDS using total rounds so far,
    # so A/B tests stay balanced across restarts.
    total_so_far = count_rounds_in_log()
    model_used = MODEL_IDS[total_so_far % len(MODEL_IDS)]
    if len(MODEL_IDS) > 1:
        print(f"  model   : {model_used}")

    extra_msgs = []

    # 0. Session blacklist: warn about countries the bot recently guessed wrong,
    # so it doesn't fall into the same trap twice in a row. Soft penalty — not a
    # ban, just awareness.
    recent_wrong = recent_wrong_countries(n=8)
    if recent_wrong:
        extra_msgs.append({
            "role": "user",
            "content": (
                f"SESSION HISTORY: in the last few rounds you incorrectly guessed: "
                f"{', '.join(recent_wrong[:5])}. Do NOT default to any of these again "
                f"unless you have SPECIFIC visual evidence (unique landmark, readable "
                f"sign text, plate code) that confirms it. Prefer the RAG top-1 if you "
                f"would otherwise pick one of these by default."
            ),
        })
        print(f"  [blacklist] recent wrong: {', '.join(recent_wrong[:5])}")

    # Removido: No modo NMPZ puro/mapa atual, o carro não é visível e só gasta tokens.
    # (Código do car-meta estava aqui)

    # 1.5 + 1.6 + 2. OCR, compass and RAG run in parallel — no dependencies between them.
    # In duels, mask the HP/avatar/compass/minimap UI overlays so OCR doesn't read
    # player country flags or world-map country labels as if they were Street View text.
    ocr_input_png = mask_duels_ui(png) if IS_MULTIPLAYER else png
    start_parallel = time.time()
    _ocr_future = asyncio.to_thread(extract_text_from_image, ocr_input_png) if _ocr_reader is not None else None
    _compass_future = asyncio.to_thread(crop_compass_meta, png)
    _rag_future = asyncio.to_thread(build_rag_examples, shot_path)

    _parallel_results = await asyncio.gather(
        _ocr_future if _ocr_future is not None else asyncio.sleep(0, result=""),
        _compass_future,
        _rag_future,
        return_exceptions=True,
    )
    print(f"  [parallel] OCR+compass+RAG concluídos em {time.time()-start_parallel:.1f}s")

    ocr_text = _parallel_results[0] if not isinstance(_parallel_results[0], BaseException) else ""

    # Defensive filter: if OCR captured between-rounds breakdown text, opponent
    # notifications, or other UI noise instead of actual Street View signage,
    # discard it entirely.
    if ocr_text and IS_MULTIPLAYER:
        _ocr_lc = ocr_text.lower()
        _ui_noise_signals = (
            "round 1", "round 2", "round 3", "round 4", "round 5",
            "round 6", "round 7", "round 8", "watch replay",
            "game breakdown", "your score", "your health",
            "has guessed", "hos gvessed", "hasguessed",  # opponent guess notifications
            "team blue", "team red",
        )
        if any(sig in _ocr_lc for sig in _ui_noise_signals):
            print(f"  [ocr] discarded — UI noise detected ('{[s for s in _ui_noise_signals if s in _ocr_lc][0]}')")
            ocr_text = ""
    _compass_result = _parallel_results[1] if not isinstance(_parallel_results[1], BaseException) else None
    _rag_result = _parallel_results[2] if not isinstance(_parallel_results[2], BaseException) else ("", [], None, None, {})

    ocr_script: str | None = None
    ocr_literal_country: str | None = None
    if ocr_text:
        print(f"  [ocr] texto detetado: {ocr_text[:80]}...")
        extra_msgs.append({
            "role": "user",
            "content": f"OCR System detected the following text in the environment: '{ocr_text}'. Use this to identify the language, alphabet (e.g., Cyrillic vs Latin), or specific city/street names."
        })

        ocr_script = detect_script(ocr_text)
        if ocr_script:
            allowed = _SCRIPT_COUNTRIES.get(ocr_script, ())
            if allowed:
                print(f"  [ocr-script] {ocr_script} → restringido a: {', '.join(allowed)}")
                extra_msgs.append({
                    "role": "user",
                    "content": (
                        f"HARD CONSTRAINT: the OCR text contains {ocr_script.upper()} script. "
                        f"The answer MUST be one of: {', '.join(allowed)}. "
                        f"No other country can produce this script on public signage."
                    )
                })

        ocr_literal_country = detect_country_from_ocr(ocr_text)
        if ocr_literal_country:
            print(f"  [ocr-literal] país identificado pelo texto OCR: {ocr_literal_country}")
            extra_msgs.append({
                "role": "user",
                "content": (
                    f"HARD CONSTRAINT: the OCR text contains a verbatim reference to "
                    f"{ocr_literal_country} (country name or unique local token). "
                    f"The answer MUST be {ocr_literal_country}."
                )
            })

    # 1.6. Compass result (already computed in parallel)
    try:
        compass_direction = _compass_result
        if compass_direction and not isinstance(compass_direction, str):
            compass_direction = None
        if compass_direction:
            cd = compass_direction.lower()
            rel_map = {
                "north":       ("ahead=N", "right=E", "behind=S", "left=W"),
                "north-east":  ("ahead=NE", "right=SE", "behind=SW", "left=NW"),
                "east":        ("ahead=E", "right=S", "behind=W", "left=N"),
                "south-east":  ("ahead=SE", "right=SW", "behind=NW", "left=NE"),
                "south":       ("ahead=S", "right=W", "behind=N", "left=E"),
                "south-west":  ("ahead=SW", "right=NW", "behind=NE", "left=SE"),
                "west":        ("ahead=W", "right=N", "behind=E", "left=S"),
                "north-west":  ("ahead=NW", "right=NE", "behind=SE", "left=SW"),
            }
            rel = rel_map.get(cd, ())
            rel_line = " ".join(rel)
            extra_msgs.append({
                "role": "user",
                "content": (
                    f"COMPASS + SUN ANALYSIS (opencv-detected).\n"
                    f"Camera is facing {compass_direction}. Screen-direction → world-bearing: "
                    f"{rel_line}.\n"
                    f"HEMISPHERE INFERENCE (use this):\n"
                    f"1. Look at shadows of vertical objects (poles, people, trees, signs).\n"
                    f"2. Translate the shadow direction on screen into a world bearing using "
                    f"the mapping above (e.g., if shadow points LEFT on screen and camera faces "
                    f"North, shadow points West → sun is in the East → it is morning).\n"
                    f"3. If shadows point NORTH (±45°) → sun in the south → NORTHERN hemisphere.\n"
                    f"4. If shadows point SOUTH (±45°) → sun in the north → SOUTHERN hemisphere.\n"
                    f"5. Very short/near-vertical shadows → near the tropics (sun overhead).\n"
                    f"State your inferred hemisphere in reasoning and let it narrow down the "
                    f"candidate country list."
                ),
            })
            print(f"  [compass meta] OpenCV detectou: a câmara está virada para {compass_direction}")
    except Exception as e:
        print(f"  [compass meta error] {e}")

    # 2. RAG visual memory (already computed in parallel)
    examples, similar_countries, rag_top_country, rag_top_dist, rag_scores = _rag_result
    if examples:
        print(f"  [rag] visually similar to: {', '.join(similar_countries)}")
        extra_msgs.append({"role": "user", "content": examples})

    # 3. Soil/biome analysis — fires globally when HSV detects a distinctive biome.
    try:
        soil_hint = analyze_ground_color(pil_img)
        if soil_hint and "DENSE GREEN TROPICAL" in soil_hint and similar_countries:
            # Boreal taiga and temperate green fields can match the green-HSV
            # thresholds but are NOT tropical. Suppress the tropical hint when
            # RAG strongly suggests non-tropical regions.
            non_tropical = {
                "russian federation", "russia", "finland", "sweden", "norway",
                "iceland", "denmark", "estonia", "latvia", "lithuania", "belarus",
                "canada", "united kingdom", "ireland", "netherlands", "germany",
                "poland", "czechia", "slovakia", "austria", "switzerland",
                "france", "ukraine", "romania", "hungary",
            }
            rag_lc = {c.lower() for c in similar_countries[:3]}
            overlap = rag_lc & non_tropical
            if overlap:
                print(f"  [soil] suppressing tropical hint — RAG suggests non-tropical: {overlap}")
                soil_hint = None
        if soil_hint:
            extra_msgs.append({"role": "user", "content": soil_hint})
            print(f"  [soil] {soil_hint[:80]}...")
    except Exception as e:
        print(f"  [soil error] {e}")

    if examples:

        # 4. Dynamic Country Metas — top-3 ranked countries so the right answer
        # is covered even when it falls to rank #3 in the RAG.
        top_countries = similar_countries[:3]
        if top_countries:
            metas_text = get_country_metas(top_countries)
            if metas_text:
                print(f"  [meta] loaded cheat sheets from library for suspected countries (length: {len(metas_text)} chars)")
                # Print a small snippet to explicitly show what was loaded
                preview = metas_text.replace('\n', ' ')[:150] + "..." if len(metas_text) > 150 else metas_text.replace('\n', ' ')
                print(f"  [meta preview] {preview}")
                
                # Selective disambiguation: always include the universal clues header,
                # then only add region-specific pairs relevant to the RAG top countries.
                _top_conts = {continent_of(c) for c in top_countries if continent_of(c)}
                _top_norm  = {_COUNTRY_ALIASES.get(c.lower(), c.lower()) for c in top_countries}

                _disambig_parts = [
                    "IMPORTANT — DO NOT FALL FOR FAKE-UNIQUE CLUES. Many features that cheat "
                    "sheets claim are 'US-specific' or 'X-specific' actually exist in several "
                    "countries:\n"
                    "• Yellow center road lines: USA, Canada, Mexico, Japan, Norway, Finland, "
                    "and also parts of Russia, Albania, South Korea — NOT uniquely US.\n"
                    "• Left-hand driving: UK, Ireland, Japan, Australia, NZ, India, South Africa, "
                    "Thailand, Indonesia, Malaysia, Hong Kong, Macau, Kenya, Bhutan — cross-check.\n"
                    "• Latin alphabet on signs: most of the world — NOT evidence of Europe.\n"
                    "• Cyrillic: Russia, Ukraine, Belarus, Bulgaria, Serbia, Kazakhstan, Mongolia, "
                    "North Macedonia — NOT uniquely Russia.\n"
                ]

                if "North America" in _top_conts or any(c in _top_norm for c in ("united states", "canada", "mexico")):
                    _disambig_parts.append(
                        "• US vs Canada: both yellow center lines + English. Canada = km/metric, "
                        "occasional ARRÊT bilingual stop, maple-leaf flags, less billboard density.\n"
                    )

                if "Europe" in _top_conts:
                    _disambig_parts.append(
                        "• Germany vs Austria: Germany = YELLOW town entry signs, 'Einbahnstraße', "
                        "2-bolt bollards, wind turbines. Austria = WHITE town signs, 'EINBAHN'.\n"
                        "• France vs neighbors: pointed white bollards + full reflector band + small "
                        "yellow D-number signs = France. Absent → consider Spain/Portugal/Italy.\n"
                        "• Romania vs Eastern Europe: holey poles all the way down + yellow pole "
                        "stickers = Romania. Poland has thick HORIZONTAL RED BAND bollards.\n"
                        "• Sweden vs Finland: short white edge dashes + 'väg'/'gata' = Sweden. "
                        "Finland = double vowels (aamu, katu). Norway = YELLOW center lines.\n"
                        "• Türkiye vs Romania: Ğ/İ/Ş chars + TR plate red strip = Türkiye. "
                        "Minaret = Türkiye.\n"
                    )

                if "Africa" in _top_conts or "south africa" in _top_norm:
                    _disambig_parts.append(
                        "• South Africa vs Europe: LEFT-HAND traffic + YELLOW outer road lines + "
                        "bird poles. No European country has all three.\n"
                    )

                if "South America" in _top_conts:
                    _disambig_parts.append(
                        "• South America — DO NOT default to Brazil. Distinguishing clues:\n"
                        "  - Brazil: red soil, white+blue-stripe Mercosul plates, rectangular "
                        "concrete poles, double yellow center lines, Portuguese (-ção/-nh).\n"
                        "  - Argentina: black/white+blue plates, white-and-red chevrons, Pampas "
                        "flat grassland, wide tree-lined roads.\n"
                        "  - Chile: narrow country, Atacama desert north, Andes east, EU-style "
                        "white+blue plates.\n"
                        "  - Peru: Andes or Pacific coastal desert, adobe construction, yellow "
                        "plates, poor roads.\n"
                        "  - Colombia: yellow plates, black-white cross on sign backs.\n"
                        "  - Ecuador: Andes always nearby, yellow plates.\n"
                        "  - Uruguay: flat grassland, Mercosul plates, red/white chevrons.\n"
                    )

                _disambig_parts.append(
                    "Always prefer the #1 RAG match, but override when you see a country-specific "
                    "clue from the cheat sheet that definitively identifies a different country.\n"
                )
                disambiguation = "".join(_disambig_parts)
                extra_msgs.append({
                    "role": "user",
                    "content": (
                        f"{disambiguation}\n"
                        f"Here are expert GeoGuessr cheat sheets for the TOP visually-similar "
                        f"countries. Cross-reference these with the visual clues (poles, lines, "
                        f"cars) to find the exact match:\n\n{metas_text}"
                    )
                })

    # 3.5. India regional reasoning: when India appears in RAG top-3, force the model
    # to reason about which region of India it's seeing BEFORE committing to coordinates.
    # This prevents the "always Bangalore" coordinate anchor from the RAG.
    if similar_countries and any(c.lower() == "india" for c in similar_countries[:3]):
        extra_msgs.append({
            "role": "user",
            "content": (
                "INDIA REGIONAL REASONING REQUIRED: If you conclude this is India, you MUST "
                "first identify the region before picking coordinates:\n"
                "- NORTH INDIA (lat 28-32°N): Delhi NCR (28.6°N,77.2°E), Punjab, Haryana, UP — "
                "flat plains, wheat fields, wide highways, Hindi/Punjabi signage.\n"
                "- SOUTH INDIA (lat 10-15°N): Karnataka/Bangalore (12.9°N,77.6°E), Tamil Nadu, "
                "Kerala — lush green, palm trees, Kannada/Tamil/Malayalam script.\n"
                "- WEST INDIA (lat 18-23°N): Mumbai (19°N,73°E), Gujarat, Rajasthan — "
                "arid/semi-arid, Marathi/Gujarati/Hindi signs, flat desert (Rajasthan).\n"
                "- EAST INDIA (lat 20-25°N): Kolkata (22.5°N,88.3°E), West Bengal, Odisha — "
                "humid, Bengali script, rice paddies, red soil.\n"
                "- NORTHEAST INDIA (lat 24-28°N, lon 88-97°E): Assam, Meghalaya — "
                "very green, hilly, Assamese/Bengali script.\n"
                "- CENTRAL INDIA (lat 20-24°N, lon 75-82°E): MP, Chhattisgarh — "
                "dry deciduous forest, rocky terrain.\n"
                "Do NOT default to Bangalore (12.97,77.59) unless visual clues confirm South India."
            )
        })
        print("  [india] injecting regional reasoning constraint")

    # 3.6. Russia regional reasoning — DISABLED for A/B test.
    # Same hypothesis as Brazil: fires when RAG suggests Russia, but RAG
    # over-suggests Russia for any boreal taiga / coniferous landscape,
    # which biases the model toward Russia even when actual is Finland,
    # Sweden, Canada, or central Europe (verified from log analysis).
    # Re-enable if disabling regresses Russia-internal accuracy.

    # 3.7. Brazil regional reasoning — DISABLED for A/B test.
    # Hypothesis: this hint fires when RAG suggests Brazil, but RAG over-suggests
    # Brazil and reinforces the bias that causes Brazil→Cambodia/Guatemala disasters.
    # Re-enable if 100-round comparison shows score regression.

    # 3.8. South Africa regional reasoning: mistakenly placed on wrong continents 4/6 times.
    if similar_countries and any(c.lower() == "south africa" for c in similar_countries[:3]):
        extra_msgs.append({
            "role": "user",
            "content": (
                "SOUTH AFRICA IDENTIFICATION: Key confirms — LEFT-HAND traffic, YELLOW outer road "
                "lines (unique in Africa), 'bird poles' (concrete poles with 1-5 horizontal bars + "
                "white insulators), green N-road/R-road signs. "
                "If you see yellow outer road lines + left-hand traffic = South Africa, NOT Spain/Europe/Australia. "
                "Australia has white outer lines. No European country drives on the left with yellow outer lines.\n"
                "REGIONAL COORDS: Cape Town (-33.9,18.4), Garden Route (-33.5,22.0), "
                "Karoo dry plateau (-32,24), Joburg/Gauteng (-26.2,28.0), Durban/KZN (-29.9,31.0), "
                "Limpopo bush (-23.9,29.5)."
            )
        })
        print("  [south africa] injecting identification constraint")

    # 4. Continent pre-filter: if RAG top-3 MAJORITY agree on the same continent,
    # inject a hard region constraint BEFORE the main call so the model never
    # picks a country on the wrong continent in the first place.
    # Changed from "all agree" to "majority (>=2/3)" to catch more wrong-continent errors.
    if similar_countries and len(similar_countries) >= 2:
        rag_conts = [continent_of(c) for c in similar_countries[:3]]
        rag_conts = [c for c in rag_conts if c]
        if rag_conts:
            from collections import Counter as _CCounter
            _most_common_cont, _most_common_count = _CCounter(rag_conts).most_common(1)[0]
            # Majority = at least 2 out of up to 3, or the only one present
            if _most_common_count >= 2 or len(rag_conts) == 1:
                pre_constraint = (
                    f"STRONG HINT (not absolute): {_most_common_count}/{len(rag_conts)} visually "
                    f"similar reference images are from {_most_common_cont} "
                    f"({', '.join(similar_countries[:3])}). Lean towards {_most_common_cont} "
                    f"UNLESS you see specific text, plate format, alphabet, or unique landmark "
                    f"evidence that points elsewhere. RAG is visual-only and can be wrong on "
                    f"biome-similar regions across continents."
                )
                extra_msgs.insert(0, {"role": "user", "content": pre_constraint})
                print(f"  [region] pre-filter: {_most_common_cont} {_most_common_count}/{len(rag_conts)} ({', '.join(similar_countries[:3])})")

    # 5. Dynamic confusion pairs: inject data-driven near-miss warnings for
    # countries currently suspected by the RAG. Only fires when we have enough
    # historical evidence (≥3 confusions) to avoid noise.
    if _CONFUSION_PAIRS and similar_countries:
        suspected = {c.lower() for c in similar_countries[:3]}
        relevant = [
            (g, a, c) for g, a, c in _CONFUSION_PAIRS
            if g.lower() in suspected or a.lower() in suspected
        ]
        if relevant:
            lines = "\n".join(
                f"  - '{g}' was guessed {c}x but was actually '{a}'"
                for g, a, c in relevant[:5]
            )
            extra_msgs.append({
                "role": "user",
                "content": (
                    f"HISTORICAL CONFUSION PAIRS (your own past errors for suspected countries):\n"
                    f"{lines}\n"
                    f"If you are about to guess one of these 'wrong' countries, pause and "
                    f"double-check the specific clues that distinguish it from the actual one."
                ),
            })
            print(f"  [confusion] {len(relevant)} relevant pair(s): "
                  + ", ".join(f"{g}→{a}" for g, a, _ in relevant[:3]))

    extra = extra_msgs if extra_msgs else None

    guess = None
    for attempt in range(3):
        try:
            # On the last retry, drop extras to minimize chance of context-related failure
            use_extra = extra if attempt < 2 else None
            guess = ask_model(png, extra_user_msgs=use_extra, model_id=model_used, car_bytes=car_bytes, compass_bytes=compass_bytes)
            break
        except Exception as e:
            print(f"  model error (attempt {attempt+1}): {e}")
    if guess is None:
        # Fallback: use the top RAG country centroid if available, else a neutral
        # point (Paris) — anything is better than (0,0) which always loses the round.
        fb_country = rag_top_country or (similar_countries[0] if similar_countries else None)
        fb_lat, fb_lon = 48.85, 2.35  # Paris
        if fb_country:
            c_n = _COUNTRY_ALIASES.get(fb_country.lower(), fb_country.lower())
            if c_n in _COUNTRY_CENTROIDS:
                fb_lat, fb_lon = _COUNTRY_CENTROIDS[c_n]
        print(f"  {_C.RED}✗ all 3 model attempts failed — fallback to {fb_country or 'Paris'} ({fb_lat:.2f}, {fb_lon:.2f}){_C.RESET}")
        guess = Guess(country=fb_country or "?", region="?", latitude=fb_lat, longitude=fb_lon, confidence=0.0, reasoning="fallback")
    _conf_col = _C.GREEN if guess.confidence >= 0.80 else (_C.YELLOW if guess.confidence >= 0.55 else _C.RED)
    print(f"  {_C.GREY}{_ts()}{_C.RESET} {_C.BOLD}GUESS    {_C.RESET} {_C.CYAN}{guess.country}{_C.RESET} / {guess.region or '-'}  ({guess.latitude:.2f}, {guess.longitude:.2f})  {_conf_col}conf {guess.confidence:.2f}{_C.RESET}")
    if guess.reasoning:
        print(f"  {_C.GREY}{_ts()}{_C.RESET} {_C.DIM}think    {_C.RESET} {_C.DIM}{guess.reasoning}{_C.RESET}")
    if guess.candidates:
        cand_str = ", ".join(f"{c}({p:.2f})" for c, p in guess.candidates[:3])
        print(f"  {_C.GREY}{_ts()}{_C.RESET} {_C.DIM}cands    {_C.RESET} {_C.DIM}{cand_str}{_C.RESET}")

    # Candidate × RAG cross-check: if the primary guess does not match the top
    # RAG weighted country, but one of the model's OWN top-3 candidates DOES,
    # that's strong evidence to switch — the model already considered it plausible.
    if (
        guess.candidates
        and rag_scores
        and guess.country
    ):
        rag_ranked = sorted(rag_scores.items(), key=lambda kv: kv[1], reverse=True)
        rag_top = rag_ranked[0][0] if rag_ranked else None
        rag_top_score = rag_ranked[0][1] if rag_ranked else 0.0
        rag_second = rag_ranked[1][1] if len(rag_ranked) > 1 else 0.0
        primary_n = _COUNTRY_ALIASES.get(guess.country.lower(), guess.country.lower())
        rag_top_n = _COUNTRY_ALIASES.get((rag_top or "").lower(), (rag_top or "").lower())
        cand_map = {
            _COUNTRY_ALIASES.get(c.lower(), c.lower()): (c, conf)
            for c, conf in guess.candidates
        }
        # Trigger only if:
        # - Primary ≠ RAG top
        # - RAG top appears among model's candidates (so it's not an alien suggestion)
        # - RAG top-score is meaningfully higher than the next (>=1.5× the runner-up)
        if (
            rag_top_n
            and primary_n != rag_top_n
            and rag_top_n in cand_map
            and rag_top_score >= 1.5 * max(rag_second, 0.01)
        ):
            alt_country, alt_conf = cand_map[rag_top_n]
            primary_conf = next(
                (c for cn, c in guess.candidates if _COUNTRY_ALIASES.get(cn.lower(), cn.lower()) == primary_n),
                guess.confidence,
            )
            # Only switch if model is clearly unsure AND RAG alternative is plausible
            if primary_conf < 0.60 and alt_conf >= 0.30:
                print(f"  ⚠ cand×rag: you picked {guess.country} (conf {primary_conf:.2f}) but "
                      f"RAG-weighted top is {alt_country} (score {rag_top_score:.2f}) — and it's "
                      f"in YOUR candidates (conf {alt_conf:.2f})")
                cand_msg = (
                    f"CROSS-CHECK: your own candidates list includes {alt_country} (conf "
                    f"{alt_conf:.2f}) and the weighted RAG visual match points to {alt_country} "
                    f"with score {rag_top_score:.2f} (vs next {rag_second:.2f}). This convergent "
                    f"evidence is stronger than your current primary {guess.country}. Rewrite the "
                    f"JSON with country='{alt_country}' and coordinates inside it — unless you "
                    f"have concrete visual evidence (OCR text, plate code, named landmark) that "
                    f"rules out {alt_country}. Return only the JSON."
                )
                try:
                    guess2 = ask_model(png, car_bytes=car_bytes, compass_bytes=compass_bytes, extra_user_msgs=[{"role": "user", "content": cand_msg}], model_id=model_used)
                    if guess2 and not (abs(guess2.latitude) < 0.5 and abs(guess2.longitude) < 0.5):
                        print(f"  cand×rag retry: {guess2.country}/{guess2.region or '-'} ({guess2.latitude:.2f}, {guess2.longitude:.2f}) conf {guess2.confidence}")
                        guess = guess2
                except Exception as e:
                    print(f"  cand×rag failed: {e}")

    # Post-guess hard override: OCR text contains a verbatim country name and the
    # guess disagrees. This is stronger than any visual similarity signal.
    if ocr_literal_country and guess.country:
        want_n = _COUNTRY_ALIASES.get(ocr_literal_country.lower(), ocr_literal_country.lower())
        got_n  = _COUNTRY_ALIASES.get(guess.country.lower(), guess.country.lower())
        if want_n != got_n:
            print(f"  ⚠ ocr-override: OCR says {ocr_literal_country} but you picked {guess.country}")
            ocr_msg = (
                f"CRITICAL: the OCR has a verbatim mention of {ocr_literal_country}. "
                f"Your previous guess '{guess.country}' is wrong. Rewrite the JSON with "
                f"country='{ocr_literal_country}' and place coordinates inside it. "
                f"Return only the JSON."
            )
            try:
                guess2 = ask_model(png, car_bytes=car_bytes, compass_bytes=compass_bytes, extra_user_msgs=[{"role": "user", "content": ocr_msg}], model_id=model_used)
                if guess2 and not (abs(guess2.latitude) < 0.5 and abs(guess2.longitude) < 0.5):
                    print(f"  ocr-override: {guess2.country}/{guess2.region or '-'} ({guess2.latitude:.2f}, {guess2.longitude:.2f}) conf {guess2.confidence}")
                    if guess2.reasoning:
                        print(f"  think   : {guess2.reasoning}")
                    guess = guess2
            except Exception as e:
                print(f"  ocr-override failed: {e}")

    # Continent self-consistency check: model declared a continent in the JSON.
    # If that continent contradicts the actual country it picked, catch it immediately.
    if guess.continent and guess.country:
        declared = guess.continent.strip().lower()
        actual_cont = (continent_of(guess.country) or "").lower()
        # Normalize common aliases
        _cont_aliases = {"north america": "north america", "south america": "south america",
                         "central america": "north america", "americas": "north america"}
        declared = _cont_aliases.get(declared, declared)
        if actual_cont and declared and actual_cont != declared:
            print(f"  [continent-mismatch] model said '{guess.continent}' but {guess.country} is in {continent_of(guess.country)}")
            mismatch_msg = (
                f"CONTINENT CONTRADICTION: You declared continent='{guess.continent}' but "
                f"'{guess.country}' is actually in {continent_of(guess.country)}. "
                f"Either change the country to one actually in {guess.continent}, "
                f"or change the continent to match {guess.country}. "
                f"Return only the corrected JSON."
            )
            try:
                guess2 = ask_model(png, car_bytes=car_bytes, compass_bytes=compass_bytes,
                                   extra_user_msgs=[{"role": "user", "content": mismatch_msg}],
                                   model_id=model_used)
                if guess2 and not (abs(guess2.latitude) < 0.5 and abs(guess2.longitude) < 0.5):
                    print(f"  continent fix: {guess2.country}/{guess2.region or '-'} ({guess2.latitude:.2f},{guess2.longitude:.2f})")
                    guess = guess2
            except Exception as e:
                print(f"  continent-mismatch retry failed: {e}")

    # Continent sanity check: if the RAG top-3 countries are all on the same
    # continent and the guess is on a different one, the model likely ignored the
    # visual signal in favour of a weak textual hint. Force one retry.
    if similar_countries and len(similar_countries) >= 2 and guess.country:
        rag_conts = [continent_of(c) for c in similar_countries[:3]]
        rag_conts = [c for c in rag_conts if c]
        guess_cont = continent_of(guess.country)
        if rag_conts and guess_cont and all(c == rag_conts[0] for c in rag_conts) and guess_cont != rag_conts[0]:
            print(f"  ⚠ continent: RAG→{rag_conts[0]} but you picked {guess.country} ({guess_cont})")
            cont_msg = (
                f"WARNING: all {len(rag_conts)} visually similar reference matches are in "
                f"{rag_conts[0]} ({', '.join(similar_countries[:3])}), but you placed the pin in "
                f"{guess.country} ({guess_cont}). Unless you have CONCRETE evidence (OCR text, "
                f"alphabet, landmark, plate format) that rules out every {rag_conts[0]} country, "
                f"reconsider and place the pin in one of: {', '.join(similar_countries[:3])}. "
                f"Return only the JSON."
            )
            try:
                guess2 = ask_model(png, car_bytes=car_bytes, compass_bytes=compass_bytes, extra_user_msgs=[{"role": "user", "content": cont_msg}], model_id=model_used)
                if guess2 and not (abs(guess2.latitude) < 0.5 and abs(guess2.longitude) < 0.5):
                    print(f"  continent retry: {guess2.country}/{guess2.region or '-'} ({guess2.latitude:.2f}, {guess2.longitude:.2f}) conf {guess2.confidence}")
                    if guess2.reasoning:
                        print(f"  think   : {guess2.reasoning}")
                    guess = guess2
            except Exception as e:
                print(f"  continent retry failed: {e}")

    # Validate: do the declared country AND region match the reverse-geocoded coords?
    try:
        def _norm_place(s: str) -> str:
            if not s:
                return ""
            s = s.lower().strip()
            s = s.replace("ß", "ss")
            for a, b in (("ü", "ue"), ("ö", "oe"), ("ä", "ae"),
                         ("å", "aa"), ("ø", "oe"), ("æ", "ae")):
                s = s.replace(a, b)
            s = unicodedata.normalize("NFD", s)
            s = "".join(c for c in s if unicodedata.category(c) != "Mn")
            s = "".join(c if c.isalnum() else " " for c in s)
            return " ".join(s.split())

        def _validate(g: Guess) -> tuple[bool, bool, bool, str, str]:
            cc, a1, c_iso = lookup_country(g.latitude, g.longitude)
            dc = (g.country or "").strip().lower()
            dr = (g.region or "").strip().lower()
            cc_lc = cc.lower()
            a1_lc = a1.lower()
            dc_n = _COUNTRY_ALIASES.get(dc, dc)
            cc_n = _COUNTRY_ALIASES.get(cc_lc, cc_lc)
            dc_norm = _norm_place(dc_n)
            cc_norm = _norm_place(cc_n)
            c_ok = (
                not dc or not cc or cc_n == dc_n
                or dc_n in cc_n or cc_n in dc_n
                or dc == c_iso.lower()
                or (dc_norm and cc_norm and (dc_norm == cc_norm
                    or dc_norm in cc_norm or cc_norm in dc_norm))
            )
            dr_norm = _norm_place(dr)
            a1_norm = _norm_place(a1_lc)
            r_ok = (
                not dr or not a1 or a1_lc == dr
                or dr in a1_lc or a1_lc in dr
                or (dr_norm and a1_norm and (dr_norm == a1_norm
                    or dr_norm in a1_norm or a1_norm in dr_norm))
            )
            on_land = distance_to_nearest_land_km(g.latitude, g.longitude) <= 80.0
            return c_ok, r_ok, on_land, cc, a1

        for attempt in range(2):
            c_ok, r_ok, on_land, coord_country, coord_admin1 = _validate(guess)
            if c_ok and r_ok and on_land:
                break
            if not on_land:
                print(f"  mismatch: coords ({guess.latitude:.2f}, {guess.longitude:.2f}) are in open water")
                ocean_hint = region_coord_hint(guess.country, guess.region)
                ocean_bbox = country_bbox_hint(guess.country)
                if ocean_hint:
                    print(f"  [hint] {ocean_hint}")
                if ocean_bbox:
                    print(f"  [bbox] {ocean_bbox}")
                ocean_hint_line = f" {ocean_hint} Use coordinates close to that reference." if ocean_hint else ""
                ocean_bbox_line = f" {ocean_bbox}" if ocean_bbox else ""
                msg = (
                    f"Your previous JSON coordinates ({guess.latitude:.3f}, {guess.longitude:.3f}) are in the OCEAN/open water, not on land. "
                    f"Move the coordinates INLAND inside {guess.country or 'the country you identified'}."
                    f"{ocean_hint_line}{ocean_bbox_line} "
                    f"Street View is only on land — the pin must be on a road/city, never in the sea. "
                    f"Return only the JSON."
                )
                try:
                    guess2 = ask_model(png, car_bytes=car_bytes, compass_bytes=compass_bytes, extra_user_msgs=[{"role": "user", "content": msg}], model_id=model_used)
                    if abs(guess2.latitude) < 0.5 and abs(guess2.longitude) < 0.5:
                        print(f"  retry   : rejected (lat/lon ≈ 0,0) — keeping previous")
                        break
                    print(f"  retry   : {guess2.country}/{guess2.region or '-'} ({guess2.latitude:.2f}, {guess2.longitude:.2f}) conf {guess2.confidence}")
                    if guess2.reasoning:
                        print(f"  think   : {guess2.reasoning}")
                    guess = guess2
                    continue
                except Exception as e:
                    print(f"  retry failed: {e} — keeping previous")
                    break
            hint = region_coord_hint(guess.country, guess.region)
            bbox = country_bbox_hint(guess.country)
            hint_line = f" {hint} Use coordinates close to that reference." if hint else ""
            bbox_line = f" {bbox}" if bbox else ""
            if not c_ok:
                print(f"  mismatch: said {guess.country} but coords → {coord_country}")
                if hint:
                    print(f"  [hint] {hint}")
                if bbox:
                    print(f"  [bbox] {bbox}")
                msg = (
                    f"Your previous JSON said country='{guess.country}' but the coordinates "
                    f"({guess.latitude:.3f}, {guess.longitude:.3f}) are in {coord_country}. "
                    f"Decide which is correct: if the image is in {guess.country}, fix the coordinates to be inside {guess.country}"
                    f"{hint_line}{bbox_line} "
                    f"If the image is actually in {coord_country}, change the country field instead. "
                    f"IMPORTANT: Western longitudes are NEGATIVE (Ireland≈-6°, Portugal≈-8°, UK≈-2°, US East≈-70°, Brazil≈-50°); "
                    f"Eastern longitudes are POSITIVE (Poland≈+18°, India≈+77°, Japan≈+138°). "
                    f"Do NOT output 0.0 for latitude or longitude unless the location is literally on the equator or prime meridian. Return only the JSON."
                )
            else:
                print(f"  region note: said {guess.region} but coords → {coord_admin1} (accepted, country OK)")
                break
            try:
                guess2 = ask_model(png, car_bytes=car_bytes, compass_bytes=compass_bytes, extra_user_msgs=[{"role": "user", "content": msg}], model_id=model_used)
                if abs(guess2.latitude) < 0.5 and abs(guess2.longitude) < 0.5:
                    print(f"  retry   : rejected (lat/lon ≈ 0,0) — keeping previous")
                    break
                print(f"  retry   : {guess2.country}/{guess2.region or '-'} ({guess2.latitude:.2f}, {guess2.longitude:.2f}) conf {guess2.confidence}")
                if guess2.reasoning:
                    print(f"  think   : {guess2.reasoning}")
                guess = guess2
            except Exception as e:
                print(f"  retry failed: {e} — keeping previous")
                break

        c_ok_final, r_ok_final, on_land_final, coord_country_final, coord_admin1_final = _validate(guess)

        # 4. Pre-click Polygon Clamping (hard guard).
        # Always run if we have the country polygon: if the guess coords aren't strictly
        # inside the declared country, snap to the nearest point on the polygon and pull
        # inland. This is independent of reverse_geocoder (which can disagree near
        # borders) and catches cases where the model names Germany but coords are in
        # Kazakhstan (real example from the log).
        if guess.country and WORLD_BORDERS:
            target_country = guess.country.lower()
            target_country = _COUNTRY_ALIASES.get(target_country, target_country)
            if target_country in _COUNTRY_POLYGONS:
                target_geom = _COUNTRY_POLYGONS[target_country]
                point_guess = Point(guess.longitude, guess.latitude)
                if not target_geom.contains(point_guess):
                    try:
                        pol_pt, _ = nearest_points(target_geom, point_guess)
                        cent = target_geom.representative_point()
                        # Pull 20% toward centroid — large enough that projection
                        # drift (~100 km avg) won't spill into the neighbor country.
                        alpha_pull = 0.20
                        lat_safe = pol_pt.y + (cent.y - pol_pt.y) * alpha_pull
                        lon_safe = pol_pt.x + (cent.x - pol_pt.x) * alpha_pull
                        d_km = haversine_km(guess.latitude, guess.longitude, lat_safe, lon_safe)
                        print(f"  [pre-click clamp] {guess.country}: ({guess.latitude:.2f},{guess.longitude:.2f}) → ({lat_safe:.2f},{lon_safe:.2f}) Δ{d_km:.0f}km")
                        guess.latitude = lat_safe
                        guess.longitude = lon_safe
                    except Exception as e:
                        print(f"  [pre-click clamp error] {e}")
        
    except Exception as e:
        print(f"  validation skipped: {e}")

    placed = await place_guess(page, guess)
    if not placed:
        return

    await human_delay(3.0, 5.0)
    api_data = await read_round_from_api(page, round_num)
    actual = api_data["actual"] if api_data else None
    placed = api_data.get("placed") if api_data else None

    # error_km = distance from where the pin ACTUALLY landed (placed) to actual.
    # If we can't read placed, fall back to the requested guess coords.
    ref = placed or (guess.latitude, guess.longitude)
    error_km = haversine_km(ref[0], ref[1], *actual) if actual else None
    projection_drift_km = (
        haversine_km(guess.latitude, guess.longitude, placed[0], placed[1]) if placed else None
    )

    actual_entry = None
    country_hit = None
    if actual:
        country, admin1, cc = lookup_country(*actual)
        actual_entry = {
            "latitude": actual[0],
            "longitude": actual[1],
            "country": country,
            "admin1": admin1,
            "cc": cc,
        }
        # country_hit now compares the country of where the pin LANDED to actual.
        placed_country = ""
        if placed:
            placed_country, _, _ = lookup_country(*placed)
        else:
            placed_country = guess.country or ""
        placed_country_lc = placed_country.strip().lower()
        country_hit = bool(country) and bool(placed_country_lc) and (
            placed_country_lc == country.lower()
            or country.lower() in placed_country_lc
            or placed_country_lc in country.lower()
        )

    placed_entry = None
    if placed:
        pc, pa, pcc = lookup_country(*placed)
        placed_entry = {
            "latitude": placed[0],
            "longitude": placed[1],
            "country": pc,
            "admin1": pa,
            "cc": pcc,
        }

    if guess and guess.country:
        guess.country = _COUNTRY_ALIASES.get(guess.country.lower(), guess.country.title())

    append_log({
        "timestamp": ts,
        "round": round_num,
        "game_mode": f"nmpz-{MODE}",
        "model": model_used,
        "screenshot": str(shot_path.relative_to(PROJECT_DIR)),
        "guess": guess.__dict__,
        "placed": placed_entry,
        "actual": actual_entry,
        "error_km": error_km,
        "projection_drift_km": projection_drift_km,
        "country_hit": country_hit,
    })

    # Refresh confusion pairs every 25 new rounds so the bot adapts to recent mistakes.
    current_count = count_rounds_in_log()
    if current_count - _last_confusion_refresh_count >= 25:
        _load_confusion_pairs()
    if error_km is not None:
        hit_str = "✓" if country_hit else "✗"
        print(f"  actual  : {actual_entry['country']}/{actual_entry['admin1'] or '-'} ({actual[0]:.2f}, {actual[1]:.2f})")
        if placed_entry:
            print(f"  placed  : {placed_entry['country']}/{placed_entry['admin1'] or '-'} ({placed[0]:.2f}, {placed[1]:.2f})")
            if projection_drift_km is not None and projection_drift_km > 50:
                print(f"  ⚠ projection drift: {projection_drift_km:.0f} km between asked and clicked")
        print(f"  result  : {error_km:.0f} km  country {hit_str}")

        # 3. Aprendizagem Contínua Automática (Auto-Index no ChromaDB)
        if actual_entry:
            try:
                coll, coll_car = get_chroma_collection()
                if coll:
                    start_idx = time.time()
                    shot_stem = shot_path.stem
                    
                    img = Image.open(shot_path).convert("RGB")
                    img_array = np.array(img)
                    
                    w, h = img.size
                    car_box = (0, int(h * 0.70), w, h)
                    car_array = np.array(img.crop(car_box))

                    # Quality gate: only index rounds with reasonable accuracy to
                    # avoid polluting RAG with completely wrong examples. Keep if
                    # country was correct OR error < 2000 km (right region, wrong spot).
                    rag_worthy = bool(country_hit) or (error_km is not None and error_km < 2000)
                    if not rag_worthy:
                        print(f"  [rag-skip] not indexing: wrong country + {error_km:.0f}km error")
                    else:
                        meta = [{
                            "country": actual_entry.get("country", "?"),
                            "admin1": actual_entry.get("admin1", "?"),
                            "lat": float(actual_entry.get("latitude", 0)),
                            "lon": float(actual_entry.get("longitude", 0)),
                            "reasoning": guess.reasoning,
                            "is_perfect": error_km < 100 and bool(country_hit)
                        }]

                        # Adiciona a imagem global
                        coll.upsert(ids=[shot_stem], images=[img_array], metadatas=meta)
                    
                        # Adiciona o crop do carro se existir a coleção
                        if coll_car:
                            coll_car.upsert(ids=[f"{shot_stem}_car"], images=[car_array], metadatas=meta)
                        
                    print(f"  [auto-index] ronda guardada na memória para jogos futuros ({time.time()-start_idx:.1f}s)")
            except Exception as e:
                print(f"  [auto-index erro] {e}")

    # Click 'Next round' / 'View results'.
    for sel in [
        "button[data-qa='close-round-result']",
        "button:has-text('Next')",
        "button:has-text('Play again')",
    ]:
        btn = await page.query_selector(sel)
        if btn:
            await human_delay(1.0, 2.0)
            await btn.click()
            break

    # End-of-round summary block.
    _round_dur = time.time() - _round_started_at
    if guess and guess.country and guess.country != "?":
        if error_km is not None:
            _hit_icon = f"{_C.GREEN}✓{_C.RESET}" if country_hit else f"{_C.RED}✗{_C.RESET}"
            _err_col = _C.GREEN if error_km < 200 else (_C.YELLOW if error_km < 2000 else _C.RED)
            print(f"  {_C.BOLD}└─ {_hit_icon} {guess.country} → actual {actual_entry['country'] if actual_entry else '?'}  ·  {_err_col}{error_km:.0f} km{_C.RESET}  ·  {_C.DIM}{_round_dur:.1f}s total{_C.RESET}")
        else:
            print(f"  {_C.BOLD}└─ {_C.CYAN}{guess.country}{_C.RESET}/{guess.region or '-'} pinned  ·  {_C.DIM}{_round_dur:.1f}s total{_C.RESET}")
    else:
        print(f"  {_C.BOLD}└─ {_C.RED}no guess submitted{_C.RESET}  ·  {_C.DIM}{_round_dur:.1f}s total{_C.RESET}")


async def main() -> None:
    _load_confusion_pairs()
    async with Stealth().use_async(async_playwright()) as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(USER_DATA_DIR),
            channel="chrome",
            headless=False,
            viewport={"width": 1366, "height": 850},
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
            ignore_default_args=["--enable-automation"],
        )

        # Capture ALL google.maps.Map instances (Duels has multiple — small
        # minimap + the big guess map). window.__geoMap keeps the latest for
        # backwards compatibility, but window.__getGuessMap() picks the Map
        # whose div is inside the guess-map container, which is the one the
        # user actually sees and clicks on.
        await context.add_init_script(
            """(() => {
                window.__geoMaps = [];
                window.__geoMap = null;
                window.__geoPano = null;
                window.__getGuessMap = () => {
                    const container = document.querySelector(
                        "[data-qa='guess-map'], .guess-map, div[class*='guess-map']"
                    );
                    if (container) {
                        for (const m of window.__geoMaps) {
                            try {
                                const div = m.getDiv && m.getDiv();
                                if (div && container.contains(div)) return m;
                            } catch (e) { /* ignore */ }
                        }
                    }
                    // Fallback: largest map by current rendered area.
                    let best = null, bestArea = 0;
                    for (const m of window.__geoMaps) {
                        try {
                            const div = m.getDiv && m.getDiv();
                            if (!div) continue;
                            const r = div.getBoundingClientRect();
                            const a = r.width * r.height;
                            if (a > bestArea) { bestArea = a; best = m; }
                        } catch (e) { /* ignore */ }
                    }
                    return best || window.__geoMap;
                };
                const iv = setInterval(() => {
                    if (!window.google || !window.google.maps) return;
                    let mapOk = false, panoOk = false;
                    if (window.google.maps.Map) {
                        const Orig = window.google.maps.Map;
                        try {
                            window.google.maps.Map = new Proxy(Orig, {
                                construct(target, args) {
                                    const inst = Reflect.construct(target, args);
                                    window.__geoMaps.push(inst);
                                    window.__geoMap = inst;
                                    return inst;
                                }
                            });
                            mapOk = true;
                        } catch (e) { /* ignore */ }
                    }
                    if (window.google.maps.StreetViewPanorama) {
                        const OrigP = window.google.maps.StreetViewPanorama;
                        try {
                            window.google.maps.StreetViewPanorama = new Proxy(OrigP, {
                                construct(target, args) {
                                    const inst = Reflect.construct(target, args);
                                    window.__geoPano = inst;
                                    return inst;
                                }
                            });
                            panoOk = true;
                        } catch (e) { /* ignore */ }
                    }
                    if (mapOk && panoOk) clearInterval(iv);
                }, 30);
            })();"""
        )

        page = context.pages[0] if context.pages else await context.new_page()

        if IS_MULTIPLAYER:
            _label = "TEAM DUELS" if MODE == "team-duels" else "DUELS"
            print(f"{_label} MODE — open a private {_label.lower()} lobby with NMPZ, invite friends, start the match.")
            await page.goto("https://www.geoguessr.com/multiplayer", wait_until="domcontentloaded")
            print("Press Enter here once you are in the first round of the FIRST duel…")
            await asyncio.to_thread(input)

            global _last_canvas_sig
            match_count = 0
            while True:
                match_count += 1
                bar = "█" * 64
                print(f"\n{_C.MAGENTA}{bar}")
                print(f"  MATCH {match_count}")
                print(f"{bar}{_C.RESET}")

                round_num = 0
                consecutive_failures = 0
                last_logged_count = count_rounds_in_log()
                match_start_count = last_logged_count
                while True:
                    if await is_duels_game_over(page):
                        print(f"\n  {_C.YELLOW}[match {match_count}] ended after {round_num} rounds (game-over signal){_C.RESET}")
                        break
                    round_num += 1
                    try:
                        await play_round(page, round_num)
                    except Exception as e:
                        print(f"  {_C.RED}round {round_num} failed: {e}{_C.RESET}")

                    current_count = count_rounds_in_log()
                    if current_count == last_logged_count:
                        consecutive_failures += 1
                        print(f"  {_C.YELLOW}[match {match_count}] no new log entry — failure {consecutive_failures}/3{_C.RESET}")
                        if consecutive_failures >= 3:
                            print(f"\n  {_C.YELLOW}[match {match_count}] 3 consecutive rounds without progress — assuming match ended{_C.RESET}")
                            break
                    else:
                        consecutive_failures = 0
                        last_logged_count = current_count
                    await human_delay(1.5, 2.5)

                # If THIS match logged zero rounds, no new match likely starting — exit.
                rounds_logged_this_match = count_rounds_in_log() - match_start_count
                if rounds_logged_this_match == 0:
                    print(f"\n  {_C.RED}[duels] no rounds played in match {match_count} — exiting{_C.RESET}")
                    break

                # Try to click "CONTINUE" / "Back to lobby" to advance to the next match.
                clicked_continue = False
                for sel in [
                    "button:has-text('CONTINUE')",
                    "button:has-text('Continue')",
                    "button:has-text('Find a new game')",
                    "button:has-text('Find new game')",
                    "button:has-text('Play again')",
                    "button:has-text('Back to lobby')",
                ]:
                    try:
                        btn = await page.query_selector(sel)
                        if btn and await btn.is_visible():
                            await btn.click()
                            print(f"  {_C.GREEN}[match] clicked '{sel}'{_C.RESET}")
                            clicked_continue = True
                            break
                    except Exception:
                        continue
                if not clicked_continue:
                    print(f"  {_C.YELLOW}[match] no continue button found — you may need to start the next match manually{_C.RESET}")

                # Loading screen buffer.
                print(f"  {_C.DIM}[match] waiting 5s for loading screen of next match…{_C.RESET}")
                await asyncio.sleep(5.0)

                # Reset canvas signature so the new match's first round is treated as fresh.
                _last_canvas_sig = None
                print(f"  {_C.CYAN}[match] resuming — ready for next match{_C.RESET}")

            print("\nAll duels finished. Press Enter to close, or Ctrl+C.")
            try:
                await asyncio.to_thread(input)
            except (EOFError, KeyboardInterrupt):
                pass
            await context.close()
            return

        print("Opening GeoGuessr World map — pick NMPZ and press Play.")
        await page.goto("https://www.geoguessr.com/maps/world", wait_until="domcontentloaded")
        print("Press Enter here once you are in the first round…")
        await asyncio.to_thread(input)

        target_rounds = 3000
        total = count_rounds_in_log()
        print(f"\nStarting autonomous mode — target {target_rounds} total rounds (have {total}).")

        while total < target_rounds:
            for i in range(1, 6):
                if total >= target_rounds:
                    break
                try:
                    await play_round(page, i)
                    total = count_rounds_in_log()
                    print(f"  [progress] {total}/{target_rounds}")
                except Exception as e:
                    print(f"  round {i} failed: {e}")
                await human_delay(2.0, 4.0)

            if total >= target_rounds:
                break

            print("\n[game over] clicking Play Again…")
            try:
                import stats as _stats
                all_rounds = json.loads(LOG_FILE.read_text(encoding="utf-8")) if LOG_FILE.exists() else []
                leaderboard = _stats.top_games(all_rounds, 10)
                if leaderboard:
                    print("\n🏆 Top 10 games (GeoGuessr score / 25000):")
                    print(f"  {'#':<3}{'Game':<6}{'Score':<8}{'Per-round':<30}Countries")
                    for rank, tg in enumerate(leaderboard, 1):
                        per = ",".join(str(p) for p in tg["per_round"])
                        cs = ", ".join(tg["countries"])[:40]
                        print(f"  {rank:<3}#{tg['game_idx']:<5}{tg['total']:<8}{per:<30}{cs}")
            except Exception as e:
                print(f"  [leaderboard error] {e}")
            if not await click_play_again(page):
                print("  could not find Play Again — stopping")
                break
            await human_delay(3.0, 5.0)

        print(f"\nDone. {total} rounds logged in log.json")
        await asyncio.to_thread(input, "Press Enter to close…")
        await context.close()


if __name__ == "__main__":
    asyncio.run(main())
