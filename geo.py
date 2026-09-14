"""Country tables, boundary polygons and coordinate maths.

No project imports beyond config, so it can be used standalone by analysis
scripts without pulling in the browser, the model client or the OCR models.
"""
import json
import math
from pathlib import Path

import reverse_geocoder as rg

from config import say

# Sistema de Fronteiras (Clamping Pós-Decisão)
try:
    from shapely.geometry import shape, Point
    from shapely.ops import nearest_points
    from pathlib import Path
    import json
    
    # Every exit from this block must leave WORLD_BORDERS and _COUNTRY_POLYGONS
    # defined. pipeline.py imports both by name at module level, so leaving one
    # undefined turns a missing data file into an ImportError that takes down the
    # whole agent instead of just disabling the clamp.
    WORLD_BORDERS = None
    _COUNTRY_POLYGONS = {}

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
            say(f"[geo] loaded {len(_COUNTRY_POLYGONS)} country polygon keys from countries.geojson")
            
    except FileNotFoundError:
        # Not committed (14 MB, regenerable). Country clamping is skipped without it.
        say("Aviso: countries.geojson em falta — corre 'python obter_fronteiras.py' "
              "para o descarregar. O Country Clamping fica inativo até lá.")
        WORLD_BORDERS = None
    except Exception as e:
        say(f"Erro ao ler countries.geojson: {e}")
        WORLD_BORDERS = None
except ImportError:
    say("Aviso: biblioteca 'shapely' em falta. O Country Clamping não estará ativo.")
    WORLD_BORDERS = None
    _COUNTRY_POLYGONS = {}

    # Stand-ins so callers can import these unconditionally; the clamp checks
    # WORLD_BORDERS before using them.
    def shape(*_a, **_k):                     # noqa: D103
        raise RuntimeError("shapely is not installed")

    def nearest_points(*_a, **_k):            # noqa: D103
        raise RuntimeError("shapely is not installed")

    class Point:                              # noqa: D101
        def __init__(self, *_a, **_k):
            raise RuntimeError("shapely is not installed")

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


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
