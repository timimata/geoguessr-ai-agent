"""Everything that reads pixels: cropping, the OpenCV compass, biome colour
analysis and OCR."""
import base64
import math
import threading
import time
import warnings
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import os

from config import MODE, PROJECT_DIR, feature_on, say

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


# Rounds are stored as JPEG. The model is sent JPEG anyway (see encode_image),
# the retrieval embedding is computed from a resized copy, and PNG was costing
# roughly ten times the disk for pixels nothing ever reads losslessly.
SCREENSHOT_SUFFIX = ".jpg"
SCREENSHOT_QUALITY = 92


def save_screenshot(png_bytes: bytes, path) -> Path:
    """Write a captured panorama to `path`, re-encoding to JPEG when that is the
    configured suffix. Returns the path actually written."""
    path = Path(path)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        img = Image.open(BytesIO(png_bytes)).convert("RGB")
        img.save(path, format="JPEG", quality=SCREENSHOT_QUALITY, optimize=True)
    else:
        path.write_bytes(png_bytes)
    return path


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
# Writes one annotated crop per round to compass_debug/. Useful when the needle
# detection is misbehaving, wasteful the rest of the time (43 MB accumulated
# before this was switched off), so it is opt-in: COMPASS_DEBUG=1 in .env.
COMPASS_DEBUG = os.getenv("COMPASS_DEBUG", "").strip().lower() in ("1", "true", "yes")


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


_ocr_reader = None
_ocr_reader_latin = None
_ocr_reader_cjk = None
# easyocr Readers are not thread-safe; benchmark.py evaluates rounds concurrently.
_ocr_lock = threading.Lock()
try:
    if not feature_on("ocr"):
        raise ImportError("ocr feature disabled")  # skip the model load entirely
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
                    with _ocr_lock:
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
        say(f"  [ocr error] {e}")
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
