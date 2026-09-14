"""Environment, paths, feature flags and prompts.

Imported first by everything else: load_dotenv() runs here, so any module that
reads an environment variable must import this one rather than call os.getenv
before it has run.
"""
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

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


def start_round_clock() -> float:
    """Reset the clock that _ts() counts from. Called once per round."""
    global _round_started_at
    _round_started_at = time.time()
    return _round_started_at


def round_elapsed() -> float:
    return time.time() - _round_started_at if _round_started_at > 0 else 0.0


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

# Feature flags: every heuristic that touches the prompt or the final pin can be
# switched off with FEATURES_OFF="soil,blacklist" in .env, or benchmark.py --off.
FEATURES: dict[str, bool] = {
    "blacklist": True,             # "you recently guessed X wrong" warning
    "ocr": True,                   # EasyOCR text → prompt
    "ocr_script": True,            # non-Latin script → hard country-set constraint
    "ocr_literal": True,           # verbatim country name in OCR → hard constraint + review
    "compass": True,               # OpenCV compass heading + crop image + hemisphere hint
    "rag": True,                   # CLIP nearest-neighbour reference rounds
    # OFF: rag_calibrate.py measured the bottom-30%-of-frame crop naming the right
    # country 20.9% of the time, below the 21.9% you get by always answering
    # "United States". It also costs an embedding and a query per round, and the
    # crop itself has never been sent to the model. Re-measure before re-enabling.
    "rag_car": False,              # separate nearest-neighbour search on the car crop
    # OFF: the ~1500-token GeoGuessr tradecraft half of the system prompt, sent on
    # every call, measured as a net loss. On a 150-round set with StreetCLIP
    # retrieval, dropping it took country accuracy from 84.0% to 88.7% and the
    # mean round score from 4089 to 4186. The model's own geolocation priors beat
    # the hand-written rules, and the rules crowd out the rest of the context.
    # The output contract half is always sent.
    "rich_system_prompt": False,
    "soil": True,                  # HSV ground-colour biome hint
    "metas": True,                 # Plonkit cheat sheets for RAG top-3 + disambiguation block
    "india_hint": True,
    "south_africa_hint": True,
    "region_prefilter": True,      # RAG-majority continent lean (prepended)
    "confusion_pairs": True,       # data-driven near-miss warnings
    "correction_call": True,       # ONE review call when automatic checks find issues
    # OFF: measured on the 150-round benchmark set (gemini-3.1-flash-lite). It fired
    # 6 times, changed the answer twice, and both changes were losses — one turned a
    # correct 22 km answer into 738 km. The other 4 fires cost a model call each and
    # changed nothing. Re-measure before switching it back on.
    "correction_rag_continent": False,  # RAG-unanimous continent disagreement is an issue
    "correction_cand_rag": True,       # candidate×RAG disagreement is an issue (never fired yet)
    "clamp": True,                 # polygon clamp when coords fall outside the country
    "calibration": True,           # replace raw confidence with historical hit-rate
    # OFF: measured on the same set. It fired 8 times for a total of +21 score points
    # across 150 rounds (+0.1/round, i.e. nothing) and flipped one correct country to
    # a wrong one. The model's errors are either small or continental, so there is
    # little middle ground for a hedge to win. Keep the flag: worth re-measuring on a
    # larger set or a model with flatter candidate distributions.
    "hedge": False,                # expected-score pin hedging between candidates
}
for _f in (s.strip() for s in os.getenv("FEATURES_OFF", "").split(",")):
    if _f in FEATURES:
        FEATURES[_f] = False


def feature_on(name: str) -> bool:
    return FEATURES.get(name, True)

# The system prompt has two halves with very different costs.
#
# SYSTEM_PROMPT_CONTRACT is the output contract: the JSON shape, the coordinate
# sign rules and the confidence calibration. It is small and non-negotiable.
#
# SYSTEM_PROMPT_GUIDANCE is roughly 1500 tokens of GeoGuessr tradecraft sent on
# every single call. It has never been measured. The "rich_system_prompt"
# feature flag drops it so benchmark.py can say whether it earns its place.
SYSTEM_PROMPT_GUIDANCE = """You are a GeoGuessr expert. Before answering, silently identify these concrete signals in the image (do NOT include them in output, only use them to decide):

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

"""

SYSTEM_PROMPT_CONTRACT = """Then output ONE JSON object only. No text before or after, no code fences, no chain-of-thought.

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

SYSTEM_PROMPT = SYSTEM_PROMPT_GUIDANCE + SYSTEM_PROMPT_CONTRACT


def active_system_prompt() -> str:
    """The system prompt for this run, honouring the rich_system_prompt flag."""
    return SYSTEM_PROMPT if feature_on("rich_system_prompt") else SYSTEM_PROMPT_CONTRACT


# --- output control -------------------------------------------------------
# benchmark.py replays hundreds of rounds at once and sets QUIET, so the modules
# below the entry point report through say() instead of print(). The live entry
# point (bot.py) prints directly: its output is the thing the user is watching.
QUIET = False


def say(*args, **kwargs) -> None:
    if not QUIET:
        print(*args, **kwargs)


USER_PROMPT = (
    "Identify where this Street View was captured. "
    "Remember: Western hemisphere (Americas) = NEGATIVE longitude. "
    "Eastern hemisphere (Europe, Africa, Asia, Oceania) = POSITIVE longitude. "
    "Southern hemisphere = NEGATIVE latitude. "
    "Return only the JSON."
)
