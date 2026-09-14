"""The model client, the request it sends and the answer it parses back."""
import json
import re
from dataclasses import dataclass, field

from openai import OpenAI

from config import (MODEL_ID, OPENAI_API_KEY, OPENAI_BASE_URL, USER_PROMPT,
                    _CLIENT_TIMEOUT, active_system_prompt, say)
from vision import encode_image

client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY, timeout=_CLIENT_TIMEOUT)


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
        {"role": "system", "content": active_system_prompt()},
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
        say(f"  [thinking] {short}{'…' if len(thinking) > 220 else ''}")
        
    if not raw.strip() and thinking.strip():
        # O modelo (ex: Qwen) pode colocar acidentalmente o JSON final dentro do bloco de raciocínio
        raw = thinking

    if not raw.strip():
        raise ValueError("empty model output (check LM Studio logs — image may have been rejected)")
    
    return parse_guess(raw)


def _parse_confidence(value) -> float:
    """Models sometimes answer 85 or 0.85 (or garbage). Normalise to [0, 1]."""
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.0
    if c != c:  # NaN
        return 0.0
    if c > 1.0:
        c = c / 100.0
    return min(max(c, 0.0), 1.0)


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
        confidence=_parse_confidence(data.get("confidence", 0.0)),
        reasoning=str(data.get("reasoning", "")),
        candidates=candidates,
    )
