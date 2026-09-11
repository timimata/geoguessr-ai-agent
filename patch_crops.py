import re

with open('bot.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update the signature of ask_model and its content builder
ask_model_signature = """def ask_model(
    png_bytes: bytes,
    extra_user_msgs: list[dict] | None = None,
    model_id: str | None = None,
) -> Guess:
    b64 = encode_image(png_bytes)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        },
    ]"""

new_ask_model = """def ask_model(
    png_bytes: bytes,
    extra_user_msgs: list[dict] | None = None,
    model_id: str | None = None,
    car_bytes: bytes | None = None,
    compass_bytes: bytes | None = None,
) -> Guess:
    b64 = encode_image(png_bytes)
    
    content = [
        {"type": "text", "text": "MAIN VIEW:\\n" + USER_PROMPT},
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
    ]"""

text = text.replace(ask_model_signature, new_ask_model)

crop_inject = """    shot_path.write_bytes(png)
    
    # Gerar os recortes detalhados para alimentar no prompt e aumentar a resolução visual do carro / bússola
    import io
    pil_img = Image.open(io.BytesIO(png)).convert("RGB")
    w_img, h_img = pil_img.size
    
    # 1. Car Meta
    car_box = (0, int(h_img * 0.70), w_img, h_img)
    car_cropped = pil_img.crop(car_box)
    car_io = io.BytesIO()
    car_cropped.save(car_io, format="JPEG", quality=85)
    car_bytes = car_io.getvalue()
    
    # 2. Compass Meta
    compass_box = (0, int(h_img * 0.75), int(w_img * 0.20), h_img)
    compass_cropped = pil_img.crop(compass_box)
    compass_io = io.BytesIO()
    compass_cropped.save(compass_io, format="JPEG", quality=85)
    compass_bytes = compass_io.getvalue()"""

text = text.replace("    shot_path.write_bytes(png)", crop_inject)

text = text.replace(
    "guess = ask_model(png, extra_user_msgs=first_msgs,",
    "guess = ask_model(png, extra_user_msgs=first_msgs, car_bytes=car_bytes, compass_bytes=compass_bytes,"
)

text = text.replace(
    "guess = ask_model(png, extra_user_msgs=first_msgs)",
    "guess = ask_model(png, extra_user_msgs=first_msgs, car_bytes=car_bytes, compass_bytes=compass_bytes)"
)

text = text.replace(
    "guess2 = ask_model(png, extra_user_msgs=first_msgs,",
    "guess2 = ask_model(png, extra_user_msgs=first_msgs, car_bytes=car_bytes, compass_bytes=compass_bytes,"
)

text = text.replace(
    "guess2 = ask_model(png, extra_user_msgs=",
    "guess2 = ask_model(png, car_bytes=car_bytes, compass_bytes=compass_bytes, extra_user_msgs="
)

with open('bot.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Patch applied!")
