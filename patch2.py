with open('bot.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace(
    "guess = ask_model(png, extra_user_msgs=use_extra, model_id=model_used)",
    "guess = ask_model(png, extra_user_msgs=use_extra, model_id=model_used, car_bytes=car_bytes, compass_bytes=compass_bytes)"
)

with open('bot.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Patch 2 applied!")
