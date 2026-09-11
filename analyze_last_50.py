import json
from collections import Counter
import copy

data = []
with open("log.json", "r", encoding="utf-8") as f:
    try:
        full_data = json.load(f)
        if isinstance(full_data, list):
            data = full_data
    except Exception as e:
        print(f"Erro ao ler JSON: {e}")

# Filtrar as ultimas 50 rondas
last_50 = data[-50:]

total = len(last_50)
country_matches = 0
total_error_km = 0
errors = []

for r in last_50:
    t_country = r.get("actual", {}).get("country", "Unknown") if isinstance(r.get("actual"), dict) else r.get("actual_country", "Unknown")
    
    g_country = r.get("guess", {}).get("country", "Unknown") if isinstance(r.get("guess"), dict) else r.get("guess_country", "Unknown")
    if t_country == "Unknown": t_country = r.get("real_country", "Unknown")
    
    dist = r.get("error_km", 0)
    
    total_error_km += dist
    if t_country == g_country:
        country_matches += 1
    else:
        errors.append((t_country, g_country, dist))

avg_error = total_error_km / total if total > 0 else 0
acc = (country_matches / total) * 100 if total > 0 else 0

print(f"--- ANÁLISE DAS ÚLTIMAS {total} RONDAS ---")
print(f"Precisão de País: {acc:.1f}% ({country_matches}/{total})")
print(f"Erro Médio: {avg_error:.0f} km")

print("\n--- TOP ERROS (Real vs Advinha) ---")
error_counts = Counter(f"{t} vs {g}" for t, g, d in errors)
for pair, count in error_counts.most_common(10):
    print(f"{pair}: {count} vezes")
