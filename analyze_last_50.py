"""analyze_last_50.py - quick read on how the bot has been doing lately.

    python analyze_last_50.py           # last 50 rounds
    python analyze_last_50.py 200       # last 200

Rounds with no result yet (a duel still waiting on an opponent, or a round the
bot failed to submit) are counted separately rather than folded in as zero.
"""
import sys
from collections import Counter

from storage import load_rounds, log_path


def country_of(entry, key: str, fallback: str) -> str:
    value = entry.get(key)
    if isinstance(value, dict):
        return value.get("country") or "Unknown"
    return entry.get(fallback) or "Unknown"


def main() -> int:
    n = 50
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print(f"Uso: python {sys.argv[0]} [n]")
            return 1

    rounds = load_rounds()
    if not rounds:
        print(f"{log_path().name} está vazio ou não existe.")
        return 1
    recent = rounds[-n:]

    scored, unscored = [], 0
    for r in recent:
        if r.get("error_km") is None or not isinstance(r.get("actual"), dict):
            unscored += 1
        else:
            scored.append(r)

    print(f"--- ÚLTIMAS {len(recent)} RONDAS ({log_path().name}) ---")
    if unscored:
        print(f"Sem resultado: {unscored} (duelo por resolver ou palpite não submetido)")
    if not scored:
        print("Nenhuma ronda pontuada neste intervalo.")
        return 0

    hits = [r for r in scored if r.get("country_hit")]
    errors = sorted(r["error_km"] for r in scored)
    median = errors[len(errors) // 2]
    print(f"Precisão de país: {len(hits) / len(scored) * 100:.1f}% ({len(hits)}/{len(scored)})")
    print(f"Erro médio: {sum(errors) / len(errors):.0f} km   ·   mediana: {median:.0f} km")
    print(f"Abaixo de 200 km: {sum(e < 200 for e in errors) / len(errors) * 100:.0f}%"
          f"   ·   acima de 2000 km: {sum(e > 2000 for e in errors) / len(errors) * 100:.0f}%")

    by_model = Counter(r.get("model") or "(desconhecido)" for r in scored)
    if len(by_model) > 1:
        print("\n--- POR MODELO ---")
        for model, count in by_model.most_common():
            rows = [r for r in scored if (r.get("model") or "(desconhecido)") == model]
            h = sum(1 for r in rows if r.get("country_hit"))
            print(f"{model:<32} n={count:<4} país {h / count * 100:.0f}%")

    misses = [(country_of(r, "actual", "real_country"),
               country_of(r, "guess", "guess_country"))
              for r in scored if not r.get("country_hit")]
    if misses:
        print("\n--- TOP ERROS (real vs palpite) ---")
        for (actual, guessed), count in Counter(misses).most_common(10):
            print(f"{actual} vs {guessed}: {count}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
