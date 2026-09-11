import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROJECT_DIR = Path(__file__).parent.resolve()
METAS_DIR = PROJECT_DIR / "metas"
RESUMOS_DIR = PROJECT_DIR / "metas_resumidas"
RESUMOS_DIR.mkdir(exist_ok=True)

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
MODEL_ID = os.getenv("MODEL_ID", "qwen/qwen3.6-35b-a3b")

client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio", timeout=600.0)

PROMPT_SISTEMA = """You are an expert GeoGuessr AI assistant. 
Your task is to take a very long 'cheat sheet' text about a specific country and SUMMARIZE it into maximum 8-10 concise bullet points.
- Only keep VISUAL clues that are useful for NMPZ (No Move, Pan, Zoom) mode.
- Focus heavily on: road lines (color/dashes), utility poles (material/shape/insulators), unique architecture, language/alphabet quirks, distinct vegetation, unique bollards or license plate colors.
- EXCLUDE: Car meta (Google car roofs, antennas), camera generations (gen 2/3/4 sky artifacts), coverage histories, and any general chat.
- Keep the summary brutally short, info-dense and direct. Under 1000 characters total.

Respond ONLY with the summarized bullet points. No introductory text."""

def _strip_thinking(text: str) -> str:
    """Remove Qwen thinking preamble; return only the final bullet-point answer."""
    import re

    def _clean_bullet_block(lines: list[str]) -> str:
        """Strip markdown markers and drop truncated last line."""
        out = [re.sub(r'\*+([^*]+)\*+', r'\1', l) for l in lines]
        if out and out[-1].strip() and out[-1].strip()[-1] not in '.?!)':
            out = out[:-1]
        return '\n'.join(out).strip()

    def _extract_last_bullet_block(src: str, min_lines: int = 3) -> str | None:
        """Find the LAST contiguous block of bullet lines with at least min_lines entries."""
        blocks: list[list[str]] = []
        current: list[str] = []
        for line in src.splitlines():
            s = line.strip()
            if s.startswith('- ') or s.startswith('• '):
                current.append(s)
            else:
                if len(current) >= min_lines:
                    blocks.append(current)
                current = []
        if len(current) >= min_lines:
            blocks.append(current)
        if blocks:
            result = _clean_bullet_block(blocks[-1])
            if result:
                return result
        return None

    # 1. Strip explicit <think>...</think> blocks then check if clean
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    first = cleaned.splitlines()[0].strip() if cleaned else ''
    if first.startswith('-') or first.startswith('•'):
        return _clean_bullet_block(cleaned.splitlines()) or cleaned

    # 2. Model output the thinking as plain prose — extract the last bullet block
    #    (the final answer always appears as the last ≥3-line bullet sequence)
    result = _extract_last_bullet_block(cleaned or text)
    if result:
        return result

    # 3. Fallback: strip everything before the first bullet of any indentation
    m = re.search(r'^\s*[-•]', text, re.MULTILINE)
    if m and m.start() > 0:
        return text[m.start():].strip()

    return text.strip()


def get_summary(text: str, country: str) -> str:
    try:
        print(f"   -> A enviar pedido para {LM_STUDIO_URL} usando modelo {MODEL_ID}...")
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user", "content": f"Summarize the NMPZ visual clues for {country.upper()}:\n\n{text}"}
            ],
            temperature=0.2,
            max_tokens=1500,
            extra_body={"thinking": {"type": "disabled"}},
        )

        # Qwen-specific: reasoning may arrive in reasoning_content or inside <think> tags in content
        content = resp.choices[0].message.content or ""
        thinking = getattr(resp.choices[0].message, "reasoning_content", None) or ""

        if (not content.strip()) and thinking.strip():
            print("   [Aviso] Qwen: resposta em reasoning_content, a extrair bullets...")
            content = thinking

        content = _strip_thinking(content)

        if not content:
            print(f"   [Erro API] O modelo devolveu uma resposta vazia! (Verifica o LM Studio)")
            return ""

        return content.strip()
    except Exception as e:
        print(f"   [Erro API Detalhado] {e}")
        return ""

def main():
    if not METAS_DIR.exists():
        print(f"Directory {METAS_DIR} not found.")
        return

    files = list(METAS_DIR.glob("*.txt"))
    # Ordenar por ordem alfabética para ser previsível e podermos continuar de onde parámos
    files.sort(key=lambda x: x.stem.lower())
    
    print(f"Found {len(files)} cheat sheets. Checking which need summarization...")

    for f in files:
        country = f.stem
                
        out_file = RESUMOS_DIR / f"{country}.txt"
        
        if out_file.exists():
            print(f"[skip] {country} - already summarized.")
            continue

        print(f"[...] Summarizing {country}...")

        try:
            with open(f, "r", encoding="utf-8") as file:
                content = file.read()

            # Se já for pequeno, copiamos logo
            if len(content) < 800:
                print(f"   -> Already very short. Direct copy.")
                summary = content
            else:
                summary = get_summary(content, country)

            if summary:
                with open(out_file, "w", encoding="utf-8") as out:
                    out.write(summary)
                print(f"[ok] Saved {country} ({len(content)} -> {len(summary)} chars).")
            else:
                print(f"[fail] Failed to summarize {country}.")
                
        except Exception as e:
            print(f"[error] {country}: {e}")

if __name__ == "__main__":
    main()
