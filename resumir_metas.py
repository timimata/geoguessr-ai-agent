import os
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio", timeout=120.0)

def summarize_metas(test_mode=False):
    metas_dir = Path("metas")
    out_dir = Path("metas_resumidas")
    out_dir.mkdir(exist_ok=True)
    
    if not metas_dir.exists():
        print("Pasta metas/ não encontrada.")
        return
        
    files = list(metas_dir.glob("*.txt"))
    if not files:
        print("Nenhum ficheiro na pasta metas/.")
        return
        
    print(f"Encontrados {len(files)} ficheiros. Destino: pastas metas_resumidas/")
    if test_mode:
        files = files[:3]
        print("⚠️ MODO TESTE (Apenas os primeiros 3 ficheiros serão processados)\n")
        
    system_prompt = (
        "You are an expert GeoGuessr player. Condense the provided text into a highly dense, "
        "bulleted cheat sheet of MAX 100 words. Focus STRICTLY on hard visual clues: "
        "road lines, utility poles, license plates, bollards, architecture, and Google car meta. "
        "Do NOT include conversational intro/outro text, just the raw facts."
    )
    
    count = 0
    for f in files:
        out_file = out_dir / f.name
        if out_file.exists() and not test_mode:
            continue
            
        content = f.read_text(encoding="utf-8")
        if len(content) < 300:
            out_file.write_text(content, encoding="utf-8")
            continue
            
        print(f"A resumir: {f.name} (Tamanho original: {len(content)} chars)")
        # Truncar se for muito massivo para manter a geração rápida no LM local
        content_to_send = content[:7000] 
        
        try:
            start = time.time()
            resp = client.chat.completions.create(
                model="local-model", # O LM Studio ignora mas exige este campo
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarize this for GeoGuessr:\n\n{content_to_send}"}
                ],
                temperature=0.3,
                max_tokens=250
            )
            summary = resp.choices[0].message.content.strip()
            out_file.write_text(summary, encoding="utf-8")
            dur = time.time() - start
            
            print(f" ✅ Concluído ({dur:.1f}s). Reduzido de {len(content)} para {len(summary)} chars.")
            print(f"    Preview: {summary[:120].replace('\n', ' ')}...\n")
            count += 1
            
        except Exception as e:
            print(f" ❌ Erro ao resumir {f.name}: {e}")

    print(f"Fim! {count} ficheiros resumidos.")

if __name__ == "__main__":
    import sys
    is_test = len(sys.argv) > 1 and sys.argv[1] == "--test"
    summarize_metas(test_mode=is_test)