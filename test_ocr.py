import os
import time
from pathlib import Path
from PIL import Image
import numpy as np

try:
    import easyocr
    print("A carregar modelo OCR (pode demorar alguns segundos da primeira vez)...")
    reader = easyocr.Reader(['en', 'ru'], gpu=True)
except ImportError:
    print("Erro: easyocr não está instalado.")
    exit(1)

def test_last_screenshot():
    screenshots_dir = Path("screenshots")
    if not screenshots_dir.exists():
        print("Pasta screenshots/ não encontrada.")
        return

    # Obter a imagem mais recente
    images = list(screenshots_dir.glob("*.png"))
    if not images:
        print("Nenhuma imagem encontrada na pasta screenshots/.")
        return
        
    latest_image = max(images, key=os.path.getmtime)
    print(f"\nA testar OCR na imagem mais recente: {latest_image.name}")
    
    try:
        img = Image.open(latest_image).convert("RGB")
        w, h = img.size
        # Cortar os 15% do topo e os 20% de baixo do ecrã para limpar o lixo do GeoGuessr
        crop_box = (0, int(h * 0.15), w, int(h * 0.80))
        img_cropped = img.crop(crop_box)
        
        # Teste visual para o utilizador: vamos guardar o ficheiro que o OCR vê
        img_cropped.save("test_ocr_vision.jpg", format="JPEG", quality=85)
        print("✅ Guardada uma imagem de teste: 'test_ocr_vision.jpg'")
        print("   -> (Abre para ver a 'faixa limpa' sem os menus que o OCR passa agora a analisar)")

        img_np = np.array(img_cropped)
        
        start_time = time.time()
        print("A extrair texto da imagem...")
        results = reader.readtext(img_np, detail=0)
        end_time = time.time()
        
        print(f"Tempo de execução: {end_time - start_time:.2f} segundos")
        
        texts = [t for t in results if len(t.strip()) >= 3]
        if texts:
            print("\n✅ Texto válido encontrado (entrará no prompt do bot):")
            for t in texts:
                print(f"  - {t}")
            
            joined_text = " | ".join(texts)
            print(f"\nString final enviada ao LLM: '{joined_text}'")
        else:
            if results:
                print(f"\n⚠️ Todo o texto detetado era ruído com menos de 3 caracteres: {results}")
            else:
                print("\n❌ Nenhum texto detetado na imagem.")
                
    except Exception as e:
        print(f"Erro durante o teste OCR: {e}")

if __name__ == "__main__":
    test_last_screenshot()