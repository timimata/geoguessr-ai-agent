import os
from pathlib import Path
from PIL import Image

def test_crops():
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
    print(f"\nA testar os recortes visuais na imagem mais recente: {latest_image.name}")
    
    try:
        img = Image.open(latest_image).convert("RGB")
        w, h = img.size
        print(f"Resolução da imagem original: {w}x{h}")
        
        # 1. Simular o corte do Car Meta (Últimos 30% da altura, toda a largura)
        car_box = (0, int(h * 0.70), w, h)
        car_cropped = img.crop(car_box)
        car_out = "test_car_meta.jpg"
        car_cropped.save(car_out, format="JPEG", quality=85)
        print(f"✅ Corte 'Car Meta' guardado como '{car_out}'.")
        print("   -> (Abre este ficheiro para confirmar se o carro/capô está bem enquadrado na base)")
        
        # 2. Simular o corte do Compass Meta (Canto inferior esquerdo: 20% largura, 25% altura do fundo)
        compass_box = (0, int(h * 0.75), int(w * 0.20), h)
        compass_cropped = img.crop(compass_box)
        compass_out = "test_compass_meta.jpg"
        compass_cropped.save(compass_out, format="JPEG", quality=85)
        print(f"✅ Corte 'Compass Meta' guardado como '{compass_out}'.")
        print("   -> (Abre este ficheiro para confirmar se apanhou o widget da bússola de forma legível)")

    except Exception as e:
        print(f"Erro durante o teste: {e}")

if __name__ == "__main__":
    test_crops()