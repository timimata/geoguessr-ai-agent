import cv2
import numpy as np
import math
from pathlib import Path

def analyze_compass():
    img_path = "test_compass_meta.jpg"
    if not Path(img_path).exists():
        print(f"Erro: {img_path} não existe.")
        return

    img = cv2.imread(img_path)
    if img is None:
        print("Erro ao decodificar imagem.")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Vamos alargar ligeiramente as máscaras vermelhas do HSV, pois as vezes as cores são esbatidas (NMPZ com nuvens)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Não foi detetada a agulha vermelha da bússola.")
        return

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 2:  # Bússola no canto costuma ser pequena
        print("Área vermelha muito pequena. Ruído.")
        return

    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return
    rx = int(M["m10"] / M["m00"])
    ry = int(M["m01"] / M["m00"])
    
    cv2.circle(img, (rx, ry), 4, (0, 255, 0), -1)

    # Encontrar o círculo principal da bússola
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=60)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # Encontrar o círculo mais próximo da nossa agulha vermelha
        best_circle = None
        min_dist = float('inf')
        for i in circles[0, :]:
            cx_c, cy_c, r_c = i[0], i[1], i[2]
            dist = (cx_c - rx)**2 + (cy_c - ry)**2
            if dist < min_dist:
                min_dist = dist
                best_circle = i
                
        if best_circle is not None:
            cx, cy, r = best_circle[0], best_circle[1], best_circle[2]
            cv2.circle(img, (cx, cy), r, (255, 0, 0), 2)
            cv2.circle(img, (cx, cy), 2, (0, 0, 255), 3)
            
            # Calcular o ângulo usando o centro do círculo e a agulha
            dx = rx - cx
            dy = ry - cy
            
            # No OpenCV, Y aumenta para baixo.
            # Se a agulha vermelha aponta para CIMA (Norte na imagem), dy negativo, dx 0. atan2(-1, 0) = -90º.
            # No GeoGuessr:
            # - Se Norte está para CIMA, a câmara está virada para NORTE.
            # - Se Norte está para a DIREITA, a câmara está virada para OESTE (West).
            # - Se Norte está para a ESQUERDA, a câmara está virada para ESTE (East).
            # - Se Norte está para BAIXO, a câmara está virada para SUL.
            
            rads = math.atan2(dy, dx)
            degs = math.degrees(rads)
            
            # Converter de coordenadas de ecrã para ângulo cardinal (-90 -> Norte=0, 0 -> Oeste=270, 90 -> Sul=180, 180/-180 -> Este=90)
            facing_deg = (-degs - 90) % 360
            
            dirs = ["Norte", "Nordeste", "Este", "Sudeste", "Sul", "Sudoeste", "Oeste", "Noroeste"]
            idx = int(round(facing_deg / 45.0)) % 8
            compass_dir = dirs[idx]
            
            print(f"Centro: ({cx},{cy}), Agulha: ({rx},{ry})")
            print(f"Ângulo Interno: {degs:.1f}º -> A Câmara está virada para: {compass_dir}")
            
            cv2.putText(img, compass_dir, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite("compass_debug.jpg", img)
            print("✅ Análise guardada em compass_debug.jpg para visualização.")
        else:
            print("Nenhum círculo correspondente encontrado nas proximidades da agulha.")
    else:
        print("Nenhum círculo de bússola detetado.")

if __name__ == "__main__":
    analyze_compass()