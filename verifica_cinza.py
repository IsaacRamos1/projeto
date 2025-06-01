import os
from PIL import Image
import numpy as np

def is_grayscale(image_path):
    img = Image.open(image_path).convert('RGB')  # Garante formato RGB para checagem
    img_np = np.array(img)
    
    if len(img_np.shape) < 3:  # Imagem já é 1 canal
        return True
    
    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    return np.array_equal(r, g) and np.array_equal(g, b)

def check_all_grayscale(directory):
    not_grayscale = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                path = os.path.join(root, file)
                if not is_grayscale(path):
                    not_grayscale.append(path)
    
    if not_grayscale:
        print(f"🔴 {len(not_grayscale)} imagens não estão em escala de cinza:")
        for path in not_grayscale:
            print(f" - {path}")
    else:
        print("✅ Todas as imagens estão em escala de cinza.")

# Caminho da pasta
check_all_grayscale('data_ceratite')
