import os
import cv2
from glob import glob
import numpy as np
from tqdm import tqdm

# Caminho base do diretório de treino
train_dir = '../data_ceratite_original_1112x4/train'
classes = ['AK', 'FK', 'NSK', 'Normal']

# Função para contar imagens
def count_images(class_path):
    return len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Descobrir classe majoritária
class_counts = {cls: count_images(os.path.join(train_dir, cls)) for cls in classes}
target_count = max(class_counts.values())
print("Classe majoritária:", max(class_counts, key=class_counts.get), "-", target_count, "imagens")

# Transforma uma imagem com 3 técnicas
def apply_transforms(img):
    hflip = cv2.flip(img, 1)
    vflip = cv2.flip(img, 0)

    # CLAHE (em cada canal separadamente)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    clahe_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return hflip, vflip, clahe_img

# Aplicar aumento nas classes minoritárias
for cls in classes:
    class_path = os.path.join(train_dir, cls)
    images = sorted([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    current_count = len(images)
    
    if current_count >= target_count:
        print(f"Classe {cls} já tem {current_count} imagens. Nenhum aumento necessário.")
        continue

    print(f"Aumentando classe {cls} ({current_count} -> {target_count})...")

    i = 0
    while len(os.listdir(class_path)) < target_count:
        img_name = images[i % len(images)]
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Erro ao ler imagem: {img_path}")
            i += 1
            continue

        # Nome base sem extensão
        base_name = os.path.splitext(img_name)[0]

        hflip, vflip, clahe = apply_transforms(img)

        # Salvar imagens aumentadas
        cv2.imwrite(os.path.join(class_path, f"{base_name}_hflip.jpg"), hflip)
        if len(os.listdir(class_path)) >= target_count: break

        cv2.imwrite(os.path.join(class_path, f"{base_name}_vflip.jpg"), vflip)
        if len(os.listdir(class_path)) >= target_count: break

        cv2.imwrite(os.path.join(class_path, f"{base_name}_clahe.jpg"), clahe)
        if len(os.listdir(class_path)) >= target_count: break

        i += 1
