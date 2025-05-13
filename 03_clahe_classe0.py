import os
import random
import numpy as np
from PIL import Image
from glob import glob
from albumentations import CLAHE
from tqdm import tqdm
import cv2

def apply_clahe_and_laplacian_overlay(class0_dir, prob_clahe=0.3):
    images = glob(os.path.join(class0_dir, '*.png')) + glob(os.path.join(class0_dir, '*.jpg'))
    clahe = CLAHE(p=1.0)  # Sempre aplica, mas o controle é externo via if

    print(f"Total de imagens: {len(images)}")
    print(f"Aplicando CLAHE com {int(prob_clahe*100)}% de chance")

    for img_path in tqdm(images):
        img = np.array(Image.open(img_path).convert('RGB'))
        original_img = img.copy()

        # Aplica CLAHE com probabilidade
        if random.random() < prob_clahe:
            transformed_img = clahe(image=img)['image']

            # Verifica se houve mudança real na imagem
            if not np.array_equal(original_img, transformed_img):
                Image.fromarray(transformed_img).save(img_path)
                print(f"Transformada: {img_path}")
            else:
                print(f"Transformação resultou em imagem igual, ignorando: {img_path}")

    print("Processo finalizado.")

# Exemplo de uso
apply_clahe_and_laplacian_overlay(class0_dir='../data_ceratite/train/AK', prob_clahe=0.5)
