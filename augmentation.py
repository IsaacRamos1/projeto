import torch
import numpy as np
from PIL import Image
import random
from albumentations import HorizontalFlip, VerticalFlip, CLAHE, Compose
from glob import glob
import cv2
import os
from tqdm import tqdm

def balance_dataset_via_augmentation(train_dir, target_multiplier=1.0):
    # Define as transformações aplicáveis com probabilidade
    augment = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        CLAHE(p=0.5),
    ])

    class_paths = [os.path.join(train_dir, cls_name) for cls_name in os.listdir(train_dir)]
    class_counts = {
        cls_path: len(glob(os.path.join(cls_path, '*.png'))) + len(glob(os.path.join(cls_path, '*.jpg')))
        for cls_path in class_paths
    }

    print("Distribuição inicial:", {os.path.basename(k): v for k, v in class_counts.items()})
    max_count = int(max(class_counts.values()) * target_multiplier)

    for class_path, count in class_counts.items():
        if count >= max_count:
            continue

        images = glob(os.path.join(class_path, '*.png')) + glob(os.path.join(class_path, '*.jpg'))
        n_to_generate = max_count - count
        print(f"Gerando {n_to_generate} imagens para a classe {os.path.basename(class_path)}")

        generated = 0
        attempts = 0
        max_attempts = n_to_generate * 10  # evita loops infinitos

        while generated < n_to_generate and attempts < max_attempts:
            attempts += 1
            img_path = random.choice(images)
            img = np.array(Image.open(img_path).convert('RGB'))
            augmented_img = augment(image=img)['image']

            # Se a imagem aumentada for igual à original, ignora
            if np.array_equal(img, augmented_img):
                continue

            augmented_pil = Image.fromarray(augmented_img)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            new_name = f"{base_name}_aug_{generated}.png"
            save_path = os.path.join(class_path, new_name)

            augmented_pil.save(save_path)
            generated += 1

    print("Aumento finalizado!")

# Exemplo de uso
balance_dataset_via_augmentation(train_dir='../data_ceratite/train', target_multiplier=0.9)
