import torch
import torchvision
import numpy as np
from PIL import Image
import random
from modelss import CNNTrainer
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import albumentations as A
from albumentations import Compose, HorizontalFlip, VerticalFlip, CLAHE, Rotate, Resize
from albumentations.pytorch import ToTensorV2
from glob import glob
import cv2
import os
from tqdm import tqdm

def balance_dataset_via_augmentation(train_dir, target_multiplier=1.0):
    augment = A.Compose([
        Resize(224, 224, p=1.0),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Rotate(limit=45, p=0.5),
        CLAHE(p=0.5),
    ])

    class_paths = [os.path.join(train_dir, cls_name) for cls_name in os.listdir(train_dir)]
    class_counts = {cls_path: len(glob(os.path.join(cls_path, '*.png'))) + len(glob(os.path.join(cls_path, '*.jpg'))) for cls_path in class_paths}
    print("Distribuição inicial:", {os.path.basename(k): v for k, v in class_counts.items()})

    max_count = int(max(class_counts.values()) * target_multiplier)
    for class_path, count in class_counts.items():
        if count >= max_count:
            continue
        
        images = glob(os.path.join(class_path, '*.png')) + glob(os.path.join(class_path, '*.jpg'))
        n_to_generate = max_count - count

        print(f"Gerando {n_to_generate} imagens para a classe {os.path.basename(class_path)}")

        for i in tqdm(range(n_to_generate)):
            img_path = random.choice(images)
            img = np.array(Image.open(img_path).convert('RGB'))
            augmented = augment(image=img)['image']
            augmented = Image.fromarray(augmented)

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            new_name = f"{base_name}_aug_{i}.png"
            save_path = os.path.join(class_path, new_name)
            print(save_path)

            augmented.save(save_path)

    print("Aumento finalizado!")

balance_dataset_via_augmentation(train_dir='../data_original/train', target_multiplier=0.9)