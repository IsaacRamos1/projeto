import os
import random

# Caminho base para o conjunto de treino
train_base_path = '../data_ceratite_original_226x4/train'

# Lista das classes
classes = ['AK', 'FK', 'NSK', 'Normal']

# Função para contar imagens por classe
def count_images(folder_path):
    return len([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Descobrir a menor quantidade de imagens entre as classes
class_counts = {}
for class_name in classes:
    folder = os.path.join(train_base_path, class_name)
    if os.path.exists(folder):
        class_counts[class_name] = count_images(folder)
    else:
        print(f"Aviso: Pasta não encontrada: {folder}")

min_count = min(class_counts.values())
print(f"\nMenor classe: {min_count} imagens\n")

# Função para fazer undersampling
def undersample_class(folder_path, target):
    all_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(all_images) > target:
        to_remove = len(all_images) - target
        remove_list = random.sample(all_images, to_remove)
        for img_name in remove_list:
            os.remove(os.path.join(folder_path, img_name))
        print(f"[{folder_path}] -> {to_remove} imagens removidas")
    else:
        print(f"[{folder_path}] já tem {len(all_images)} imagens ou menos. Nenhuma remoção necessária.")

# Aplicar undersampling usando a menor quantidade como referência
for class_name in classes:
    folder = os.path.join(train_base_path, class_name)
    if os.path.exists(folder):
        undersample_class(folder, min_count)
