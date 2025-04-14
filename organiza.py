import os
import shutil
import random

# Defina seu diretório base
base_dir = '../data_original'
classes = ['BASH', 'BBH', 'GMA', 'SHC', 'TSH']

# Define a porcentagem para split
train_split = 0.8
val_split = 0.1
test_split = 0.1

# Cria as pastas train, val e test
for split in ['train', 'val', 'test']:
    split_path = os.path.join(base_dir, split)
    os.makedirs(split_path, exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(split_path, cls), exist_ok=True)

# Função para mover os arquivos
def move_files(file_list, src_folder, dst_folder, class_name):
    for f in file_list:
        src_path = os.path.join(src_folder, f)
        dst_path = os.path.join(dst_folder, class_name, f)
        shutil.move(src_path, dst_path)

# Faz o split para cada classe
for cls in classes:
    cls_folder = os.path.join(base_dir, cls)
    if not os.path.isdir(cls_folder):
        continue

    images = [f for f in os.listdir(cls_folder) if os.path.isfile(os.path.join(cls_folder, f))]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    train_files = images[:n_train]
    val_files = images[n_train:n_train + n_val]
    test_files = images[n_train + n_val:]

    move_files(train_files, cls_folder, os.path.join(base_dir, 'train'), cls)
    move_files(val_files, cls_folder, os.path.join(base_dir, 'val'), cls)
    move_files(test_files, cls_folder, os.path.join(base_dir, 'test'), cls)

    # Remove a pasta original se ficar vazia
    if not os.listdir(cls_folder):
        os.rmdir(cls_folder)

print("Organização concluída!")
