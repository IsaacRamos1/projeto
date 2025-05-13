import os
from collections import Counter

base_dir = '../data_ceratite_original'
splits = ['train', 'test', 'val']
total_geral = 0
total_por_classe = Counter()

for split in splits:
    split_path = os.path.join(base_dir, split)
    class_counts = Counter()
    total_split = 0

    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if os.path.isdir(class_path):
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            count = len(image_files)
            class_counts[class_name] = count
            total_por_classe[class_name] += count
            total_split += count

    print(f"\nContagem de imagens em '{split}':")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} imagens")
    print(f"Total em '{split}': {total_split} imagens")
    total_geral += total_split

print("\nTotal de imagens por classe (train + test):")
for class_name, count in total_por_classe.items():
    print(f"{class_name}: {count} imagens")

print(f"\nTotal geral de imagens: {total_geral}")
