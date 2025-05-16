import torch
import datetime
import torchvision
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class EdgeEnhancement(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(EdgeEnhancement, self).__init__(always_apply, p)

    def apply(self, img, **params):
        # Aplica filtro Laplaciano para realce de bordas
        laplacian = cv2.Laplacian(img, ddepth=cv2.CV_64F)
        laplacian = np.uint8(np.clip(laplacian, 0, 255))

        # Combina a imagem original com o realce de bordas
        enhanced = cv2.addWeighted(img, 0.8, laplacian, 0.2, 0)
        return enhanced


class AlbumentationsTransform:
    def __init__(self):
        self.transform = Compose([
            Resize(224, 224),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            CLAHE(p=0.5),
            Rotate(limit=45, p=0.5),
            EdgeEnhancement(p=0.5),
            ToTensorV2()
        ])

    def __call__(self, img):
        img = np.array(img)  # PIL Image -> numpy array
        augmented = self.transform(image=img)
        return augmented['image']


def load_cifar10(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return dataset


def load_test_val_from_folder(data_dir, batch_size=32):
    #transform = AlbumentationsTransform()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    #val_dataset = ImageFolder(root=f'../{data_dir}/val', transform=transform)
    test_dataset = ImageFolder(root=f'../{data_dir}/test', transform=transform)

    #valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Também podemos extrair as classes
    classes = test_dataset.classes  # Lista com os nomes das classes

    return None, testloader, classes

def load_dataset_kfold(data_dir):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(root=data_dir, transform=transform)
    return dataset

def print_dataset_sizes(data_dir):
    sets = ['train', 'val', 'test']
    print("\n Distribuição de imagens nas pastas:")
    for set_name in sets:
        path = os.path.join(data_dir, set_name)
        num_images = len(glob(os.path.join(path, '**', '*.*'), recursive=True))
        print(f"{set_name.capitalize():5s}: {num_images} imagens")
    print("\n")

def plot_and_save_confusion_matrix(y_true, y_pred, class_names, fold, model_name):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matriz de Confusão - Fold {fold}')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    filename = f"resultados/{model_name}_confusion_matrix_fold{fold}_{timestamp}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Matriz de confusão salva como {filename}")

def plot_and_save_roc_auc(y_true, y_pred, class_names, fold, model_name):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    y_pred_bin = label_binarize(y_pred, classes=list(range(n_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title(f'Curva ROC - Fold {fold}')
    plt.legend(loc="lower right")
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    filename = f"resultados/{model_name}_roc_curve_fold{fold}_{timestamp}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Curva ROC/AUC salva como {filename}")


def load_checkpoint_and_evaluate(checkpoint_path, model_name, num_classes, testloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 1. Recria a arquitetura do modelo
    trainer = CNNTrainer(model_name=model_name, num_classes=num_classes)
    model = trainer.model
    model.to(device)
    
    # 2. Carrega o checkpoint (o arquivo É o state_dict)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)  # <- direto, sem ['model_state_dict']
    
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Testando modelo carregado"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print("\nResultados do modelo carregado:")
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return acc, prec, rec, f1

#classes = ('BASH', 'BBH', 'GMA', 'SHC', 'TSH')

# models = resnet18, mobilenetv3, resnet18_mobilenetv3, vgg16_mobilenetv3, vgg16, densenet161,
#          densenet161_mobilenetv3, inceptionv3

model_name = 'densenet161_mobilenetv3'

#print('testando_checkpoit.pt...')
#_, testloader, classes = load_test_val_from_folder(batch_size=8)
#checkpoint_path = 'checkpoint.pt'
#num_classes = len(classes)
#exit()

if __name__ == '__main__':
    batch_size = 16
    print_dataset_sizes('../data_ceratite')

    #dataset = load_cifar10(batch_size)
    dataset = load_dataset_kfold(data_dir='../data_ceratite/train')

    kfold = KFold(n_splits=7, shuffle=True, random_state=42)

    accuracies, precisions, recalls, f1s, aucs = [], [], [], [], []
    metrics_per_fold = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        _, testloader, classes = load_test_val_from_folder(data_dir='data_ceratite', batch_size=batch_size)

        trainer = CNNTrainer(model_name=model_name, fold=fold+1,patience=5)
        trainer.train(trainloader, valloader, epochs=100, lr=0.0001, fold=fold+1)

        print(f"Testando no conjunto de teste completo após Fold {fold+1}")
        _, test_acc, test_prec, test_rec, test_f1, test_auc = trainer.evaluate(testloader, type='teste')
        print(f"Acurácia no Teste: {test_acc:.4f} | Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | F1-Score: {test_f1:.4f} | AUC: {test_auc:.4f}")
        
        accuracies.append(test_acc)
        precisions.append(test_prec)
        recalls.append(test_rec)
        f1s.append(test_f1)
        aucs.append(test_auc)

        # Armazena as métricas deste fold
        metrics_per_fold.append(
            f"Fold {fold+1}:\n"
            f"Acurácia: {test_acc:.4f}\n"
            f"Precisão: {test_prec:.4f}\n"
            f"Recall: {test_rec:.4f}\n"
            f"F1-Score: {test_f1:.4f}\n"
            f"AUC: {test_auc:.4f}\n"
            "--------------------------------------------\n"
        )

        # GERAR MATRIZ DE CONFUSÃO DO CHECKPOINT
        y_true = []
        y_pred = []
        model = trainer.model
        state_dict = torch.load(f'checkpoint_{fold+1}.pt', map_location='cuda')
        model.load_state_dict(state_dict)
        model.eval()
        
        with torch.no_grad():  # TESTE DO CHECKPOINT
            for inputs, labels in testloader:
                inputs, labels = inputs.to(trainer.device), labels.to(trainer.device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        plot_and_save_confusion_matrix(y_true, y_pred, classes, fold+1, model_name) # MATRIZ DE CONFUSÃO DO CHECKPOIT COM IMAGENS DE TESTE
        plot_and_save_roc_auc(y_true, y_pred, classes, fold+1, model_name)

    mean_acc = sum(accuracies) / len(accuracies)
    mean_prec = sum(precisions) / len(precisions)
    mean_rec = sum(recalls) / len(recalls)
    mean_f1 = sum(f1s) / len(f1s)
    mean_auc = sum(aucs) / len(aucs)

    print("\n><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><")
    print("Resultados Médios após 5 Folds:")
    print(f"Acurácia Média: {mean_acc:.4f}")
    print(f"Precision Média: {mean_prec:.4f}")
    print(f"Recall Médio: {mean_rec:.4f}")
    print(f"F1-Score Médio: {mean_f1:.4f}")
    print(f"AUC Médio: {mean_auc:.4f}")
    print("><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><\n")

    metrics_text = "\n".join(metrics_per_fold)  # Junta métricas de cada fold
    metrics_text += (
        "\nMÉTRICAS MÉDIAS FINAIS:\n"
        f"Acurácia Média: {mean_acc:.4f}\n"
        f"Precision Média: {mean_prec:.4f}\n"
        f"Recall Médio: {mean_rec:.4f}\n"
        f"F1-Score Médio: {mean_f1:.4f}\n"
        f"AUC Médio: {mean_auc:.4f}\n"
    )


    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    output_path = f"resultados/{model_name}_metrics_{timestamp}.txt"

    # Salva todas as métricas no arquivo
    with open(output_path, "w") as f:
        f.write(metrics_text)