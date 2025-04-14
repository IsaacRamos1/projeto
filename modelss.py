import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score
from pytorchtools import EarlyStopping


class CombinedVGG16MobileNetV3(nn.Module):
    def __init__(self, num_classes=5):
        super(CombinedVGG16MobileNetV3, self).__init__()

        # Backbone convolucional VGG16
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.vgg_features = vgg16.features
        self.vgg_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.vgg_fc = nn.Linear(512, 512)  # VGG16 tem 512 canais finais

        # Backbone convolucional MobileNetV3-Large
        mobilenet_v3_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.mobilenet_features = mobilenet_v3_large.features
        self.mobilenet_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.mobilenet_fc = nn.Linear(960, 512)  # MobileNetV3-Large tem 960 canais finais

    
        self.mlp = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Passa pela VGG16
        x1 = self.vgg_features(x)
        x1 = self.vgg_avgpool(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.vgg_fc(x1)

        # Passa pela MobileNetV3-Large
        x2 = self.mobilenet_features(x)
        x2 = self.mobilenet_avgpool(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.mobilenet_fc(x2)

        # Produto elemento a elemento
        x_combined = torch.mul(x1, x2)

        # MLP final
        output = self.mlp(x_combined)
        return output


class CombinedResNet18MobileNetV3(nn.Module):
    def __init__(self, num_classes=5):
        super(CombinedResNet18MobileNetV3, self).__init__()

        # Backbone convolucional Resnet18
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet_conv = nn.Sequential(*list(resnet18.children())[:-2])
        self.resnet_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.resnet_fc = nn.Linear(resnet18.fc.in_features, 512)

        # Backbone convolucional MobileNetV3
        mobilenet_v3 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.mobilenet_features = mobilenet_v3.features
        self.mobilenet_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.mobilenet_fc = nn.Linear(576, 512)  # Reduz para 512

        # MLP final (adaptado da MobileNetV3)
        self.mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Passa pela ResNet18
        x1 = self.resnet_conv(x)
        x1 = self.resnet_avgpool(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.resnet_fc(x1)

        # Passa pela MobileNetV3
        x2 = self.mobilenet_features(x)
        x2 = self.mobilenet_avgpool(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.mobilenet_fc(x2)

        # Produto elemento a elemento
        x_combined = torch.mul(x1, x2)
        x_min = x_combined.min(dim=1, keepdim=True)[0]   # Normalizacao
        x_max = x_combined.max(dim=1, keepdim=True)[0]
        x_combined = (x_combined - x_min) / (x_max - x_min + 1e-8)

        output = self.mlp(x_combined)
        return output


class CNNTrainer:
    def __init__(self, model_name, fold, num_classes=5, device=None, patience=5):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_pretrained_model(model_name, num_classes)
        self.model.to(self.device)
        self.early_stopping = EarlyStopping(patience=patience, fold=fold, verbose=True)

    def _load_pretrained_model(self, model_name, num_classes):
        if model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

        elif model_name == 'resnet18_mobilenetv3':
            self.model = CombinedResNet18MobileNetV3(num_classes=num_classes)
            model = self.model

        elif model_name == 'vgg16_mobilenetv3':
            self.model = CombinedVGG16MobileNetV3(num_classes=num_classes)
            model = self.model

        else:
            raise ValueError(f"Modelo '{model_name}' não suportado.")
        return model

    def train(self, trainloader, valloader, epochs=10, lr=0.001, fold=0):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            print(f"\nFold {fold} | Epoch {epoch+1}/{epochs}")
            print("--------------------------------------------------------------------------------")

            for inputs, labels in tqdm(trainloader, desc=f"Treinando Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(trainloader)
            val_loss, val_acc, _, _, _ = self.evaluate(valloader)
            print(f"Train Loss: {val_loss:.4f}  |  Val acc: {val_acc:4f}")

            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping acionado!")
                break

    def evaluate(self, dataloader, type='Validacao'):
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        all_preds = []
        all_labels = []
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc=type):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy =  correct / total
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        avg_loss = running_loss / len(dataloader)

        return avg_loss, accuracy, precision, recall, f1

    def predict(self, inputs):
        print("Iniciando Teste...")
        self.model.eval()
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
        return preds.cpu()