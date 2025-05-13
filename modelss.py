import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from pytorchtools import EarlyStopping
import matplotlib.pyplot as plt
from teste import LightSelfAttention

class CombinedDenseNet161MobileNetV3(nn.Module):
    def __init__(self, num_classes=4):
        super(CombinedDenseNet161MobileNetV3, self).__init__()

        # DenseNet161
        densenet = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        self.densenet_features = densenet.features
        self.densenet_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.densenet_fc = nn.Linear(2208, 960)  # DenseNet161 termina com 2208 canais

        # MobileNetv3_Large
        mobilenet_v3_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.mobilenet_features = mobilenet_v3_large.features
        self.mobilenet_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.mobilenet_fc = nn.Linear(960, 960)  # MobileNetV3-Large tem 960 canais finais

        # Mecanismo de atenção
        self.attn = nn.Sequential(
            nn.Linear(960, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 960),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Linear(960, 1024),
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
        # DenseNet
        x1 = self.densenet_features(x)
        x1 = self.densenet_avgpool(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.densenet_fc(x1)

        # MobileNet
        x2 = self.mobilenet_features(x)
        x2 = self.mobilenet_avgpool(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.mobilenet_fc(x2)

        # Fusão com atenção
        alpha = self.attn(x1)
        x_combined = alpha * x1 + (1 - alpha) * x2

        return self.mlp(x_combined)


class CombinedVGG16MobileNetV3(nn.Module):
    def __init__(self, num_classes=4):
        super(CombinedVGG16MobileNetV3, self).__init__()

        # Backbone convolucional VGG16
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.vgg_features = vgg16.features
        self.vgg_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.vgg_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.vgg_fc = nn.Linear(512, 512)  # VGG16 tem 512 canais finais

        # Backbone convolucional MobileNetV3-Large
        mobilenet_v3_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.mobilenet_features = mobilenet_v3_large.features
        self.mobilenet_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.mobilenet_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.mobilenet_fc = nn.Linear(960, 512)  # MobileNetV3-Large tem 960 canais finais

        self.attn_block = LightSelfAttention(512)

        self.attn = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1024),
            nn.Sigmoid()
        )

    
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
        #x1_avg = self.vgg_avgpool(x1)
        x1 = self.vgg_maxpool(x1)
        #x1 = torch.cat([x1_avg, x1_max], dim=1)
        x1 = torch.flatten(x1, 1)
        #x1 = self.vgg_fc(x1)   # vetor jah eh (B, 512)

        # Passa pela MobileNetV3-Large
        x2 = self.mobilenet_features(x)
        #x2_avg = self.mobilenet_avgpool(x2)
        x2 = self.mobilenet_maxpool(x2)
        #x2 = torch.cat([x2_avg, x2_max], dim=1)
        x2 = torch.flatten(x2, 1)
        x2 = self.mobilenet_fc(x2)  # transofrmar para vetor (B, 512)



        # Produto elemento a elemento
        #x_combined = torch.mul(x1, x2)

        x1 = nn.functional.normalize(x1, dim=1)
        x2 = nn.functional.normalize(x2, dim=1)

        #alpha = self.attn(x1)
        #beta = self.attn(x2)
        #x_combined = alpha * x1 + beta * x2
        #x_combined = nn.functional.normalize(x_combined, dim=1)

        # Concat x1 e x2, e gerar pesos juntos

        fused = torch.concat([x1, x2], dim=1)
        weights = self.attn(fused)
        alpha, beta = torch.split(weights, 512, dim=1)
        x_combined = alpha * x1 + beta * x2

        # Self Attention em X1 e X2 -------------
        # alpha = self.attn(x1)
        # beta = self.attn(x2)
        # x_combined = alpha * x1 + beta * x2
        
        # Attention block em X1 -----------------
        #attended_x1 = self.attn_block(x2)
        #x_combined = (alpha * attended_x1) + (1.0 - alpha) * x2     
        
        output = self.mlp(x_combined)
        return output


class CombinedResNet18MobileNetV3(nn.Module):
    def __init__(self, num_classes=4):
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
            nn.Linear(512, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1256, num_classes)
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
    def __init__(self, model_name, fold, num_classes=4, device=None, patience=5):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_pretrained_model(model_name, num_classes)
        self.model.to(self.device)
        self.history = {'train_loss': [], 'val_accuracy': [], 'val_loss': []}
        self.early_stopping = EarlyStopping(patience=patience, fold=fold, verbose=True)
        self.models = ['vgg16', 'mobilenetv3', 'densenet161', 'inceptionv3', 'densenet121']

        
        if model_name in self.models:
            for layer in self.convolutional_layers:
                for param in layer.parameters():
                    if not isinstance(layer, nn.BatchNorm2d):
                        param.requires_grad = False

        elif model_name == 'vgg16_mobilenetv3':
            # VGG16
            for layer in self.model.vgg_features:
                for param in layer.parameters():
                    if not isinstance(layer, nn.BatchNorm2d):
                        param.requires_grad = False

            # MobileNetV3-Large
            for layer in self.model.mobilenet_features:
                for param in layer.parameters():
                    if not isinstance(layer, nn.BatchNorm2d):
                        param.requires_grad = False

        elif model_name == 'densenet161_mobilenetv3':
            # DenseNet161
            for layer in self.model.densenet_features:
                for param in layer.parameters():
                    if not isinstance(layer, nn.BatchNorm2d):
                        param.requires_grad = False

            # MobileNetV3-Large
            for layer in self.model.mobilenet_features:
                for param in layer.parameters():
                    if not isinstance(layer, nn.BatchNorm2d):
                        param.requires_grad = False
        

    def _load_pretrained_model(self, model_name, num_classes):
        if model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            self.print_model_info(model)

        elif model_name == 'mobilenetv3':
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            self.convolutional_layers = model.features  
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
            self.print_model_info(model)

        elif model_name == 'resnet18_mobilenetv3':
            self.model = CombinedResNet18MobileNetV3(num_classes=num_classes)
            model = self.model
            self.print_model_info(model)

        elif model_name == 'vgg16_mobilenetv3':
            self.model = CombinedVGG16MobileNetV3(num_classes=num_classes)
            model = self.model
            self.print_model_info(model)

        elif model_name == 'densenet161_mobilenetv3':
            self.model = CombinedDenseNet161MobileNetV3(num_classes=num_classes)
            model = self.model
            self.print_model_info(model)

        elif model_name == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.convolutional_layers = model.features 
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            self.print_model_info(model)

        elif model_name == 'densenet161':
            model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
            self.convolutional_layers = model.features  
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            self.print_model_info(model)

        elif model_name == 'inceptionv3':
            model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            self.convolutional_layers = nn.Sequential(*list(model.children())[:-1])
            self.print_model_info(model)
        
        elif model_name == 'densenet121':
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            self.convolutional_layers = model.features
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            self.print_model_info(model)

        else:
            raise ValueError(f"Modelo '{model_name}' não suportado.")
        return model
    
    def print_model_info(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2)

        print(f"📊 Total de parâmetros: {total_params:,}")
        print(f"🧠 Parâmetros treináveis: {trainable_params:,}")
        print(f"💾 Tamanho do modelo: {model_size_mb:.2f} MB")

    def train(self, trainloader, valloader, epochs=10, lr=0.001, fold=0):
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Início: somente os parâmetros com requires_grad=True (congelados parcialmente)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

        for epoch in range(epochs):
            if epoch == 5:
                print("Descongelando camadas convolucionais do modelo...")
                if hasattr(self, 'convolutional_layers'):
                    for layer in self.convolutional_layers:
                        for param in layer.parameters():
                            param.requires_grad = True
                
                elif hasattr(self.model, 'vgg_features') and hasattr(self.model, 'mobilenet_features'):
                    for layer in self.model.vgg_features:
                        for param in layer.parameters():
                            param.requires_grad = True
                    for layer in self.model.mobilenet_features:
                        for param in layer.parameters():
                            param.requires_grad = True
                    
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
            self.model.train()
            running_loss = 0.0
            print(f"\nFold {fold} | Epoch {epoch+1}/{epochs}")
            print("--------------------------------------------------------------------------------")

            for inputs, labels in tqdm(trainloader, desc=f"Treinando Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):  # lidar com InceptionV3
                    main_output, aux_output = outputs
                    loss = criterion(main_output, labels) + 0.4 * criterion(aux_output, labels)
                    outputs = main_output
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                

            avg_train_loss = running_loss / len(trainloader)
            val_loss, val_acc, _, _, _, _ = self.evaluate(valloader)
            print(f"Train Loss: {val_loss:.4f}  |  Val acc: {val_acc:4f}")

            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping acionado!")
                break

            self.history['train_loss'].append(avg_train_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['val_loss'].append(val_loss)
        
        self.plot_training_metrics(fold=fold, save_path=f"metrics_fold{fold}.png")

    

    def evaluate(self, dataloader, type='Validacao'):
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_outputs = []
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
                all_outputs.extend(outputs.softmax(dim=1).cpu().numpy())

        accuracy =  correct / total
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        avg_loss = running_loss / len(dataloader)

        try:
            auc = roc_auc_score(all_labels, all_outputs, multi_class='ovr', average='macro')
        except ValueError:
            auc = float('nan')

        return avg_loss, accuracy, precision, recall, f1, auc

    def predict(self, inputs):
        print("Iniciando Teste...")
        self.model.eval()
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
        return preds.cpu()
    
    def plot_training_metrics(self, fold=0, save_path='metrics_plot.png'):
        epochs = range(1, len(self.history['train_loss']) + 1)
        train_loss = self.history['train_loss']
        val_loss = self.history['val_loss']
        val_acc = self.history['val_accuracy']
        
        # Encontra a época com menor validação loss
        min_loss_epoch = val_loss.index(min(val_loss)) + 1

        plt.figure(figsize=(12, 6))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label='Loss de Treinamento')
        plt.plot(epochs, val_loss, label='Loss de Validação')
        plt.axvline(min_loss_epoch, color='gray', linestyle='--', label=f'Mínimo Loss (Época {min_loss_epoch})')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.title('Loss por Época')
        plt.legend()
        plt.grid(True)

        # Val
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_acc, label='Acurácia de Validação', color='green')
        plt.axvline(min_loss_epoch, color='gray', linestyle='--', label=f'Mínimo Loss (Época {min_loss_epoch})')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.title('Acurácia por Época')
        plt.legend()
        plt.grid(True)

        plt.suptitle(f'Métricas de Treinamento - Fold {fold}')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path)
        plt.close()
        print(f"📈 Gráfico salvo em: {save_path}")