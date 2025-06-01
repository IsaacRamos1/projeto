import numpy as np
import re

# Conteúdo do txt (cole como string multilinha)
texto = """
Fold 1:
Acurácia: 0.9356
Precisão: 0.8924
Recall: 0.9193
F1-Score: 0.9036
AUC: 0.9885
--------------------------------------------

Fold 2:
Acurácia: 0.9323
Precisão: 0.8810
Recall: 0.9129
F1-Score: 0.8939
AUC: 0.9884
--------------------------------------------

Fold 3:
Acurácia: 0.9356
Precisão: 0.8892
Recall: 0.9210
F1-Score: 0.9028
AUC: 0.9905
--------------------------------------------

Fold 4:
Acurácia: 0.9389
Precisão: 0.8896
Recall: 0.9164
F1-Score: 0.9001
AUC: 0.9909
--------------------------------------------

Fold 5:
Acurácia: 0.9389
Precisão: 0.8953
Recall: 0.9184
F1-Score: 0.9053
AUC: 0.9893
--------------------------------------------

Fold 6:
Acurácia: 0.9439
Precisão: 0.8974
Recall: 0.9210
F1-Score: 0.9073
AUC: 0.9893
--------------------------------------------

Fold 7:
Acurácia: 0.9406
Precisão: 0.8942
Recall: 0.9157
F1-Score: 0.9034
AUC: 0.9905
"""

# Expressões regulares para extrair os valores
padrao = {
    'Acurácia': r"Acurácia:\s+(\d+\.\d+)",
    'Precisão': r"Precisão:\s+(\d+\.\d+)",
    'Recall': r"Recall:\s+(\d+\.\d+)",
    'F1-Score': r"F1-Score:\s+(\d+\.\d+)",
    'AUC': r"AUC:\s+(\d+\.\d+)"
}

# Dicionário para armazenar os valores extraídos
valores = {chave: [] for chave in padrao}

# Extrair e armazenar os valores
for chave, regex in padrao.items():
    encontrados = re.findall(regex, texto)
    valores[chave] = [float(v) for v in encontrados]

# Calcular e imprimir os desvios padrão
print("Desvio padrão das métricas:")
for chave, lista in valores.items():
    desvio = np.std(lista)
    print(f"{chave}: {desvio:.4f}")
