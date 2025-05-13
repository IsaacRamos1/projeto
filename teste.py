import torch
import torch.nn as nn

# Atenção 1: Atenção Simples (usando MLP de 1 camada)
class SimpleAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SimpleAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.attention(x)
        attended = x * weights
        return attended


# Atenção 2: Squeeze-and-Excitation Block (SE Block)
class SEBlock(nn.Module):
    def __init__(self, feature_dim, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(feature_dim, feature_dim // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(feature_dim // reduction, feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, feature_dim)
        se = x.mean(dim=0, keepdim=True)
        se = self.fc1(x)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        attended = x * se
        return attended


# Atenção 3: Self-Attention Leve (tipo Transformer Encoder simplificado)
class LightSelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(LightSelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.feature_dim = feature_dim

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.feature_dim ** 0.5)
        attention_probs = self.softmax(attention_scores)
        attended = torch.matmul(attention_probs, V)
        return attended
    
def select_top_features(x, top_k=8):
    values, indices = torch.topk(x, top_k, dim=1)
    return values


if __name__ == "__main__":
    batch_size = 1
    feature_dim = 12

    # Exemplo de vetor de características (batch_size, feature_dim)
    features = torch.randn(batch_size, feature_dim)
    print(features)
    print()

    # Atenção Simples
    simple_att = SimpleAttention(feature_dim)
    out_simple = simple_att(features)
    top_simple = select_top_features(out_simple)
    print("Simple Attention Output Shape:", out_simple.shape)
    print("Simple Attention Top 258 Features Shape:", top_simple.shape)
    print("Simple Attention Top 258 Features:\n", top_simple)

    # SE Block
    se_block = SEBlock(feature_dim)
    out_se = se_block(features)
    top_se = select_top_features(out_se)
    print("SE Block Output Shape:", out_se.shape)
    print("SE Block Top 258 Features Shape:", top_se.shape)
    print("SE Block Top 258 Features:\n", top_se)

    # Light Self-Attention
    self_attention = LightSelfAttention(feature_dim)
    out_self_att = self_attention(features)
    top_self_att = select_top_features(out_self_att)
    print("Self-Attention Output Shape:", out_self_att.shape)
    print("Self-Attention Top 258 Features Shape:", top_self_att.shape)
    print("Self-Attention Top 258 Features:\n", top_self_att)


    
