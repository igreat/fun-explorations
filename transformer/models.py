import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyper Parameters
seq_len = 128
embedding_dim = 156
num_heads = 6
num_layers = 6

device = "mps"


class SelfAttention(nn.Module):
    """
    Masked Self Attention
    """

    def __init__(self, emb_dim, head_dim):
        super(SelfAttention, self).__init__()

        # query, key, value
        self.QKV = nn.Linear(emb_dim, head_dim * 3)

        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)))

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        q, k, v = self.QKV(x).chunk(3, dim=-1)

        # (batch_size, seq_len, seq_len)
        similarity_scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        # mask out future tokens to float('-inf')
        similarity_scores = similarity_scores.masked_fill(
            self.mask[:seq_len, :seq_len] == 0, float("-inf")
        )
        attention_weights = F.softmax(similarity_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = attention_weights @ v  # (batch_size, seq_len, value_dim)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(MultiHeadAttention, self).__init__()

        head_dim = emb_dim // num_heads
        self.attention_heads = nn.ModuleList(
            [SelfAttention(emb_dim, head_dim) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # (batch_size, seq_len, query_dim)
        # multi head self attention
        output = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        output = self.dropout(self.linear(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(emb_dim, num_heads)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
            nn.Dropout(0.2),
        )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super(GPT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(seq_len, embedding_dim)

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embedding_dim, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        T = x.shape[1]

        token_emb = self.token_embedding(x)  # (B, T, C)
        position_emb = self.position_embedding(torch.arange(T, device=device))  # (T, C)

        x = token_emb + position_emb
        x = self.transformer_blocks(x)
        x = self.norm(x)
        logits = self.linear(x)

        return logits
