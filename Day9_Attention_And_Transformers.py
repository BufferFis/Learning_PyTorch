import torch
import torch.nn as nn
import torch.nn.functional as F

### DUMMY DATASET

posts = [
    "This movie was so good I cried",
    "Terrible acting, I hated it",
    "The plot was confusing but visuals were amazing",
    "One of the best experiences in cinema",
    "Not worth the time, super boring"
]

# Build a vocab

vocab = {}
for post in posts:
    for word in post.lower().split():
        if word not in vocab:
            vocab[word] = len(word)

vocab_size = len(vocab)
print("Vocab: ", vocab)

encoded = [[vocab[word] for word in post.lower().split()] for post in posts]

max_len = max(len(seq) for seq in encoded)

padded = [seq + [0]*(max_len - len(seq)) for seq in encoded]

inputs = torch.tensor(padded)
print("Input Shapes: ", inputs.shape)


class AttentionSummarizer(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.W_q = nn.Linear(embed_dim, hidden_dim)
        self.W_k = nn.Linear(embed_dim, hidden_dim)
        self.W_v = nn.Linear(embed_dim, hidden_dim)

    def forward(self, x):
        emb = self.embedding(x) # (batch, seq_len, embed_dim)

        Q = self.W_q(emb) # (batch, seq_len, hidden_dim)
        K = self.W_k(emb)
        V = self.W_v(emb)

        d_k = K.size(-1)

        # Attention scores

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1) # (batch, seq_len, seq_len)

        out = torch.matmul(attn_weights, V)

        # For summarization → take mean over seq_len
        summary = out.mean(dim=1)

        return summary, attn_weights
    

model = AttentionSummarizer(vocab_size, embed_dim=16, hidden_dim=16)

summaries, weights = model(inputs)

print("Summaries shape:", summaries.shape)   # (batch, hidden_dim)
print("Attention weights shape:", weights.shape)  # (batch, seq_len, seq_len)


print("\nPost:", posts[0])
print("Attention matrix for first post:\n", weights[0].detach())

