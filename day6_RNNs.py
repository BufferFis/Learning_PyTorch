import torch
import torch.nn as nn

"""
Self Notes:
    Vocab size: how many unique tokens
    Hidden size: capacity to learn patterns

    dims:
        x = self.embed(x)     # (1,4,10)
        out, _ = self.lstm(x) # (1,4,20)
        out = self.fc(out)    # (1,4,5)
    
    Input: (1,4) → [0,1,2,3]

    Embedding: (1,4,10)
    LSTM: (1,4,20)
    Linear: (1,4,5)
    Squeeze: (4,5)
    Compare with target: [1,2,3,4]

"""


data = torch.tensor([0,1,2,3])
target = torch.tensor([1,2,3,4])

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 10)
        self.lstm = nn.LSTM(10, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x) # Out: (batch, seq, hidden)
        out = self.fc(out)
        return out
    

vocab = 5
model = SimpleLSTM(vocab, hidden_size=20)
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr = 0.01)

for epoch in range(100):
    opt.zero_grad()
    output = model(data)
    loss = criterion(output.squeeze(0), target)
    loss.backward()
    opt.step()
    print(f"Epoch {epoch+1} loss {loss.item():2f}")


print("Final loss:", loss.item())




