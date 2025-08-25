import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import yfinance as yf

import matplotlib.pyplot as plt


""" 
NOTE on using RNNs/LSTMs in PyTorch:

1. Input shape must always be (batch_size, seq_len, input_size).
   - batch_size = how many sequences you feed at once
   - seq_len    = how long each sequence is
   - input_size = how many features per time step (for stock price = 1)

2. If you forget the last "input_size" dimension (i.e. pass (batch, seq_len) 
   instead of (batch, seq_len, 1)), PyTorch will squash dimensions and LSTM 
   output will be 2D (batch, hidden_size) instead of 3D (batch, seq_len, hidden_size).
   → This causes indexing errors like `IndexError: too many indices for tensor`.

3. When predicting one new sequence, reshape input to (1, seq_len, input_size) 
   so it looks like a batch with size 1.

4. After LSTM, if you want the last time step output, use: out[:, -1, :] 
   (this only works if output is 3D).
"""


"""
NOTE on hidden_size and LSTM outputs:

1. hidden_size = how many "memory slots" the LSTM keeps at each time step.  
   - Think of it like the dimensionality of the "summary" it learns about 
     the past sequence.  
   - Bigger hidden_size = more capacity to learn complex patterns, but slower 
     and risk of overfitting.  
   - Example: hidden_size=64 → each time step is represented as a 64-dim vector.

2. LSTM returns TWO things:
   out, (h_n, c_n) = lstm(x)

   - out: shape = (batch, seq_len, hidden_size)  
       → hidden state for EVERY time step in the sequence.  
       → often you take `out[:, -1, :]` (last time step) if predicting the future.

   - h_n: shape = (num_layers * num_directions, batch, hidden_size)  
       → final hidden state (at the last time step, for each layer).  

   - c_n: shape = same as h_n  
       → final "cell state" (the LSTM’s internal long-term memory).

3. For most prediction tasks:  
   - Use `out[:, -1, :]` when you want the final hidden representation of the sequence.  
   - Use h_n (or c_n) if you specifically need the last hidden state of the LSTM layers.

Rule of thumb:
- hidden_size = "how much brain power per step"  
- out = "the whole memory tape"  
- h_n = "the last thought"  
- c_n = "the last long-term memory"
"""





data = yf.download("AAPL", start="2018-01-01", end="2023-01-01")["Close"]

plt.plot(data)
plt.title("MEHENGAYIIIII")
plt.show()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data.values.reshape(-1, 1))

def create_sequences(data, seq_len = 10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    
    return np.array(X), np.array(y)

seq_len = 10
X, y = create_sequences(scaled, seq_len)

X = X.reshape(X.shape[0], X.shape[1], 1)  # add feature dim
X = torch.tensor(X, dtype = torch.float32) # (samples, seq_len, 1)
y = torch.tensor(y, dtype=torch.float32) # (samples, 1)




class StockPredictor(nn.Module):
    def __init__(self, hidden_size = 50):
        super().__init__()
        self.lstm = nn.LSTM(input_size = 1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)
    
model = StockPredictor(hidden_size=64)
criterion = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr = 0.001)

epochs = 20
for epoch in range(epochs):
    opt.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    opt.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():6f}")


# Now we take the last seq

last_seq = torch.tensor(scaled[-seq_len:], dtype = torch.float32)

model.eval()

with torch.no_grad():
    next_scaled = model(last_seq).item()

next_price = scaler.inverse_transform([[next_scaled]])[0][0]
print("Predicted next closing price for Apple paisa", next_price)
