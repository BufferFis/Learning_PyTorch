import torch
import torch.nn as nn
import torch.nn.functional as F

class Stonks(nn.Module):
    def __init__(self, input_size = 5, hidden_size = 10, out_size = 1):
        super(Stonks, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size * 2)
        self.layer3 = nn.Linear(hidden_size*2, out_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        
    
    def forward(self, x):
        # I KNOW I CAN USE nn.Sequential, shush
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.sigmoid(x)

        return x

stock_net = Stonks(input_size = 5, hidden_size = 10, out_size = 1)

print("FULL ARCHITECTURE")
print(stock_net)

total_params = sum(p.numel() for p in stock_net.parameters())
print(f"\n Total params: {total_params}")

