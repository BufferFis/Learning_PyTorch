import torch
from Day1_autograd_and_tensors import SmallNet

# Hook to check that if the gradient vanishes or explodes
def grad_hook(name):
    def hook(grad):
        print(f"{name} grad mean: {grad.abs().mean():.6f}")
    return hook

x = torch.randn(10, 1, requires_grad=True)
print(x)
y = x ** 3

y.register_hook(grad_hook("y"))
loss = y.sum()
loss.backward

def track_gradient(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: mean = {param.grad.abs().mean():.6f}, max={param.grad.abs().max():.6f}")

torch.manual_seed(17062005)
model = SmallNet()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
loss_func = torch.nn.MSELoss()

X = torch.randn(100,1)
y_true = 3*X + 0.5

for epoch in range(5):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_func(y_pred, y_true)
    loss.backward()

    print(f"Epoch {epoch+1}, loss = {loss.item():.6f}")
    track_gradient(model)

    optimizer.step()