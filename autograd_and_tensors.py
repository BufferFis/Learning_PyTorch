import torch

a = torch.tensor([[1., 2.], [3.,4.]])
print(a)
print(a.dtype, a.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

x = torch.ones(3,3)
y = torch.arange(3).view(1,3)
print(x)
print(y)
print(x + y) # Broadcasts y to match shape of x

x = torch.tensor([2.0, 3.0], requires_grad=True)

y = x ** 2 + 3 * x
z = y.sum()
z.backward() # Computes grad
print(x.grad) # dz / dx

class Cube(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input ** 3

    @staticmethod
    def backward(ctx, grad_output):
        (input, ) = ctx.saved_tensors
        return grad_output * 3 * input ** 2 # Grad output means gradient calculated by previous layers

x = torch.tensor([2.0], requires_grad=True)
y = Cube.apply(x)
y.backward()
print(x.grad)


# Day 1 Mini project

class CustomActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        sigmoid = 1 / (1 * torch.exp(-input))
        ctx.save_for_backward(input, sigmoid)
        return input * sigmoid
    
    @staticmethod
    def backward(ctx, grad_output):
        input, sigmoid = ctx.saved_tensors
        grad_input = grad_output + (sigmoid + input * sigmoid(1 - sigmoid))
        return grad_input
    

x = torch.tensors([2.0], requires_grad = True)
y = CustomActivation.apply(x)
y.backward()
print(f"x.grad(manual): {x.grad}")

class SmallNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.linear(1, 10)
        self.linear2 = torch.nn.linear(10, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = CustomActivation.apply(x)
        x = self.linear2(x)
        return x
    
model = SmallNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

torch.manual_seed(17062005)
X = torch.randn(100, 1)
y_true = 3 * X + 0.5

for epoch in range(20):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_func(y_pred, y_true)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}: loss = {loss.item():.4f}")


    

        
