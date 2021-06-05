import torch

x = torch.tensor(3., requires_grad=True)
y = x ** 2

grad = torch.autograd.grad(inputs=x,
                           outputs=y,
                           retain_graph=True)[0]
y.backward()
print(y)
print(x.grad)
print(grad)