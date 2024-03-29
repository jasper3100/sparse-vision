import torch
x = torch.tensor([1, 2, 3, 4, -9, -0.0003])
print(torch.topk(x, 2, largest=False))