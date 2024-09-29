import torch

print("{:.3f}MB allocated".format(torch.cuda.memory_allocated()/1024**2))
# 0.0MB allocated

model = models.resnet18().cuda()
print("{:.3f}MB allocated".format(torch.cuda.memory_allocated()/1024**2))
# 44.690MB allocated

x = torch.randn(1, 3, 224, 224, device="cuda")
print("{:.3f}MB allocated".format(torch.cuda.memory_allocated()/1024**2))
# 45.265MB allocated

# no memory increase as running stats are empty
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
print("{:.3f}MB allocated".format(torch.cuda.memory_allocated()/1024**2))
# 45.265MB allocated
print(optimizer.state_dict())
# {'state': {}, 'param_groups': [{'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.01, 'amsgrad': False, 'foreach': None, 'maximize': False, 'capturable': False, 'differentiable': False, 'fused': None, 'params': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]}]}

# stores forward activations
out = model(x)
print("{:.3f}MB allocated".format(torch.cuda.memory_allocated()/1024**2))
# 74.011MB allocated

# calculates gradients so should increase by ~param size and delete forward activations
out.mean().backward()
print("{:.3f}MB allocated".format(torch.cuda.memory_allocated()/1024**2))
# 107.254MB allocated

# updates parameters and creates internal states
optimizer.step()
print("{:.3f}MB allocated".format(torch.cuda.memory_allocated()/1024**2))
# 198.693MB allocated
print(optimizer.state_dict())