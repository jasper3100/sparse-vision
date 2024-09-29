# mixed3a 199 exp fac 8 lambda 5.0

# pixel-wise sparsity, this is the only place where I properly store this data!!!
pixel_sparsity = [(1,0.757),
        (2,0.728),
        (3,0.773),
        (4,0.808), 
        (5,0.838),
        (6,0.886),
        (7,0.921),
        (8,0.936), 
        (9,0.942),
        (10,0.947),
        (11,0.946), 
        (12,0.946),
        (13,0.937),
        (14,0.931),
        (15,0.921)]

# channel-wise sparsity
channel_sparsity = [(0, 7.99), (1, 0.76), (2, 0.74), (3, 0.78), (4, 0.82), (5, 0.85), (6, 0.9), (7, 0.93), (8, 0.95), (9, 0.96), (10, 0.96), (11, 0.97), (12, 0.97), (13, 0.97), (14, 0.97), (15, 0.97)]

# plot
import matplotlib.pyplot as plt

# plot both lines in one plot
plt.plot(*zip(*pixel_sparsity), label='Pixel-wise sparsity')
plt.plot(*zip(*channel_sparsity), label='Channel-wise sparsity')
plt.xlabel('Epochs')
plt.ylabel('Sparsity')
plt.legend()
plt.title('Sparsity over epochs')
plt.show()