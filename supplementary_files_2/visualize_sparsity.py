
import matplotlib.pyplot as plt

# 15524764, output < 1e-2
# 15524761, output < 1e-5
# 15524756, output < 1e-8
# all of the above had the same sparsity values for all epochs!!!
# also exactly the same as the ones with output == 0 (rounded to 3 decimal places)!

sparsity_values_1 = [
    (1,0.757),
    (2,0.744),
    (3,0.785),
    (4,0.816),
    (6,0.901),
    (8,0.950),
    (10,0.965)
]
