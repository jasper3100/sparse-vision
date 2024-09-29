import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Example data (replace with your actual data)
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 7, 11])

# Sort the data by x to ensure it's in increasing order
sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
y_sorted = y[sorted_indices]

# Create a spline interpolation
spline = UnivariateSpline(x_sorted, y_sorted, k=2)  # You can adjust the order (k) as needed

# Generate more points on the curve for smoother visualization
x_smooth = np.linspace(x.min(), x.max(), 1000)
y_smooth = spline(x_smooth)

# Plot the original points and the curve
plt.scatter(x, y, label='Original Points')
plt.plot(x_smooth, y_smooth, label='Spline Interpolation', color='red')
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Spline Interpolation of Points')
plt.show()