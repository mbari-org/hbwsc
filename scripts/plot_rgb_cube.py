import matplotlib.pyplot as plt
import numpy as np

# Create a uniform grid of points between 0 and 1
grid_size = 15
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 1, grid_size)
z = np.linspace(0, 1, grid_size)
X, Y, Z = np.meshgrid(x, y, z)

X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()

# Color is exactly the coordinate
colors = np.column_stack((X, Y, Z))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter points to form the cube
ax.scatter(X, Y, Z, c=colors, s=100, alpha=0.9, linewidths=0)

ax.set_title("RGB Color Cube Mapping")
ax.set_xlabel("X (Red Axis)")
ax.set_ylabel("Y (Green Axis)")
ax.set_zlabel("Z (Blue Axis)")

# Force equal aspect ratio so it looks like a perfect cube
ax.set_box_aspect([1, 1, 1])

fig.tight_layout()
print("Opening RGB color cube plot...")
plt.show()
