import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Box bounds
L = 10.0

# Sphere properties
pos = np.array([5.0, 5.0, 5.0])
vel = np.array([0.2, 0.15, -0.25])
radius = 0.8

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim([0, L])
ax.set_ylim([0, L])
ax.set_zlim([0, L])

# Create sphere mesh
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sphere = radius * np.cos(u) * np.sin(v)
y_sphere = radius * np.sin(u) * np.sin(v)
z_sphere = radius * np.cos(v)

surf = [ax.plot_surface(x_sphere + pos[0], y_sphere + pos[1], z_sphere + pos[2], color='b')]

def update(frame):
    global pos, vel
    pos += vel
    
    # Check wall collisions
    for i in range(3):
        if pos[i] - radius < 0:
            pos[i] = radius
            vel[i] = -vel[i]
        elif pos[i] + radius > L:
            pos[i] = L - radius
            vel[i] = -vel[i]
            
    surf[0].remove()
    surf[0] = ax.plot_surface(x_sphere + pos[0], y_sphere + pos[1], z_sphere + pos[2], color='b')
    return surf

ani = FuncAnimation(fig, update, frames=200, interval=30, blit=False)
plt.show()