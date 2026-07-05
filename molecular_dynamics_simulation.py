import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Setup Simulation Parameters
num_particles = 1000
box_size = 10.0
dt = 0.05
steps = 500

# 2. Initialize Positions and Velocities randomly
positions = np.random.uniform(0, box_size, (num_particles, 2))
# Velocities scaled so particles move cleanly over the steps
velocities = np.random.uniform(-2.0, 2.0, (num_particles, 2)) 

# 3. Setup the Matplotlib figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
scatter = ax.scatter(positions[:, 0], positions[:, 1], s=50, color='blue')

# 4. Update function for the animation
def update(frame):
    global positions
    
    # Update positions based on velocity and time step
    positions += velocities * dt
    
    # Check for wall collisions and bounce (invert velocity component)
    mask_left = positions[:, 0] < 0
    mask_right = positions[:, 0] > box_size
    positions[mask_left, 0] = 0
    positions[mask_right, 0] = box_size
    velocities[mask_left | mask_right, 0] *= -1
    
    mask_bottom = positions[:, 1] < 0
    mask_top = positions[:, 1] > box_size
    positions[mask_bottom, 1] = 0
    positions[mask_top, 1] = box_size
    velocities[mask_bottom | mask_top, 1] *= -1
    
    # Update plot data
    scatter.set_offsets(positions)
    return scatter,

# 5. Run the animation
ani = animation.FuncAnimation(fig, update, frames=steps, interval=20, blit=True)
plt.show()
