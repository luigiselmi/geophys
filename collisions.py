# References: 
# 1) Mungan - Oblique elastic collisions of two smooth round objects 
# url: https://www.usna.edu/Users/physics/mungan/_files/documents/Publications/EJP28.pdf 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =============================================================================
# 1. INITIAL PARAMETERS OF THE SPHERES (Custom Radii, Masses, and Velocities)
# =============================================================================
# Sphere 1 (Small, light, and very fast)
r1 = 15.0
m1 = 2.0
p1_0 = np.array([80.0, 80.0])    # Initial position (x, y)
v1_0 = np.array([0.0, 0.0])   # Initial velocity (vx, vy)

# Sphere 2 (Large, heavy, slow, and moving vertically)
r2 = 25.0
m2 = 8.0
p2_0 = np.array([130.0, 10.0])   # Initial position (x, y)
v2_0 = np.array([-30.0, 90.0])   # Initial velocity (vx, vy)

e = 1.0  # Coefficient of restitution (1.0 = perfectly elastic)


# =============================================================================
# 2. VECTOR MATHEMATICS FUNCTION FOR 2D OBLIQUE IMPACT
# =============================================================================
def calculate_2d_impact(v1, m1, v2, m2, line_of_centers, e=1.0):
    distance = np.linalg.norm(line_of_centers)
    n = line_of_centers / distance  # Unit normal vector (along the line of centers)
    t = np.array([-n[1], n[0]])     # Unit tangent vector (rotated 90 degrees)

    # Project initial velocities onto the impact axes (dot product)
    v1n = np.dot(v1, n)
    v1t = np.dot(v1, t)
    v2n = np.dot(v2, n)
    v2t = np.dot(v2, t)

    # Overlap check: if normal relative velocity shows they are already moving apart, skip
    if (v1n - v2n) < 0:
        return v1, v2

    # Standard 1D elastic collision equations applied strictly to the normal axis
    v1n_final = (m1 * v1n + m2 * v2n + m2 * e * (v2n - v1n)) / (m1 + m2)
    v2n_final = (m2 * v2n + m1 * v1n + m1 * e * (v1n - v2n)) / (m1 + m2)

    # Tangential components 't' experience no force (frictionless assumption)
    v1t_final = v1t
    v2t_final = v2t

    # Reconstruct scalar components back into global Cartesian vectors (x, y)
    v1_post = v1n_final * n + v1t_final * t
    v2_post = v2n_final * n + v2t_final * t

    return v1_post, v2_post


# =============================================================================
# 3. SIMULATION LOOP (Position updates over time)
# =============================================================================
dt = 0.005       # Time step for each frame
num_steps = 1600  # Total number of frames in the animation

# Arrays to store the historical path coordinates for rendering the trails
pos1_history = np.zeros((num_steps, 2))
pos2_history = np.zeros((num_steps, 2))

p1 = p1_0.copy()
v1 = v1_0.copy()
p2 = p2_0.copy()
v2 = v2_0.copy()

already_collided = False  # Flag to prevent multiple checks in adjacent frames due to minor overlap

for i in range(num_steps):
    # Update positions based on current velocities
    p1 += v1 * dt
    p2 += v2 * dt

    # Geometric collision detection
    line_of_centers = p2 - p1
    distance = np.linalg.norm(line_of_centers)

    if distance <= (r1 + r2) and not already_collided:
        v1, v2 = calculate_2d_impact(v1, m1, v2, m2, line_of_centers, e)
        already_collided = True  # Prevents glitching if spheres slightly overlap next frame

    pos1_history[i] = p1
    pos2_history[i] = p2

# =============================================================================
# 4. PLOTTING AND ANIMATION SETUP (Matplotlib)
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, 180)
ax.set_ylim(0, 180)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_title("2D Oblique Sphere Collision Simulation (NumPy)", fontsize=12, fontweight='bold')

# Create the initial sphere patches to render on screen
circle1 = plt.Circle(pos1_history[0], r1, color='royalblue', alpha=0.8, label=f"Sphere 1 (m={m1})")
circle2 = plt.Circle(pos2_history[0], r2, color='crimson', alpha=0.8, label=f"Sphere 2 (m={m2})")
ax.add_patch(circle1)
ax.add_patch(circle2)

# Create line objects for drawing structural path trails
trail1, = ax.plot([], [], color='royalblue', linestyle=':', alpha=0.6)
trail2, = ax.plot([], [], color='crimson', linestyle=':', alpha=0.6)

ax.legend(loc='upper left')


# Update function executed at every single rendering frame
def update(frame):
    # Relocate the visual centers of the spheres
    circle1.set_center(pos1_history[frame])
    circle2.set_center(pos2_history[frame])

    # Extend the movement trails up to the current frame
    trail1.set_data(pos1_history[:frame, 0], pos1_history[:frame, 1])
    trail2.set_data(pos2_history[:frame, 0], pos2_history[:frame, 1])

    return circle1, circle2, trail1, trail2


# Generate the ongoing looped animation object
ani = animation.FuncAnimation(
    fig, update, frames=num_steps, interval=20, blit=True, repeat=True
)

plt.show()
