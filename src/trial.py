import roboticstoolbox as rtb
import numpy as np
from simulator import Continuous2DEnv, UnicycleDynamics, ModifiedUnicycleDynamics
import matplotlib.pyplot as plt

# Simulation parameters
dt = 0.1  # Time step
T = 10    # Total simulation time
steps = int(T / dt)

x0, y0, theta0 = 0.0, 0.0, 0.0
v = 1.0    # Forward velocity
w = 0.5    # Angular velocity
epsilon = 0.5  # Reference point offset f

unicycle_trajectory = np.zeros((steps, 3))
modified_trajectory = np.zeros((steps, 3))
tranformed_trajectory = np.zeros((steps, 3)) #transforms the trajectory of the reference point back to robot center


robot_1 = UnicycleDynamics(x=x0, y=y0, theta=theta0, v=0, w=0, dt=dt) #regular unicycle
robot_2 = ModifiedUnicycleDynamics(x=x0+epsilon*np.cos(theta0), y=y0+epsilon*np.sin(theta0), theta=theta0, v=0, w=0, dt=dt, epsilon=epsilon) #modified unicycle


# Simulate the unicycle model
for i in range(steps):
    unicycle_trajectory[i] = robot_1.update((v, w))
    modified_trajectory[i] = robot_2.update((v, w))
    tranformed_trajectory[i] = robot_2.get_robot_coordinates()
    


# Plot the trajectories
fig, ax = plt.subplots()   
ax.plot(unicycle_trajectory[:, 0], unicycle_trajectory[:, 1], label='Unicycle')
ax.plot(modified_trajectory[:, 0], modified_trajectory[:, 1], label='Modified Unicycle')
ax.plot(tranformed_trajectory[:,0], tranformed_trajectory[:,1], label='Transformed Unicycle')
ax.set_xlim(-5, 5)
ax.set_ylim(-2, 5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.show()