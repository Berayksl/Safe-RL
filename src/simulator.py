import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

from dynamics import UnicycleDynamics, SingleIntegratorDynamics
from Sequential_CBF import sequential_CBF

class Continuous2DEnv:
    def __init__(self, config):
        """
        Initializes the environment with given configuration.
        :param config: Dictionary containing environment parameters.
        """
        self.width = config.get("width", 10.0)
        self.height = config.get("height", 10.0)
        self.dt = config.get("dt", 0.1)
        self.render = config.get("render", False)
        self.dt_render = config.get("dt_render", 0.001)
        self.goal_location = np.array(config.get("goal_location", [8.0, 8.0]))
        self.goal_size = config.get("goal_size", 0.5)
        self.obstacle_location = np.array(config.get("obstacle_location", [4.0, 4.0]))
        self.obstacle_size = config.get("obstacle_size", 0.5)
        self.random_loc = config.get("randomize_loc", True)
        self.targets = config.get("targets", None)  # dictionary of targets for the CBF
        self.init_loc = np.array(config.get("init_loc", [5.0, 5.0]))
        self.action_max = config.get("action_max")
        self.action_min = config.get("action_min")
        self.dynamics = config.get("dynamics", "unicycle")
        self.u_agent_max = config.get("u_agent_max", None)  # max agent speed

        self.initial_target_centers = {target_index: self.targets[target_index]['center'] for target_index in self.targets.keys()}
        self.simulation_timer = 0


        if self.dynamics == "unicycle":
            self.observation_space = np.zeros((3,))
            self.action_space = np.zeros((2,))
            
        elif self.dynamics == "single integrator":
            self.observation_space = np.zeros((2,))
            self.action_space = np.zeros((2,))

        self.reset()
        #self.precompute_cbf_values(resolution=100)  # Precompute CBF values for visualization

        if self.render:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(-self.width, self.width)
            self.ax.set_ylim(-self.height, self.height)
            self.agent_plot, = self.ax.plot([], [], 'ro', label='Agent')
            self.goal_plot = plt.Circle(self.goal_location, self.goal_size, color='g', alpha=0.5, label='Goal')
            self.obstacle_plot = plt.Circle(self.obstacle_location, self.obstacle_size, color='r', alpha=0.5, label='Obstacle')
            #self.target_region_plot = plt.Circle(self.target_region_center, self.target_region_radius, color='b', fill=False, linestyle='-', label='Target Region')
            #self.ax.add_patch(self.target_region_plot)
            self.safe_region_plots = []
            self.ax.add_patch(self.goal_plot)
            self.ax.add_patch(self.obstacle_plot)
            self.ax.legend()
            self.ani = animation.FuncAnimation(self.fig, self.update_animation, interval=10, cache_frame_data=False)


    # def precompute_cbf_values(self, resolution=500):
    #     x_vals = np.linspace(-self.width, self.width, resolution)
    #     y_vals = np.linspace(-self.height, self.height, resolution)
    #     X, Y = np.meshgrid(x_vals, y_vals)

    #     self.CBF_cache = {}
    #     for remaining_t in range(self.task_period + 1):
    #         Z = np.zeros_like(X)
    #         for i in range(X.shape[0]):
    #             for j in range(X.shape[1]):
    #                 state = (X[i, j], Y[i, j])
    #                 Z[i, j] = CBF(state, remaining_t, self.u_agent_max, moving_target(self.task_period - remaining_t, self.target_region_center[0], self.target_region_center[1], self.omega), self.target_region_radius, self.u_target_max)
    #         self.CBF_cache[remaining_t] = Z

    #     self.CBF_grid = (X, Y)



    # def plot_cbf_zero_level(self, remaining_t):
    #     if hasattr(self, 'CBF_cache') and remaining_t in self.CBF_cache:
    #         X, Y = self.CBF_grid
    #         Z = self.CBF_cache[remaining_t]
    #         return X, Y, Z
    #     else:
    #         raise ValueError(f"CBF values for remaining_t = {remaining_t} not found in cache.")



    def cbf_zero_level(self, targets, u_agent_max, resolution=200):

        xlim = (-self.width, self.width)
        ylim = (-self.height, self.height)

        x_vals = np.linspace(xlim[0], xlim[1], resolution)
        y_vals = np.linspace(ylim[0], ylim[1], resolution)

        X, Y = np.meshgrid(x_vals, y_vals)

        cbf_levels = {}

        for target_index in targets.keys():
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    state = (X[i, j], Y[i, j])
                    Z[i, j] = sequential_CBF(state, u_agent_max, targets, target_index)

            cbf_levels[target_index] = (X,Y,Z)

        return cbf_levels
    
    
    def reset(self):
        """Resets the environment."""
        if self.random_loc:
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            theta = self.init_loc[2] + np.random.uniform(-np.pi, np.pi)
            # Ensure the agent starts within the safe region
            while np.linalg.norm(np.array([x, y]) - self.safe_region_center) >= self.safe_region_radius:
                x = np.random.uniform(0, self.width)
                y = np.random.uniform(0, self.height)
                theta = self.init_loc[2] + np.random.uniform(-np.pi, np.pi)
                # Ensure the agent starts within the safe region
                if np.linalg.norm(np.array([x, y]) - self.safe_region_center) < self.safe_region_radius:
                    break
        else:
            if self.dynamics == 'unicycle':
                x = self.init_loc[0]
                y = self.init_loc[1]
                theta = self.init_loc[2]
            elif self.dynamics == 'single integrator':
                x = self.init_loc[0]
                y = self.init_loc[1]

        if self.dynamics == "unicycle":
            self.agent = UnicycleDynamics(x=x, y=y, theta = theta, dt=self.dt)
            state = np.array([self.agent.x, self.agent.y, self.agent.theta])
        elif self.dynamics == "single integrator":
            self.agent = SingleIntegratorDynamics(x=x, y=y, dt=self.dt)
            state = np.array([self.agent.x, self.agent.y])
            #print("Single Integrator Dynamics")

        return state
    
    def compute_reward(self):
        """Computes the dense reward based on distance to the goal."""
        dist_to_goal = np.linalg.norm(np.array([self.agent.x, self.agent.y]) - self.goal_location)
        #dist_to_obstacle = np.linalg.norm(np.array([self.agent.x, self.agent.y]) - self.obstacle_location)
        reward = -dist_to_goal
        # if dist_to_obstacle <= self.obstacle_size:
        #     reward -= 10  # Penalty for hitting an obstacle
        
        if dist_to_goal <= self.goal_size:
            reward += 100  # Reward for reaching the goal
            #print("Goal Reached!")

        return reward 
    
    def step(self, action):
        """
        Takes an action (linear and angular velocity) and updates the agent's position.
        :param action: A tuple (v, w) representing linear and angular velocity.
        :return: next_state, reward, done
        """
        #make the updates:
        state = self.agent.update(action) #update the agent state
        #update the target region states:
        for target_index in self.targets.keys():
            x_new, y_new = self.dynamic_target(self.simulation_timer, target_index)
            self.targets[target_index]['center'] = (x_new, y_new)

        self.simulation_timer += 1

        self.agent.x = np.clip(self.agent.x, -self.width, self.width)
        self.agent.y = np.clip(self.agent.y, -self.height, self.height)
        
        # Compute reward
        reward = self.compute_reward()

        dist_to_goal = np.linalg.norm(np.array([self.agent.x, self.agent.y]) - self.goal_location)
        # Check if goal is reached
        done = dist_to_goal <= self.goal_size


        #remove previous safe region plots:
        for contour in self.safe_region_plots:
            for coll in contour.collections:
                try:
                    coll.remove()
                except ValueError:
                    pass

        self.safe_region_plots = []


        cbf_levels = self.cbf_zero_level(self.targets, self.u_agent_max, resolution=100)

        for target_index, (X, Y, Z) in cbf_levels.items():
            contour = self.ax.contour(X, Y, Z, levels=[0], colors=self.targets[target_index]['color'], linewidths=1, linestyles='dashed', label='CBF Zero Level')
            self.safe_region_plots.append(contour)


        if self.render:
            plt.pause(self.dt_render) #change later!!!
        
        return state, reward, done
    
    
    def update_animation(self, frame):
        if hasattr(self, 'safe_region_plot'):
            for coll in self.safe_region_plot.collections:
                try:
                    coll.remove()
                except ValueError:
                    pass  # It's already removed

        if hasattr(self, 'target_region_plots'):
            for patch in self.target_region_plots:
                if patch in self.ax.patches:
                    patch.remove()

        self.target_region_plots = []

        if self.targets:
            for target_id, target_info in self.targets.items():
                center = target_info["center"]
                radius = target_info["radius"]
                patch = plt.Circle(center, radius, color=target_info['color'], fill=False, linestyle='-', label='Target Region' if frame == 0 else "")
                self.ax.add_patch(patch)
                self.target_region_plots.append(patch)


        # Update the agent's position in the plot
        x = np.array([self.agent.x])  # Convert scalar to NumPy array
        y = np.array([self.agent.y])  # Convert scalar to NumPy array

        self.agent_plot.set_data(x, y)
        self.ax.legend()

    def dynamic_target(self, current_t, target_id):
        if self.targets is not None:
            movement_type = self.targets[target_id]['movement']['type']
            #TODO: add more movement types
            if movement_type == 'circular':
                x0, y0 = self.initial_target_centers[target_id]
                u_target_max = self.targets[target_id]['u_max']
                omega = self.targets[target_id]['movement']['omega']
                xc, yc = self.targets[target_id]['movement']['center_of_rotation']
                 # Calculate the initial angle from the center of rotation to the initial position
                theta0 = np.arctan2(y0 - yc, x0 - xc)
                
                # Calculate the new angle after time_elapsed
                theta = theta0 + omega * current_t
                turning_radius = u_target_max / omega

                x_new = xc + turning_radius * np.cos(theta)
                y_new = yc + turning_radius * np.sin(theta)
            
            elif movement_type == 'straight':
                x0, y0 = self.initial_target_centers[target_id]
                heading_angle = self.targets[target_id]['movement']['heading_angle']
                u_target_max = self.targets[target_id]['u_max']

                x_new = x0 + np.cos(heading_angle) * u_target_max * current_t
                y_new = x0 + np.sin(heading_angle) * u_target_max * current_t

            elif movement_type == "static":
                #no movement, just return the current position
                x_new, y_new = self.targets[target_id]['center']

            elif movement_type == "periodic":
                # New movement type: periodic back-and-forth between two points
                point1 = np.array(self.targets[target_id]['movement']['point1'])
                point2 = np.array(self.targets[target_id]['movement']['point2'])
                u_target_max = self.targets[target_id]['u_max']

                total_distance = np.linalg.norm(point2 - point1)
                total_time = total_distance / u_target_max

                # Determine current phase within the full cycle (forth + back)
                cycle_time = 2 * total_time
                t_mod = current_t % cycle_time

                if t_mod < total_time:
                    # Moving from point1 to point2
                    alpha = t_mod / total_time
                    position = point1 + alpha * (point2 - point1)
                else:
                    # Moving back from point2 to point1
                    alpha = (t_mod - total_time) / total_time
                    position = point2 - alpha * (point2 - point1)

                x_new, y_new = position[0], position[1]


        return (x_new, y_new)

               


    def render_env(self):
        """Renders the environment continuously if enabled."""
        plt.show()

    


if __name__ == "__main__":
    #config dictionary for the environment
    config = {
        'init_loc': [0.0, 0.0, 0.0], #initial location of the agent
        "width": 10.0,
        "height": 10.0,
        "dt": 0.1,
        "render": True,
        'dt_render': 0.1,
        "goal_location": [8.0, 8.0],
        "goal_size": 0.5,
        "obstacle_location": [4.0, 4.0],
        "obstacle_size": 0.5,
        "safe_region_center": [0.0, 0.0],
        "safe_region_radius": 3.0,
        "randomize_loc": True #whether to randomize the agent location at the end of each episode
    }

    env = Continuous2DEnv(config)
    state = env.reset()

    action = (1.0, 1)  #linear vel., angular vel.

    for _ in range(100):
        state, reward, done = env.step(action)

        #print("State:", state)
        if done:
            break

        #state = env.reset()

