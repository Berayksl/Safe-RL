import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

from dynamics import UnicycleDynamics, SingleIntegratorDynamics
#from Sequential_CBF import sequential_CBF

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
        self.goals = config.get("goals", None)  # dictionary of goals for the agent
        self.obstacle_location = np.array(config.get("obstacle_location", [4.0, 4.0]))
        self.obstacle_size = config.get("obstacle_size", 0.5)
        self.random_loc = config.get("randomize_loc", True)
        self.targets = config.get("targets", None)  # dictionary of targets for the CBF
        self.init_loc = np.array(config.get("init_loc", [5.0, 5.0]))
        self.action_max = config.get("action_max")
        self.action_min = config.get("action_min")
        self.dynamics = config.get("dynamics", "unicycle")
        self.u_agent_max = config.get("u_agent_max", None)  # max agent speed

        if config.get("disturbance") is not None: #min and max disturbance:
            self.disturbance = True
            self.w_min = config.get("disturbance")[0] 
            self.w_max = config.get("disturbance")[1]
        else:
            self.disturbance = False

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

            self.goal_plots = []
            for goal in self.goals.values():
                goal_plot = plt.Circle(goal['center'], goal['radius'], color='g', alpha=0.5, label='Goal')
                self.goal_plots.append(goal_plot)
                self.ax.add_patch(goal_plot)

            self.obstacle_plot = plt.Circle(self.obstacle_location, self.obstacle_size, color='r', alpha=0.5, label='Obstacle')
            #self.target_region_plot = plt.Circle(self.target_region_center, self.target_region_radius, color='b', fill=False, linestyle='-', label='Target Region')
            #self.ax.add_patch(self.target_region_plot)
            self.safe_region_plots = []
            self.target_region_patches = []
            self.ax.add_patch(self.obstacle_plot)
            self.ax.legend()
            #self.ani = animation.FuncAnimation(self.fig, self.update_animation, interval=10, cache_frame_data=False)
            self.fig.show()
            self.fig.canvas.draw()






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
            x = np.random.uniform(-self.width, self.width)
            y = np.random.uniform(-self.height, self.height)
            if self.dynamics == 'unicycle':
                theta = self.init_loc[2] + np.random.uniform(-np.pi, np.pi)
                self.agent = UnicycleDynamics(x=x, y=y, theta=theta, dt=self.dt)
                state = np.array([self.agent.x, self.agent.y, self.agent.theta])
            elif self.dynamics == 'single integrator':
                self.agent = SingleIntegratorDynamics(x=x, y=y, dt=self.dt)
                state = np.array([self.agent.x, self.agent.y])
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
    
    # def compute_reward(self):
    #     """Computes the dense reward based on distance to the goal."""
    #     dist_to_goal = np.linalg.norm(np.array([self.agent.x, self.agent.y]) - self.goal_location)

    #     reward = -dist_to_goal
    #     # if dist_to_obstacle <= self.obstacle_size:
    #     #     reward -= 10  # Penalty for hitting an obstacle
        
    #     if dist_to_goal <= self.goal_size:
    #         reward += 100  # Reward for reaching the goal
    #         #print("Goal Reached!")

    #     return reward
    


    def compute_reward(self): #updated to include multiple goals
        pos = np.array([self.agent.x, self.agent.y])
        # List of distances to all goal centers
        dists = [np.linalg.norm(pos - np.array(g['center'])) for g in self.goals.values()]
        dist_min = min(dists)
        
        reward = -dist_min  # Dense part: encourages reaching nearest goal
        
        # Sparse bonus if agent enters any goal region
        for goal in self.goals.values():
            if dist_min <= goal['radius']:
                reward += 100
                break
        
        return reward
    
    def step(self, action):
        """
        Takes an action (linear and angular velocity) and updates the agent's position.
        :param action: A tuple (v, w) representing linear and angular velocity.
        :return: next_state, reward, done
        """
        #make the updates:
        if self.disturbance:
            w = np.random.uniform(self.w_min, self.w_max, size=2) #sample noise
            action = action + w #add noise to the action
            #print(action)

        state = self.agent.update(action) #update the agent state
        #update the target region states:
        for target_index in self.targets.keys():
            x_new, y_new = self.dynamic_target(self.simulation_timer, target_index)
            self.targets[target_index]['center'] = (x_new, y_new)

        #update the goal states:
        for goal_id in self.goals.keys():
            x_new, y_new = self.dynamic_goal(self.simulation_timer, goal_id)
            self.goals[goal_id]['center'] = (x_new, y_new)

        self.simulation_timer += 1

        self.agent.x = np.clip(self.agent.x, -self.width, self.width)
        self.agent.y = np.clip(self.agent.y, -self.height, self.height)
        
        # Compute reward
        reward = self.compute_reward()

        # Calculate distance to the goals
        dist_to_goals = {goal_id: np.linalg.norm(np.array([self.agent.x, self.agent.y]) - np.array(goal['center'])) for goal_id, goal in self.goals.items()}
        # Check if any goal is reached
        done = any(dist <= goal['radius'] for dist, goal in zip(dist_to_goals.values(), self.goals.values()))

        #TO PLOT THE 0 LEVEL CONTOURS:

        #remove previous safe region plots:
        # for contour in self.safe_region_plots:
        #     for coll in contour.collections:
        #         try:
        #             coll.remove()
        #         except ValueError:
        #             pass

        # self.safe_region_plots = []


        # cbf_levels = self.cbf_zero_level(self.targets, self.u_agent_max, resolution=100)

        # for target_index, (X, Y, Z) in cbf_levels.items():
        #     contour = self.ax.contour(X, Y, Z, levels=[0], colors=self.targets[target_index]['color'], linewidths=1, linestyles='dashed', label='CBF Zero Level')
        #     self.safe_region_plots.append(contour)


        if self.render:
            self.update_plot()
                
        return state, reward, done
    
    
    def update_plot(self):
        time.sleep(self.dt_render)
        # Update agent position
        self.agent_plot.set_data([self.agent.x], [self.agent.y])
        
        # Update targets' positions
        for patch in self.target_region_patches:
            patch.remove()  # remove old
            
        for label in getattr(self, "target_labels", []):
            label.remove()

        self.target_region_patches = []
        self.target_labels = []
        for i, target_info in self.targets.items():
            patch = plt.Circle(target_info["center"], target_info["radius"], color=target_info['color'], fill=False, linestyle='-')
            self.ax.add_patch(patch)
            self.target_region_patches.append(patch)

            cx, cy = target_info["center"]
            target_id = target_info['id']
            label = self.ax.text(
                cx, cy, str(target_id),
                ha='center', va='center',
                fontsize=10, color=target_info['color'],
                weight='bold'
            )
            self.target_labels.append(label)


        # Update goals' positions
        for goal_plot in self.goal_plots:
            goal_plot.remove()

        self.goal_plots = []
        for goal_id, goal_info in self.goals.items():
            patch = plt.Circle(goal_info["center"], goal_info["radius"], color='g', alpha=0.5, label='Goal')
            self.ax.add_patch(patch)
            self.goal_plots.append(patch)

        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # def update_animation(self, frame):
    #     if hasattr(self, 'safe_region_plot'):
    #         for coll in self.safe_region_plot.collections:
    #             try:
    #                 coll.remove()
    #             except ValueError:
    #                 pass  # It's already removed

    #     if hasattr(self, 'target_region_plots'):
    #         for patch in self.target_region_plots:
    #             if patch in self.ax.patches:
    #                 patch.remove()

    #     self.target_region_plots = []

    #     if self.targets:
    #         for target_id, target_info in self.targets.items():
    #             center = target_info["center"]
    #             radius = target_info["radius"]
    #             patch = plt.Circle(center, radius, color=target_info['color'], fill=False, linestyle='-', label='Target Region' if frame == 0 else "")
    #             self.ax.add_patch(patch)
    #             self.target_region_plots.append(patch)


    #     # Update the agent's position in the plot
    #     x = np.array([self.agent.x])  # Convert scalar to NumPy array
    #     y = np.array([self.agent.y])  # Convert scalar to NumPy array

    #     self.agent_plot.set_data(x, y)
    #     self.ax.legend()

    def dynamic_target(self, current_t, target_id):
        if self.targets is not None:
            movement_type = self.targets[target_id]['movement']['type']
            #TODO: add more movement types
            if movement_type == 'circular':
                x0, y0 = self.initial_target_centers[target_id]
                u_target_max = self.targets[target_id]['u_max']
                #omega = self.targets[target_id]['movement']['omega']
                xc, yc = self.targets[target_id]['movement']['center_of_rotation']
                 # Calculate the initial angle from the center of rotation to the initial position
                theta0 = np.arctan2(y0 - yc, x0 - xc)

                turning_radius = np.linalg.norm(np.array([x0 - xc, y0 - yc]))
                omega = u_target_max / turning_radius #angular velocity
                
                # Calculate the new angle after time_elapsed
                theta = theta0 + omega * current_t
                turning_radius = u_target_max / omega

                x_new = xc + turning_radius * np.cos(theta)
                y_new = yc + turning_radius * np.sin(theta)
            # if movement_type == 'circular':
            #     x0, y0 = self.initial_target_centers[target_id]
            #     u_target_max = self.targets[target_id]['u_max']
            #     omega = self.targets[target_id]['movement']['omega']
               
            #      # Calculate the initial angle from the center of rotation to the initial position

            #     turning_radius = u_target_max / omega

            #     # Calculate the new angle after time_elapsed
            #     theta = theta0 + omega * current_t
            #     turning_radius = u_target_max / omega

            #     x_new = xc + turning_radius * np.cos(theta)
            #     y_new = yc + turning_radius * np.sin(theta)
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
                    direction_vector = point2 - point1
                else:
                    # Moving back from point2 to point1
                    alpha = (t_mod - total_time) / total_time
                    position = point2 - alpha * (point2 - point1)
                    direction_vector = point1 - point2

                x_new, y_new = position[0], position[1]

                heading_angle = np.arctan2(direction_vector[1], direction_vector[0])


                # Update heading angle in targets dictionary
                self.targets[target_id]['movement']['heading_angle'] = heading_angle


        return (x_new, y_new)

    def dynamic_goal(self, current_t, goal_id):
        if self.goals is not None:
            movement_type = self.goals[goal_id]['movement']['type']
            if movement_type == 'static':
                # No movement, just return the current position
                x_new, y_new = self.goals[goal_id]['center']
            elif movement_type == 'periodic':
                # New movement type: periodic back-and-forth between two points
                point1 = np.array(self.goals[goal_id]['movement']['point1'])
                point2 = np.array(self.goals[goal_id]['movement']['point2'])
                u_goal_max = self.goals[goal_id]['u_max']

                total_distance = np.linalg.norm(point2 - point1)
                total_time = total_distance / u_goal_max

                # Determine current phase within the full cycle (forth + back)
                cycle_time = 2 * total_time
                t_mod = current_t % cycle_time

                if t_mod < total_time:
                    # Moving from point1 to point2
                    alpha = t_mod / total_time
                    position = point1 + alpha * (point2 - point1)
                    direction_vector = point2 - point1
                else:
                    # Moving back from point2 to point1
                    alpha = (t_mod - total_time) / total_time
                    position = point2 - alpha * (point2 - point1)
                    direction_vector = point1 - point2

                x_new, y_new = position[0], position[1]

                heading_angle = np.arctan2(direction_vector[1], direction_vector[0])

                # Update heading angle in goals dictionary
                self.goals[goal_id]['movement']['heading_angle'] = heading_angle

            elif movement_type == 'blinking':
                # Blinking movement: switch between two points every blink_duration
                point1 = np.array(self.goals[goal_id]['movement']['point1'])
                point2 = np.array(self.goals[goal_id]['movement']['point2'])
                blink_duration = self.goals[goal_id]['movement']['blink_duration']

                # Determine if we are in the first or second half of the blink cycle
                if (current_t // blink_duration) % 2 == 0:
                    x_new, y_new = point1[0], point1[1]
                else:
                    x_new, y_new = point2[0], point2[1]
                
            else:
                raise NotImplementedError(f"Movement type '{movement_type}' for goal not implemented.")
        else:
            raise ValueError("No goals defined in the environment.")

        return (x_new, y_new)
    

    def set_agent_location(self, x, y, theta=None):
        """
        Sets the agent's location to the specified coordinates manually.
        :param x: x-coordinate of the agent.
        :param y: y-coordinate of the agent.
        :param theta: orientation angle (only for unicycle dynamics).
        """
        if self.dynamics == 'unicycle':
            self.agent = UnicycleDynamics(x=x, y=y, theta=theta, dt=self.dt)
            return np.array([self.agent.x, self.agent.y, self.agent.theta])
        elif self.dynamics == 'single integrator':
            self.agent = SingleIntegratorDynamics(x=x, y=y, dt=self.dt)
            return np.array([self.agent.x, self.agent.y])
        
if __name__ == "__main__":

    target_region_center = [-10, 2]
    target_region_radius = 8
    action_range = [3, 3] #max vx and vy (for single integrator dynamics)

    u_target_max0 = 1.5
    u_target_max1 = 1.5
    u_agent_max = 8 #max agent speed

    goals = {0: {'center': (50, 0), 'radius': 10, 'movement':{'type':'static'}}, #goal region for the agent
	#1: {'center': (-50, 0), 'radius': 10, 'movement':{'type':'static'}}
    }

    targets = {}
    #config dictionary for the environment
    config = {
        'init_loc':[-20, -20], #initial location of the agent (x, y)
        "width": 100.0,
        "height": 100.0,
        "dt": 1,
        "render": True,
		'dt_render': 0.03,
		'goals': goals, #goal regions for the agent
        "obstacle_location": [100.0, 100.0],
        "obstacle_size": 0.0,
        "randomize_loc": False, #whether to randomize the agent location at the end of each episode
		'deterministic': False,
		'auto_entropy':True,
		"dynamics": "single integrator", #dynamics model to use
		"targets": targets,
		"u_agent_max": u_agent_max, #max target speed
		"disturbance": [-0.5, 0.5] #disturbance range in both x and y directions [w_min, w_max]
    }

    env = Continuous2DEnv(config)
    state = env.reset()

    action = (1.0, 1)  #linear vel., angular vel.

    for _ in range(100):
        state, reward, done = env.step(action)

        #print("State:", state)
        # if done:
        #     break

        #state = env.reset()

