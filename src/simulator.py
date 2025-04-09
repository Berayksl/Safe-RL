import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

from dynamics import UnicycleDynamics, ModifiedUnicycleDynamics

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
        self.goal_location = np.array(config.get("goal_location", [8.0, 8.0]))
        self.goal_size = config.get("goal_size", 0.5)
        self.obstacle_location = np.array(config.get("obstacle_location", [4.0, 4.0]))
        self.obstacle_size = config.get("obstacle_size", 0.5)
        self.random_loc = config.get("randomize_loc", True)
        self.safe_region_center = np.array(config.get("safe_region_center", [5.0, 5.0]))
        self.safe_region_radius = config.get("safe_region_radius", 3.0)
        self.init_loc = np.array(config.get("init_loc", [5.0, 5.0]))

        self.observation_space = np.zeros((3,))
        self.action_space = np.zeros((2,))
        
        self.reset()
        
        if self.render:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(-self.width, self.width)
            self.ax.set_ylim(-self.height, self.height)
            self.agent_plot, = self.ax.plot([], [], 'ro', label='Agent')
            self.goal_plot = plt.Circle(self.goal_location, self.goal_size, color='g', alpha=0.5, label='Goal')
            self.obstacle_plot = plt.Circle(self.obstacle_location, self.obstacle_size, color='r', alpha=0.5, label='Obstacle')
            self.safe_region_plot = plt.Circle(self.safe_region_center, self.safe_region_radius, color='b', fill=False, linestyle='dashed', label='Safe Region')
            self.ax.add_patch(self.safe_region_plot)
            self.ax.add_patch(self.goal_plot)
            self.ax.add_patch(self.obstacle_plot)
            self.ax.legend()
            self.ani = animation.FuncAnimation(self.fig, self.update_animation, interval=50)
    
    def reset(self):
        """Resets the environment."""
        if self.random_loc:
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
        else:
            x = self.init_loc[0]
            y = self.init_loc[1]
        self.agent = UnicycleDynamics(x=x, y=y, dt=self.dt)
        return np.array([self.agent.x, self.agent.y, self.agent.theta])
    
    def compute_reward(self):
        """Computes the dense reward based on distance to the goal."""
        dist_to_goal = np.linalg.norm(np.array([self.agent.x, self.agent.y]) - self.goal_location)
        #dist_to_obstacle = np.linalg.norm(np.array([self.agent.x, self.agent.y]) - self.obstacle_location)
        reward = -dist_to_goal
        # if dist_to_obstacle <= self.obstacle_size:
        #     reward -= 10  # Penalty for hitting an obstacle
        
        if dist_to_goal <= self.goal_size:
            reward += 100  # Reward for reaching the goal

        return reward 
    
    def step(self, action):
        """
        Takes an action (linear and angular velocity) and updates the agent's position.
        :param action: A tuple (v, w) representing linear and angular velocity.
        :return: next_state, reward, done
        """
        state = self.agent.update(action)
        
        self.agent.x = np.clip(self.agent.x, -self.width, self.width)
        self.agent.y = np.clip(self.agent.y, -self.height, self.height)
        
        # Compute reward
        reward = self.compute_reward()

        dist_to_goal = np.linalg.norm(np.array([self.agent.x, self.agent.y]) - self.goal_location)
        # Check if goal is reached
        done = dist_to_goal <= self.goal_size
        
        if self.render:
            plt.pause(0.01) #change later!!!
        
        return state, reward, done
    
    def update_animation(self, frame):
        x = np.array([self.agent.x])  # Convert scalar to NumPy array
        y = np.array([self.agent.y])  # Convert scalar to NumPy array
        self.agent_plot.set_data(x, y)
    
    def render_env(self):
        """Renders the environment continuously if enabled."""
        plt.show()

    



if __name__ == "__main__":


    #config dictionary for the environment
    config = {
        'init_loc': [0.0, 0.0], #initial location of the agent
        "width": 10.0,
        "height": 10.0,
        "dt": 0.1,
        "render": True,
        "goal_location": [8.0, 8.0],
        "goal_size": 0.5,
        "obstacle_location": [4.0, 4.0],
        "obstacle_size": 0.5,
        "safe_region_center": [0.0, 0.0],
        "safe_region_radius": 3.0,
        "randomize_loc": False #whether to randomize the agent location at the end of each episode
    }

    env = Continuous2DEnv(config)
    state = env.reset()

    action = (1.0, 1)  # Linear velocity = 1.0, Angular velocity = 0.1

    for _ in range(400):
        state, reward, done = env.step(action)

        print("State:", state)
        if done:
            break

