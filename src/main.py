import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulator import Continuous2DEnv, UnicycleDynamics, ModifiedUnicycleDynamics


import gymnasium as gym
import sys
import torch

from arguments import get_args
from ppo import PPO
from network import FeedForwardNN
from eval_policy import eval_policy
import tkinter as tk
from tkinter import filedialog


def select_model_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select Configuration File",
        initialdir="/home/bera/Desktop/SafeRL Codes/TL Guided RL with CBFs Implementation/models",
        filetypes=[("Model Files", "*.pth"), ("All Files", "*.*")])
    return file_path


def train(env, hyperparameters, actor_model, critic_model):
	"""
		Trains the model.

		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training

		Return:
			None
	"""	
	print(f"Training", flush=True)

	# Create a model for PPO.
	model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

	# Tries to load in an existing actor/critic model to continue training on
	if actor_model != '' and critic_model != '':
		print(f"Loading in {actor_model} and {critic_model}...", flush=True)
		model.actor.load_state_dict(torch.load(actor_model))
		model.critic.load_state_dict(torch.load(critic_model))
		print(f"Successfully loaded.", flush=True)
	elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
		print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
		sys.exit(0)
	else:
		print(f"Training from scratch.", flush=True)

	# Train the PPO model with a specified total timesteps
	# NOTE: You can change the total timesteps here, I put a big number just because
	# you can kill the process whenever you feel like PPO is converging
	model.learn(total_timesteps=50_000)

def test(env, actor_model):
	"""
		Tests the model.

		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in

		Return:
			None
	"""
	print(f"Testing {actor_model}", flush=True)

	# If the actor model is not specified, then exit
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Extract out dimensions of observation and action spaces
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]

	# Build our policy the same way we build our actor model in PPO
	policy = FeedForwardNN(obs_dim, act_dim)

	# Load in the actor model saved by the PPO algorithm
	policy.load_state_dict(torch.load(actor_model))

	# Evaluate our policy with a separate module, eval_policy, to demonstrate
	# that once we are done training the model/policy with ppo.py, we no longer need
	# ppo.py since it only contains the training algorithm. The model/policy itself exists
	# independently as a binary file that can be loaded in with torch.
	eval_policy(policy=policy, env=env)




if __name__ == "__main__":

	safe_region_center = [0.0, 0.0]
	safe_region_radius = 5


    #config dictionary for the environment
	config = {
        'init_loc': [0.0, 0.0], #initial location of the agent
        "width": 8.0,
        "height": 8.0,
        "dt": 0.1,
        "render": True,
        "goal_location": [3.0, 3.0],
        "goal_size": 0.5,
        "obstacle_location": [10.0, 10.0],
        "obstacle_size": 0.0,
        "safe_region_center": safe_region_center,
        "safe_region_radius": safe_region_radius,
        "randomize_loc": False #whether to randomize the agent location at the end of each episode
    }


	
	CBF_parameters = {
		"safe_region_center": safe_region_center,
        "safe_region_radius": safe_region_radius,
		'epsilon' : 0.5 #for reference point
    }


	#learning hyperparameters:
	hyperparameters = {
				'timesteps_per_batch': 500, 
				'max_timesteps_per_episode': 100, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip': 0.2,
				'render': False,
				'render_every_i': 1,
				'CBF': True,
				'CBF_params': CBF_parameters
			  }


	args = get_args()

	env = Continuous2DEnv(config)

	# Train or test, depending on the mode specified
	if args.mode == 'train':
		train(env=env, hyperparameters=hyperparameters, actor_model='', critic_model='')
	else:
		config['render'] = True #enable rendering for testing
		env = Continuous2DEnv(config)
		# Load in the model file
		model_path= select_model_file()
		test(env=env, actor_model=model_path)