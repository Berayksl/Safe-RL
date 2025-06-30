#WITHOUT DIFFERENTIABLE SAFETY LAYER

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulator import Continuous2DEnv, UnicycleDynamics, ModifiedUnicycleDynamics

import gymnasium as gym
import sys
import torch
import os

from arguments import get_args
from SAC import SAC
#from eval_policy import eval_policy
import tkinter as tk
from tkinter import filedialog


def select_model_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

    file_path = filedialog.askopenfilename(
        title="Select Configuration File",
        initialdir=model_dir,
        filetypes=[("Model Files", "*.pth"), ("All Files", "*.*")])
    return file_path


def train(env,hyperparameters, actor_model, critic_model):
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


	model = SAC(env=env, **hyperparameters)

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

	model.train()

def test(env, ppo_model, action_range, actor_model, max_timesteps_per_episode):
	"""
		Tests the model.

		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in

		Return:
			None
	"""
	print(f"Testing {actor_model}", flush=True)

	#if the actor model is not specified, then exit
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	#extract out dimensions of observation and action spaces
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]

	if ppo_model == 'gaussian':
		# Build our policy the same way we build our actor model in PPO
		policy = FeedForwardNN(obs_dim, act_dim)

	elif ppo_model == 'beta':
		policy = BetaPolicyNetwork(obs_dim, act_dim)

	# Load in the actor model saved by the PPO algorithm
	policy.load_state_dict(torch.load(actor_model))

	eval_policy(policy=policy, ppo_model= ppo_model, action_range = action_range, env=env, max_timesteps_per_episode=max_timesteps_per_episode)



if __name__ == "__main__":

	target_region_center = [0.0, 3]
	target_region_radius = 0.5
	action_range = [5, 3] #max speed and max turning rate


    #config dictionary for the environment
	config = {
        'init_loc':[0.0, 0.0, 0.0], #initial location of the agent (x, y, theta)
        "width": 8.0,
        "height": 8.0,
        "dt": 0.1,
        "render": True,
		'dt_render': 0.01,
        "goal_location": [3.0, 3.0],
        "goal_size": 0.5,
        "obstacle_location": [10.0, 10.0],
        "obstacle_size": 0.0,
        "target_region_center": target_region_center,
        "target_region_radius": target_region_radius,
        "randomize_loc": False, #whether to randomize the agent location at the end of each episode
		'deterministic': False,
		'auto_entropy':True,
    }

	
	CBF_parameters = {
		"target_region_center": target_region_center,
        "target_region_radius": target_region_radius,
		'epsilon' : 0.5, #for reference point,
		'alpha': 5, #weight for the CBF term
    }


	#learning hyperparameters:
	hyperparameters = {
				'gamma': 0.99,
				'tau': 0.005,
				'hidden_size': 256, 
				'buffer_size': int(1e6),
				'batch_size': 300,
				'max_timesteps_per_episode': 200, 
				'num_episodes': 200,
				'n_updates_per_iteration': 1,
				'deterministic': False,
				'auto_entropy':True,
				'action_range': action_range, #max speed and max turning rate
				'action_clip' : True,
				'CBF': True,
				'CBF_params': CBF_parameters,
			  }


	args = get_args()

	env = Continuous2DEnv(config)

	# Train or test, depending on the mode specified
	if args.mode == 'train':
		train(env=env, hyperparameters=hyperparameters, actor_model='', critic_model='')

	elif args.mode == 'test':
		config['render'] = True #enable rendering for testing
		config['dt_render'] = 0.1
		config['init_loc'] = [0.0, 0.0, 0.0]
		env = Continuous2DEnv(config)
		ppo_model = 'gaussian'
		max_timesteps_per_episode = hyperparameters['max_timesteps_per_episode']
		# Load in the model file
		model_path= select_model_file()
		test(env=env, ppo_model = ppo_model, action_range= action_range, actor_model=model_path, max_timesteps_per_episode=max_timesteps_per_episode)