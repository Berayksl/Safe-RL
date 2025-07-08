#WITHOUT DIFFERENTIABLE SAFETY LAYER

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulator import Continuous2DEnv

import sys
import torch
import os

from arguments import get_args
from SAC import SAC
from evaluate_policy import eval_policy
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

def test(env, hyperparameters, model_path, max_timesteps_per_episode):
	"""
		Tests the model.

		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in

		Return:
			None
	"""
	print(f"Testing {model_path}", flush=True)

	#if the actor model is not specified, then exit
	if model_path == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	#extract out dimensions of observation and action spaces
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]

	model = SAC(env = env, **hyperparameters)

	# Load in the actor model saved by the PPO algorithm
	model.load_model(model_path)

	policy = model.policy_net

	eval_policy(policy=policy, env=env, max_timesteps_per_episode=max_timesteps_per_episode)



if __name__ == "__main__":

	target_region_center = [-10, 2]
	target_region_radius = 8
	action_range = [3, 3] #max vx and vy (for single integrator dynamics)

	u_target_max0 = 1.5
	u_target_max1 = 1.5
	u_agent_max = 8 #max agent speed
	
	# targets = {
	# 0: {'center': (-30, 30), 'radius': target_region_radius, 'u_max': u_target_max0, 'remaining_time': 100, 'movement':{'type': 'circular', 'omega': 0.1, 'center_of_rotation':(-25,30)}, 'color': 'blue'}, #heading angle is in rad
	# 1: {'center': (-30, -30), 'radius': target_region_radius, 'u_max': u_target_max1, 'remaining_time': 100, 'movement':{'type': 'circular', 'omega': -0.1, 'center_of_rotation':(-25,-30)}, 'color': 'red'}, #heading angle is in rad
	# #2: {'center': (-20, -20), 'radius': target_region_radius, 'u_max': 0.05	, 'remaining_time': 200, 'movement':{'type': 'straight', 'heading_angle': 5*np.pi/4}, 'color': 'green'}
    # }

	t1_p1 = (-30, 30)
	t1_p2 = (30, 30)
	t2_p1 = (30, -30)
	t2_p2 = (-30, -30)


	targets = {
	0: {'center': (-40, 40), 'radius': target_region_radius, 'u_max': u_target_max0, 'remaining_time': 100, 'movement':{'type': 'periodic', 'point1': t1_p1, 'point2': t1_p2, 'heading_angle': np.arctan2(t1_p2[1] - t1_p1[1], t1_p2[0] - t1_p1[0])}, 'color': 'blue'}, #heading angle is in rad
	1: {'center': (-40, -40), 'radius': target_region_radius, 'u_max': u_target_max1, 'remaining_time': 100, 'movement':{'type': 'periodic', 'point1': t2_p1, 'point2': t2_p2, 'heading_angle': np.arctan2(t2_p2[1] - t2_p1[1], t2_p2[0] - t2_p1[0])}, 'color': 'red'}, #heading angle is in rad
	#2: {'center': (-20, -20), 'radius': target_region_radius, 'u_max': 0.05	, 'remaining_time': 200, 'movement':{'type': 'straight', 'heading_angle': 5*np.pi/4}, 'color': 'green'}
    }


	goals = {
	0: {'center': (50, 0), 'radius': 10}, #goal region for the agent
	1: {'center': (-50, 0), 'radius': 10}
	}

    #config dictionary for the environment
	config = {
        'init_loc':[0.0, -1], #initial location of the agent (x, y)
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
    }


	
	CBF_parameters = {
		"target_region_center": target_region_center,
        "target_region_radius": target_region_radius,
		'epsilon' : 0.5, #for reference point,
		'alpha': 1.5, #weight for the CBF term
		"u_agent_max": u_agent_max, #max agent speed
		"targets": targets
    }


	#learning hyperparameters:
	hyperparameters = {
				'gamma': 0.99,
				'tau': 0.005,
				'hidden_size': 256, 
				'buffer_size': int(1e6),
				'batch_size': 256,
				'max_timesteps_per_episode': 300, 
				'num_episodes': 50,
				'n_updates_per_iteration': 1,
				'deterministic': False,
				'auto_entropy':True,
				'action_range': action_range,
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
		config['init_loc'] = [0.0, -1]
		env = Continuous2DEnv(config)
		max_timesteps_per_episode = hyperparameters['max_timesteps_per_episode']
		# Load in the model file
		model_path= select_model_file()
		test(env=env, hyperparameters=hyperparameters, model_path=model_path, max_timesteps_per_episode=max_timesteps_per_episode)