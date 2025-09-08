#WITHOUT DIFFERENTIABLE SAFETY LAYER 
#Compatible with the task scheduler (created on 9/5/2025)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulator import Continuous2DEnv
from task_schedule_py3 import task_scheduler

import sys
import random
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
	#########################################
	#SCENARIO-1:
	#########################################
	t_windows=[[[0,300],[0,100]],[[0,300],[0,100]]] # STL time windows
	subformula_types = np.array([4,4]) # 1: F, 2: G, 3: FG, 4: GF | Formula Types
	agent_init_loc = np.array([0,0]) # Initial pos. (x,y) of the agent
	roi = np.array([[-30, 30, 10],[-30, -30, 10]]) #3rd dimension is the radius

	u_tar_max = np.array([.5, .5]) #max velocities of the targets
	u_agent_max = 11 # Max vel. of the system
	disturbance_interval = [-1, 1]
	w_max = max(abs(disturbance_interval[0]), abs(disturbance_interval[1]))
	u_agent_max = u_agent_max - w_max #reduce the max agent speed by the disturbance bound (worst-case)

	target_movements = {0: {'type': 'circular', 'omega': 0.1, 'center_of_rotation': (-25, 30)}, 1: {'type': 'circular', 'omega': 0.1, 'center_of_rotation': (0, -35)}, 2: {'type': 'static'}, 3: {'type': 'static'}} #movement patterns for each target region
	#########################################
	#END OF SCENARIO-1
	#########################################

	roi_disj = [] #np.copy(roi) # Create alternative RoI's (to be modified)
	n_tar = len(roi) # of targets
	disj_map = np.array([np.arange(0,n_tar)]) 
	rois = [roi]

	target_colors = ['blue', 'red', 'green', 'black', 'yellow']

	sequence, rem_time, rem_time_realistic, gamma, portions, portions0 = task_scheduler(rois,t_windows,subformula_types,agent_init_loc,u_agent_max,u_tar_max)

	task_stypes = ["F", "G", "FG", "GF"]
    #create the target dictionary:
	targets = {}
	for i, target_id in enumerate(sequence):
		targets[i] = {
            'id': target_id,
            'type': task_stypes[subformula_types[target_id]-1],
            'time window': t_windows[target_id],
            'center': roi[target_id][:2],
            'radius': roi[target_id][2],
            'u_max': u_tar_max[target_id],
            'remaining_time': rem_time_realistic[i],
            'movement': target_movements[target_id],
            'color': target_colors[target_id]
        }


	#Static goal region:
	goal_region_radius = 10
	goals = {
	0: {'center': (50, 0), 'radius': goal_region_radius, 'movement':{'type':'static'}}, #goal region for the agent
	#1: {'center': (-50, 0), 'radius': 10, 'movement':{'type':'static'}}
	}

    #config dictionary for the environment
	config = {
        'init_loc':[0.0, -1], #initial location of the agent (x, y)
        "width": 100.0,
        "height": 100.0,
        "dt": 1,
        "render": False,
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
		"disturbance": disturbance_interval #disturbance range in both x and y directions [w_min, w_max]
    }


	
	CBF_parameters = {
		'epsilon' : 0.5, #for reference point,
		'alpha': 1.5, #weight for the CBF term
		"u_agent_max": u_agent_max, #max agent speed
		"targets": targets
    }

	action_range = [3, 3]

	#learning hyperparameters:
	hyperparameters = {
				'gamma': 0.99,
				'tau': 0.005,
				'hidden_size': 256, 
				'buffer_size': int(1e6),
				'batch_size': 256,
				'max_timesteps_per_episode': 300, 
				'num_episodes': 60,
				'n_updates_per_iteration': 1,
				'deterministic': False,
				'auto_entropy':True,
				'action_range': action_range,
				'action_clip' : True,
				'CBF': False,
				'CBF_params': CBF_parameters,
				'online': False,
				'disturbance': disturbance_interval
			  }


	args = get_args()

	env = Continuous2DEnv(config)

	# Train or test, depending on the mode specified
	if args.mode == 'train':
		train(env=env, hyperparameters=hyperparameters, actor_model='', critic_model='')

	elif args.mode == 'test':
		config['render'] = True #enable rendering for testing
		config['dt_render'] = 0.03
		config['init_loc'] = [0.0, -1]
		config['randomize_loc'] = False #randomize the agent location at the end of each episode
		env = Continuous2DEnv(config)
		max_timesteps_per_episode = hyperparameters['max_timesteps_per_episode']
		# Load in the model file
		model_path= select_model_file()
		test(env=env, hyperparameters=hyperparameters, model_path=model_path, max_timesteps_per_episode=max_timesteps_per_episode)