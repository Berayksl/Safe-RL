#WITHOUT DIFFERENTIABLE SAFETY LAYER 
#Compatible with the task scheduler (created on 9/5/2025)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulator import Continuous2DEnv
from task_schedule_py3 import task_scheduler
from datetime import datetime

import sys
import random
import torch
import os
import copy

from arguments import get_args
from SAC import SAC
from evaluate_policy import eval_policy
import tkinter as tk
from tkinter import filedialog

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="cvxpy")

def select_model_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

    file_path = filedialog.askopenfilename(
        title="Select Configuration File",
        initialdir=model_dir,
        filetypes=[("Model Files", "*.pth"), ("All Files", "*.*")])
    return file_path


# def train(env,hyperparameters, actor_model, critic_model):
# 	"""
# 		Trains the model.

# 		Parameters:
# 			env - the environment to train on
# 			hyperparameters - a dict of hyperparameters to use, defined in main
# 			actor_model - the actor model to load in if we want to continue training
# 			critic_model - the critic model to load in if we want to continue training

# 		Return:
# 			None
# 	"""	
# 	print(f"Training", flush=True)


# 	model = SAC(env=env, **hyperparameters)

# 	# Tries to load in an existing actor/critic model to continue training on
# 	if actor_model != '' and critic_model != '':
# 		print(f"Loading in {actor_model} and {critic_model}...", flush=True)
# 		model.actor.load_state_dict(torch.load(actor_model))
# 		model.critic.load_state_dict(torch.load(critic_model))
# 		print(f"Successfully loaded.", flush=True)
# 	elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
# 		print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
# 		sys.exit(0)
# 	else:
# 		print(f"Training from scratch.", flush=True)

# 	model.train()

def train(env,hyperparameters, num_runs, actor_model, critic_model):
	"""
		Trains the model.

		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			num_runs - number of independent runs to average over
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training

		Return:
			None
	"""	
	now = datetime.now()
	formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
	if hyperparameters['CBF']:
		folder_name = 'CBF_' + formatted_time
	else:
		folder_name = 'noCBF_' + formatted_time


	print(f"Training started...", flush=True)

	eps_rewards_all_runs = np.zeros((num_runs, hyperparameters['num_episodes']))
	total_violations = 0

	for i in range(num_runs):
		print(f"\nRun {i+1} / {num_runs}:", flush=True)
		print('-----------------------', flush=True)
		folder_name_run = folder_name + f'/{i+1}'

		hyperparameters['folder_name'] = folder_name_run #folder name to save the model and plots in

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

		eps_rewards_all_runs[i], num_violations = model.train()
		total_violations += num_violations

	save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', folder_name)
	os.makedirs(save_dir, exist_ok=True)

	print('Total constraint violations across all runs:', total_violations)

	np.save(os.path.join(save_dir, "eps_rewards_all_runs.npy"), eps_rewards_all_runs)
	print(f"Saved eps_rewards_all_runs to {save_dir}/eps_rewards_all_runs.npy")

	plot_final_results(eps_rewards_all_runs, folder_name)

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


def plot_final_results(rewards, folder_name):
	mean = rewards.mean(axis=0)
	std = rewards.std(axis=0, ddof=1)  # sample std
	lo = mean - 2 * std
	hi = mean + 2 * std

	x = np.arange(0, rewards.shape[1])
	plt.close()
	plt.plot(x, mean, linewidth=2)
	plt.fill_between(x, lo, hi, alpha=0.2)
	plt.xlabel('Episode')
	plt.ylabel('Episode Reward')
	plt.grid(True)


	base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots')
	plot_dir = os.path.join(base_dir, folder_name)
	os.makedirs(plot_dir, exist_ok=True)


	plot_path = os.path.join(plot_dir, 'final.png')
	plt.savefig(plot_path) #save the plot with the current date and time
	print('Plot saved!')
	plt.show()


if __name__ == "__main__":
	#########################################
	#SCENARIO-1:
	#########################################
	# t_windows=[[[0,300],[0,100]],[[0,300],[0,100]]] # STL time windows
	# subformula_types = np.array([4,4]) # 1: F, 2: G, 3: FG, 4: GF | Formula Types
	# agent_init_loc = np.array([0,0]) # Initial pos. (x,y) of the agent
	# roi = np.array([[-30, 30, 10],[-30, -30, 10]]) #3rd dimension is the radius

	# u_tar_max = np.array([.5, .5]) #max velocities of the targets
	# u_agent_max = 11 # Max vel. of the system
	# disturbance_interval = [-1, 1]
	# w_max = max(abs(disturbance_interval[0]), abs(disturbance_interval[1]))
	# u_agent_max = u_agent_max - w_max #reduce the max agent speed by the disturbance bound (worst-case)

	# target_movements = {0: {'type': 'circular', 'omega': 0.1, 'center_of_rotation': (-25, 30)}, 1: {'type': 'circular', 'omega': 0.1, 'center_of_rotation': (0, -35)}, 2: {'type': 'static'}, 3: {'type': 'static'}} #movement patterns for each target region
	
	
	#########################################
	#SCENARIO-2:
	#########################################
	# t_windows=[[[0,20]],[[0,20]],[[33,36]],[[0,20],[0,10]]] # STL time windows
	# subformula_types = np.array([1,1,2,4]) # 1: F, 2: G, 3: FG, 4: GF | Formula Types
	# agent_init_loc = np.array([0,0]) # Initial pos. (x,y) of the agent
	# u_tar_max = .1*np.array([.6, .6, .4, .5, .55])

	# roi=np.array([[-30, 30, 10],[0,-30, 10],[-30, -30, 10],[10, 20, 10]])

	# point1 = (-30, -30)
	# point2 = (-30, 30)

	# target_movements = {0: {'type': 'circular', 'omega': 0.1, 'center_of_rotation': (-25, 30)}, 1: {'type': 'circular', 'omega': 0.1, 'center_of_rotation': (0, -35)}, 2: {'type': 'periodic', 'point1': point1, 'point2': point2, 'heading_angle': np.arctan2(point2[1] - point1[1], point2[0] - point1[0])}, 3: {'type': "static"}} #movement patterns for each target region

	# u_agent_max = 11 # Max vel. of the system
	# disturbance_interval = [-1, 1]
	# w_max = max(abs(disturbance_interval[0]), abs(disturbance_interval[1]))
	# u_agent_max = u_agent_max - w_max #reduce the max agent speed by the disturbance bound (worst-case)

	#########################################
	#SCENARIO-3:
	#########################################
	t_windows=[[[0,60],[0,10]],[[150,180],[0,10]],[[0,300],[0,110]]] # STL time windows
	subformula_types = np.array([3,3,4]) # 1: F, 2: G, 3: FG, 4: GF | Formula Types
	#t_windows=[[[0,200],[0,90]],[[20,30]],[[120,130]]]
	#subformula_types = np.array([4,2,2]) # 1: F, 2: G, 3: FG, 4: GF | Formula Types
	agent_init_loc = np.array([0,0]) # Initial pos. (x,y) of the agent
	u_tar_max = np.array([1.2, 1.2, 1])
	target_labels = ['Target1', 'Target2', 'Charger']

	roi=np.array([[-30, 30, 11],[0,-10, 11],[0, -60, 11]])
	rois = [roi]

	point1 = (-70, 60)
	point2 = (70, 60)

	#target_movements = {0: {'type': 'random_walk', 'heading_angle': np.random.uniform(0, 2 * np.pi)}, 1: {'type': 'random_walk', 'heading_angle': np.random.uniform(0, 2 * np.pi)}, 2: {'type': 'periodic', 'point1': point1, 'point2': point2, 'heading_angle': np.arctan2(point2[1] - point1[1], point2[0] - point1[0])}} #movement patterns for each target region
	target_movements = {0: {'type': 'random_walk', 'heading_angle': np.random.uniform(0, 2 * np.pi)}, 1: {'type': 'random_walk', 'heading_angle': np.random.uniform(0, 2 * np.pi)}, 2: {'type': 'circular', 'center_of_rotation': (10, 0)}} #movement patterns for each target region
	
	#create a STL task dictionary:
	STL_dict = {
		'roi': roi,
		't_windows': t_windows,
		'subformula_types': subformula_types,
		'u_tar_max': u_tar_max,
		'target_movements': target_movements}


	target_colors = ['blue', 'red', 'green', 'black', 'yellow']
	simulation_targets = {}
	for key, value in target_movements.items():
		simulation_targets[key] = {
			'id': key,
			'center': roi[key][:2],
			'radius': roi[key][2],
			'u_max': u_tar_max[key],
			'movement': value,
			'color': target_colors[key],
			'label': target_labels[key]
		}


	#Static goal region:
	goal_region_radius = 11
	goals = {
	0: {'center': (50, 0), 'radius': goal_region_radius, 'movement':{'type':'static'}}, #goal region for the agent
	#1: {'center': (-50, 0), 'radius': 10, 'movement':{'type':'static'}}
	}

	u_agent_max = 15 # Max vel. of the system
	disturbance_interval = [-1, 1]


    #config dictionary for the environment
	config = {
        'init_loc':agent_init_loc, #initial location of the agent (x, y)
        "width": 100.0,
        "height": 100.0,
        "dt": 1,
        "render": False,
		'dt_render': 0.01,
		'goals': goals, #goal regions for the agent
        "obstacle_location": [100.0, 100.0],
        "obstacle_size": 0.0,
        "randomize_loc": True, #whether to randomize the agent location at the end of each episode
		'deterministic': False,
		'auto_entropy':True,
		"dynamics": "single integrator", #dynamics model to use
		"targets": simulation_targets, #target regions for the agent to visit
		"disturbance": disturbance_interval #disturbance range in both x and y directions [w_min, w_max]
    }


	CBF_parameters = {
		'alpha': 1.5, #weight for the CBF term
		"u_agent_max": u_agent_max, #max agent speed
		"STL": STL_dict,
		'sim_targets': simulation_targets,
    }

	action_range = [3, 3] #action range for the RL model (for the neural network output layer) [3,3] for case-1, [4,4] for case-2

	#learning hyperparameters:
	hyperparameters = {
				'gamma': 0.99,
				'tau': 0.005,
				'hidden_size': 256, 
				'buffer_size': int(1e6),
				'batch_size': 256,
				'max_timesteps_per_episode': 300, 
				'num_episodes': 20,
				'n_updates_per_iteration': 1,
				'deterministic': False,
				'auto_entropy':True,
				'action_range': action_range,
				'action_clip' : True,
				'CBF': True,
				'CBF_params': CBF_parameters,
				'online': False,
				'disturbance': disturbance_interval
			  }


	args = get_args()

	env = Continuous2DEnv(config)

	# Train or test, depending on the mode specified
	if args.mode == 'train':
		num_runs = 1 #number of independent runs to average over
		train(env=env, hyperparameters=hyperparameters, num_runs=num_runs, actor_model='', critic_model='')

	elif args.mode == 'test':
		config['render'] = False #enable rendering for testing
		config['dt_render'] = 0.03
		config['init_loc'] = [0.0, -1]
		config['randomize_loc'] = False #randomize the agent location at the end of each episode
		env = Continuous2DEnv(config)
		max_timesteps_per_episode = hyperparameters['max_timesteps_per_episode']
		# Load in the model file
		model_path= select_model_file()
		test(env=env, hyperparameters=hyperparameters, model_path=model_path, max_timesteps_per_episode=max_timesteps_per_episode)