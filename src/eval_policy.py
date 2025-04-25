import numpy as np

"""
	This file is used only to evaluate our trained policy/actor after
	training in main.py with ppo.py.
"""

def _log_summary(ep_len, ep_ret, ep_num):
		"""
			Print to stdout what we've logged so far in the most recent episode.

			Parameters:
				None

			Return:
				None
		"""
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

def rollout(policy, ppo_model, action_range, env):
	"""
		Returns a generator to roll out each episode given a trained policy and
		environment to test on. 

		Parameters:
			policy - The trained policy to test
			env - The environment to evaluate the policy on
			render - Specifies whether to render or not
		
		Return:
			A generator object rollout, or iterable, which will return the latest
			episodic length and return on each iteration of the generator.
	"""
	action_low = np.array([-action_range[0], -action_range[1]])
	action_high = np.array([action_range[0], action_range[1]])
	# Rollout until user kills process
	for i in range(10): # number of test episodes
		obs = env.reset()
		done = False

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return

		while not done and t < 100: # max number of timesteps in each episode
			t += 1

			# Query deterministic action from policy and run it
			if ppo_model == 'gaussian':
				action = policy(obs).detach().numpy()
				#action = action_low + (action_high - action_low) * action

			elif ppo_model == 'beta':
				# For beta policy, we need to sample from the beta distribution
				alpha, beta = policy(obs)
				alpha = alpha.detach().numpy()
				beta = beta.detach().numpy()
				# action = np.random.beta(alpha.detach().numpy(), beta.detach().numpy())
				# action = action_low + (action_high - action_low) * action
				action = alpha / (beta + alpha)
				action = action_low + (action_high - action_low) * action

			obs, rew, done = env.step(action)

			# Sum all episodic rewards as we go along
			ep_ret += rew
			
		# Track episodic length
		ep_len = t

		# returns episodic length and return in this iteration
		yield ep_len, ep_ret

def eval_policy(policy, ppo_model, action_range, env):
	"""
		The main function to evaluate our policy with. It will iterate a generator object
		"rollout", which will simulate each episode and return the most recent episode's
		length and return. We can then log it right after. And yes, eval_policy will run
		forever until you kill the process. 

		Parameters:
			policy - The trained policy to test, basically another name for our actor model
			env - The environment to test the policy on
			render - Whether we should render our episodes. False by default.

		Return:
			None

		NOTE: To learn more about generators, look at rollout's function description
	"""
	# Rollout with the policy and environment, and log each episode's data
	for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, ppo_model, action_range, env)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)