import numpy as np

def _log_summary(ep_len, ep_ret, ep_num):
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
		

def rollout(policy, env, max_timesteps_per_episode):
    deterministic = True
    
    for i in range(10):  # number of test episodes
        obs = env.reset()

        t = 0  # number of timesteps so far

        ep_len = 0  # episodic length
        ep_ret = 0  # episodic return

        while t < max_timesteps_per_episode:  # max number of timesteps in each episode
            t += 1

            action = policy.get_action(obs, deterministic=deterministic)

            next_obs, reward, done = env.step(action)

            ep_len += 1
            ep_ret += reward

            obs = next_obs

        yield ep_len, ep_ret

        _log_summary(ep_len, ep_ret, i + 1)  # Log summary for this episode



def eval_policy(policy, env, max_timesteps_per_episode):
	for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, max_timesteps_per_episode)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)