#USES THE DIFFERENTIABLE CBF LAYER FROM THE SAFE RL WITH ROBUST CBF PAPER!!!



# import comet_ml at the top of your file
#from comet_ml import Experiment

import argparse
import time
from datetime import datetime
import torch
import numpy as np

#from tcbf_sac.generate_rollouts import generate_model_rollouts
from tcbf_sac.sac_tcbf import TCBF_SAC
from tcbf_sac.replay_memory import ReplayMemory
#from tcbf_sac.dynamics import DynamicsModel
import os
from simulator import Continuous2DEnv
import TCBF
import matplotlib.pyplot as plt

from tcbf_sac.utils import prGreen, get_output_folder, prYellow


def train(agent, env, args, experiment=None):

    # Load the weight if we're continuing training
    if hasattr(args, 'load_agent'):
        agent.load_weights(args.resume)

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)
    memory_model = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0

    episode_rewards = []

    for i_episode in range(args.max_episodes):
        episode_reward = 0
        episode_cost = 0
        episode_steps = 0
        done = False
        state= env.reset()

        while episode_steps < args.max_steps and not done:
            if episode_steps % 10 == 0:
                prYellow('Episode {} - step {} - eps_rew {} - eps_cost {}'.format(i_episode, episode_steps, episode_reward, episode_cost))
            #state = dynamics_model.get_state(obs)
            # Generate Model rollouts
            if args.model_based and episode_steps % 5 == 0 and len(memory) > dynamics_model.max_history_count / 3:
                memory_model = generate_model_rollouts(env, memory_model, memory, agent, dynamics_model,
                                                       k_horizon=args.k_horizon,
                                                       batch_size=min(len(memory), 5 * args.rollout_batch_size),
                                                       warmup=args.start_steps > total_numsteps)

            # If using model-based RL then we only need to have enough data for the real portion of the replay buffer
            if len(memory) + len(memory_model) * args.model_based > args.batch_size:

                # Number of updates per step in environment
                for i in range(args.updates_per_step):

                    # Update parameters of all the networks
                    if args.model_based:
                        # Pick the ratio of data to be sampled from the real vs model buffers
                        real_ratio = max(min(args.real_ratio, len(memory) / args.batch_size),
                                         1 - len(memory_model) / args.batch_size)
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                             args.batch_size,
                                                                                                             updates,
                                                                                                             dynamics_model,
                                                                                                             memory_model,
                                                                                                             real_ratio)
                    else:
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                         args.batch_size,
                                                                                                         updates, episode_steps, CBF_parameters)

                    if experiment:
                        experiment.log_metric('loss/critic_1', critic_1_loss, updates)
                        experiment.log_metric('loss/critic_2', critic_2_loss, step=updates)
                        experiment.log_metric('loss/policy', policy_loss, step=updates)
                        experiment.log_metric('loss/entropy_loss', ent_loss, step=updates)
                        experiment.log_metric('entropy_temperature/alpha', alpha, step=updates)
                    updates += 1

            # Sample action from policy
            action, cbf_action = agent.select_action(state,
                                            warmup=args.start_steps > total_numsteps, time = episode_steps, safe_action=args.cbf_mode!='off', cbf_info = CBF_parameters)  # Sample action from policy

            next_state, reward, done = env.step(action, TCBF.gamma(episode_steps, target_region_radius))  # Step

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            #episode_cost += next_info.get('cost', 0)

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            #mask = 1 if episode_steps == env.max_episode_steps else float(not done)
            mask = 1

            if args.cbf_mode == 'baseline':  # action is (rl_action + cbf_action)
                memory.push(state, action-cbf_action, reward, next_state, mask, t=episode_steps * env.dt, next_t=(episode_steps+1) * env.dt, cbf_info=CBF_parameters)  # Append transition to memory
            else:
                memory.push(state, action, reward, next_state, mask, t=episode_steps * env.dt, next_t=(episode_steps+1) * env.dt)  # Append transition to memory

            state = next_state

        episode_rewards.append(episode_reward)

        # [optional] save intermediate model
        if i_episode > 0 and i_episode % 20 == 0:
            agent.save_model(args.output)
            #dynamics_model.save_disturbance_models(args.output)

        if experiment:
            # Comet.ml logging
            experiment.log_metric('reward/train', episode_reward, step=i_episode)
            experiment.log_metric('cost/train', episode_cost, step=i_episode)
        prGreen("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, cost: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                             round(episode_reward, 2), round(episode_cost, 2)))

        # Evaluation
        if i_episode % 5 == 0 and args.eval is True:
            print('Size of replay buffers: real : {}, \t\t model : {}'.format(len(memory), len(memory_model)))
            avg_reward = 0.
            avg_cost = 0.
            episodes = 2
            test_eps_rewards = []
            for _ in range(episodes):
                state = env.reset()
                test_episode_reward = 0
                episode_cost = 0
                done = False
                episode_steps = 0
                while episode_steps < args.max_steps and not done:
                    action = agent.select_action(state, episode_steps, evaluate=True, safe_action=args.cbf_mode!='off', cbf_info = CBF_parameters)[0]  # Sample action from policy
                    next_state, reward, done= env.step(action, TCBF.gamma(episode_steps, target_region_radius))
                    test_episode_reward += reward
                    #episode_cost += next_info.get('cost', 0)
                    state = next_state
                    #info = next_info
                    episode_steps += 1

                test_eps_rewards.append(test_episode_reward)
                avg_reward += test_episode_reward
                avg_cost += episode_cost
            avg_reward /= episodes
            avg_cost /= episodes
            if experiment:
                experiment.log_metric('avg_reward/test', avg_reward, step=i_episode)
                experiment.log_metric('avg_cost/test', avg_cost, step=i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}, Avg. Cost: {}".format(episodes, round(avg_reward, 2), round(avg_cost, 2)))
            print("----------------------------------------")

    plt.close()
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Episode Rewards Over Time')

    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")

    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots')
    plot_path = os.path.join(plot_dir, f'QP-SAC {formatted_time}.png')
    plt.savefig(plot_path) #save the plot with the current date and time
    print('Plot saved!')




    plt.savefig(os.path.join(args.output, 'episode_rewards.png'))

def test(agent, args, env_conf,visualize=True, debug=True):

    model_path = args.resume
    safe_action = False
    agent.load_weights(model_path)

    def policy(observation):
        return agent.select_action(observation, safe_action=safe_action, evaluate=True)[0]

    # if visualize and 'Unicycle' in model_path:
    #     from plot_utils import plot_value_function
    #     plot_value_function(build_env(args.env_name), agent, dynamics_model, save_path=model_path, safe_action=False)

    episode_rewards = []
    dones = []

    if visualize:  
        env_conf['render'] = True
        env_conf['dt_render'] = 0.01

    for episode in range(args.validate_episodes):

        env = Continuous2DEnv(env_conf)

        if agent.cbf_layer:
            agent.cbf_layer.env = env

        # reset at the start of episode
        observation = env.reset()
        episode_steps = 0
        episode_reward = 0.
        assert observation is not None

        # Time policy
        policy_timings = []

        # start episode
        done = False
        while episode_steps < args.max_steps and not done:

            # basic operation, action ,reward, blablabla ...
            policy_start_time = time.time()
            action = policy(observation)
            policy_timings.append(time.time() - policy_start_time)

            observation, reward, done = env.step(action,TCBF.gamma(episode_steps, target_region_radius))

            # update
            episode_reward += reward
            episode_steps += 1

        episode_rewards.append(episode_reward)
        #dones.append(done and env.episode_step < env.max_episode_steps)

        if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}, mean_reward:{}, std_reward:{}, mean_completion:{}, policy_mean_wct={}'.format(episode, episode_reward, np.mean(episode_rewards), np.std(episode_rewards), np.mean(dones), np.mean(policy_timings)))


    if debug:
        prYellow('[Evaluate]: mean_reward:{}, std_reward:{}, mean_completion:{}'.format(np.mean(episode_rewards), np.std(episode_rewards), np.mean(dones)))

    return np.mean(episode_rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

    # SAC Args
    parser.add_argument('--env_name', default="Unicycle", help='Options are Unicycle or SimulatedCars.')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--visualize', action='store_true', dest='visualize', help='visualize env -only available test mode')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 5 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automatically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=12345, metavar='N',
                        help='random seed (default: 12345)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--max_episodes', type=int, default=200, metavar='N',
                        help='maximum number of episodes (default: 200)')
    parser.add_argument('--max_steps', type=int, default=200, metavar='N', 
                        help='maximum number of steps in one episode (default: 200)'),

    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=5000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--device_num', type=int, default=0, help='Select GPU number for CUDA (default: 0)')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--validate_episodes', default=5, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--validate_steps', default=1000, type=int, help='how many steps to perform a validate experiment')
    # CBF, Dynamics, Env Args
    parser.add_argument('--gp_model_size', default=2000, type=int, help='gp')
    parser.add_argument('--gp_max_episodes', default=100, type=int, help='gp max train episodes.')
    parser.add_argument('--k_d', default=3.0, type=float)
    parser.add_argument('--gamma_b', default=20, type=float)
    parser.add_argument('--l_p', default=0.03, type=float,
                        help="Look-ahead distance for unicycle dynamics output.")
    # Model Based RL
    parser.add_argument('--model_based', action='store_true', dest='model_based', help='If selected, will use data from the model to train the RL agent.')
    parser.add_argument('--real_ratio', default=0.3, type=float, help='Portion of data obtained from real replay buffer for training.')
    parser.add_argument('--k_horizon', default=1, type=int, help='horizon of model-based rollouts')
    parser.add_argument('--rollout_batch_size', default=5, type=int, help='Size of initial states batch to rollout from.')
    # Modular Task Learning
    parser.add_argument('--cbf_mode', default='mod', help="Options are `off`, `baseline`, `full`, `mod`.")

    args = parser.parse_args()


    if args.resume == 'default':
        args.resume = os.getcwd() + '/output/{}-run0'.format(args.env_name)
    elif args.resume.isnumeric():
        args.resume = os.getcwd() + '/output/{}-run{}'.format(args.env_name, args.resume)
        args.load_agent = True

    if args.cuda:
        torch.cuda.set_device(args.device_num)

    # Environment

    target_region_center = [0.0, 3]
    target_region_radius = 0.5


    env_config = {
        'init_loc':[0.0, 0.0, 0.0], #initial location of the agent (x, y, theta)
        "width": 8.0,
        "height": 8.0,
        "dt": 0.1,
        "render": False,
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
        'action_max': np.array([5, 3]),
        'action_min': np.array([-5, -3])
    }

    CBF_parameters = {
		"target_region_center": target_region_center,
        "target_region_radius": target_region_radius,
		'epsilon' : 0.5, #for reference point,
		'alpha': 5, #weight for the CBF term
    }



    env = Continuous2DEnv(env_config)

    # Agent
    agent = TCBF_SAC(env.observation_space.shape[0], env.action_space, env, args)

    # # Random Seed
    # if args.seed > 0:
    #     env.seed(args.seed)
    #     env.action_space.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     np.random.seed(args.seed)

    if args.mode == 'train':
        args.output = get_output_folder(args.output, args.env_name)
        experiment = None
        train(agent, env, args, experiment)


    elif args.mode == 'test':
        test(agent, args, env_conf=env_config, visualize=args.visualize, debug=True)

   
