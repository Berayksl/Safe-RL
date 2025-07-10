import math
import random
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

from IPython.display import clear_output
from IPython.display import display

from networks import PolicyNetwork, SoftQNetwork, ValueNetwork
import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt
from Sequential_CBF import *
import copy



class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)



class SAC:

    def __init__(self, env, **hyperparameters):
        self._init_hyperparameters(hyperparameters)

        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=self.device)

        self.policy_net = PolicyNetwork(self.obs_dim, self.act_dim, self.hidden_size, self.device, self.action_range).to(self.device)
        self.soft_q_net1 = SoftQNetwork(self.obs_dim, self.act_dim, self.hidden_size).to(self.device)
        self.soft_q_net2 = SoftQNetwork(self.obs_dim, self.act_dim, self.hidden_size).to(self.device)
        self.target_soft_q_net1 = SoftQNetwork(self.obs_dim, self.act_dim, self.hidden_size).to(self.device)
        self.target_soft_q_net2 = SoftQNetwork(self.obs_dim, self.act_dim, self.hidden_size).to(self.device)

        # Initialize target networks with the same weights as the original networks
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.policy_optim = Adam(self.policy_net.parameters(), lr=self.policy_lr)
        self.soft_q_optim1 = Adam(self.soft_q_net1.parameters(), lr=self.soft_q_lr)
        self.soft_q_optim2 = Adam(self.soft_q_net2.parameters(), lr=self.soft_q_lr)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.alpha_lr)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()


        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.logger = {
            'delta_t': time.time_ns(),
			'i_so_far': 0,          # iterations so far
            'eps_rewards': [],
            'actor_losses': [],
            'critic_losses': []
            }


    def _init_hyperparameters(self, hyperparameters):

        #SAC hyperparameters:
        self.soft_q_lr = hyperparameters.get("soft_q_lr", 3e-4)
        self.policy_lr = hyperparameters.get("policy_lr", 3e-4)
        self.alpha_lr = hyperparameters.get("alpha_lr", 3e-4)
        self.hidden_size = hyperparameters.get("hidden_size", 256)

        self.gamma = hyperparameters.get("gamma", 0.99)
        self.tau = hyperparameters.get("tau", 0.005)
        #self.alpha = hyperparameters.get("alpha", 0.2)
        self.replay_buffer_size = hyperparameters.get("buffer_size", int(1e6))
        self.batch_size = hyperparameters.get("batch_size", 256)
        self.max_timesteps_per_episode = hyperparameters.get("max_timesteps_per_episode", 1000)
        self.num_episodes = hyperparameters.get("num_episodes", 1000)
        self.n_updates_per_iteration = hyperparameters.get('n_updates_per_iteration', 1)
        self.deterministic = hyperparameters.get("deterministic", False) #whether to use deterministic policy or not during training
        self.auto_entropy = hyperparameters.get("auto_entropy", True) #whether to use automatic entropy tuning or not

        #
        self.action_range = hyperparameters.get("action_range", None) #max speed and max turning rate
        
        #CBF parameters:
        self.CBF = hyperparameters.get("CBF", False) #whether to use CBF or not
        self.CBF_params = hyperparameters.get("CBF_params", None) #CBF parameters, if any


    def update(self, batch_size, auto_entropy=True, target_entropy=-2, gamma=0.99,soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        #reward = (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

        # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action), self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optim1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optim1.step()                                

        self.soft_q_optim2.zero_grad()                     
        q_value_loss2.backward()
        self.soft_q_optim2.step()

        # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action),self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        # Logging
        self.logger['actor_losses'].append(policy_loss.item())
        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return predicted_new_q_value.mean()


    def train(self):
        explore_steps = 0 #number of steps to explore before using the policy
        frame_idx = 0 #num of frames so far
        infeasible_solutions = 0 #number of infeasible solutions found during training

        initial_targets = [(key, copy.deepcopy(value)) for key, value in self.CBF_params['targets'].items()]
        u_agent_max = self.CBF_params['u_agent_max']

        global_t = 0

        targets = self.CBF_params['targets']
        for eps in range(self.num_episodes):
            #reset the targets
            # for target in initial_targets:
            #     targets[target[0]] = copy.deepcopy(target[1]) #use a deep copy to reset the target dictionaries
          
            state =  self.env.reset()
            episode_reward = 0

            step = 0 #reset step counter for each episode
             
            while step <= self.max_timesteps_per_episode:
                if frame_idx < explore_steps:
                    action_RL = self.policy_net.sample_action()
                    print('Exploration action:', action_RL)
                else:
                    action_RL = self.policy_net.get_action(state, deterministic = self.deterministic)
                    if self.CBF and len(targets) != 0:
                        #To hold inside a target region:
                        # if target_region_radius - np.sqrt((state[0] - target_region_center[0])**2 + (state[1] - target_region_center[1])**2) >= 0.0 and TCBF.CBF(state, remaining_t, u_agent_max, target_region_center, target_region_radius, u_target_max) < 2:
                        #     for i in range(hold_time):#stay inside the target region for hold_time steps
                        #         remaining_t = 1 #set it to 1 to stay inside the target region
                        #         target_region_center = TCBF.moving_target(global_t, x0=initial_center[0], y0=initial_center[1], u_target_max= u_target_max, omega=omega)
                        #         action = TCBF.solve_cbf_qp(TCBF.CBF, state, step, remaining_t, target_region_center, target_region_radius, omega, u_agent_max, u_target_max, action_RL)
                        #         action = (action[0] + action_RL[0] , action[1] + action_RL[1]) #a_rl + a_cbf
                        #         state, reward, done = self.env.step(action, target_region_center=target_region_center)
                        #         step += 1
                        #         global_t += 1 #increment global time step
                        #     #target_reached = True
                        #     print("Target region reached!")

                        cbf_values = {}
                        for target_index in targets.keys():
                            cbf_value = sequential_CBF(state, u_agent_max, targets, target_index)
                            cbf_values[target_index] = cbf_value

                        min_key = min(cbf_values, key=cbf_values.get)  #find the target region with the minimum CBF value

                        #Now solve the QP to get the control input for the target region with the minimum CBF value:
                        action_CBF = solve_cbf_qp(sequential_CBF, state, u_agent_max, min_key, step, targets, action_RL)

                        if action_CBF is None:
                            # If the QP fails, we can either ignore the CBF action or set it to zero
                            action= (0.0, 0.0)
                            print('CBF action is None, using zero action instead.')
                            infeasible_solutions += 1 #increment the number of infeasible solutions
                        else:
                            action = (action_RL[0] + action_CBF[0] , action_RL[1] + action_CBF[1]) #a_rl + a_cbf

                    else:
                        action = action_RL
                #print('Action:', action)
                next_state, reward, done = self.env.step(action)

                step += 1 #increment step counter
                global_t += 1 #increment global time step

                #decrease the remaining time and update the center for each target region
                for target_index in list(targets.keys()):
                    targets[target_index]['remaining_time'] -= 1

                    #calculate the signed distance to each target region:
                    target_center = targets[target_index]['center']
                    target_radius = targets[target_index]['radius']
                    dist = np.linalg.norm(state[:2] - target_center)
                    signed_distance = dist - target_radius
                    
                    if signed_distance <= 0:
                        targets.pop(target_index)  # Remove target region if the agent is inside it
                        targets[target_index] = copy.deepcopy(initial_targets[target_index][1])  # add the target region back to the dictionary with the initial parameters


                #print('reward;',reward)     
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                frame_idx += 1
                
                
                if len(self.replay_buffer) > self.batch_size:
                    for i in range(self.n_updates_per_iteration):
                        _=self.update(self.batch_size, auto_entropy=self.auto_entropy, target_entropy = -1. * self.act_dim)

                # if done: #terminate the episode if target region is reached
                #     break
            

            if eps % 20 == 0 and eps>0: # plot and model saving interval
                    self.save_model()

            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
            self.logger['eps_rewards'].append(episode_reward)
            #self._log_summary()

        #save the final model:
        self.save_model()

        # plot the results and save the graph:
        plt.close()
        plt.plot(self.logger['eps_rewards'])
        plt.xlabel('Iteration Number')
        plt.ylabel('Episodic Return')
       #plt.title(f'CBF: {self.CBF}, Violations: {self.safety_violations}, Iterations: {len(self.logger["avg_batch_rews"])}, Updates per Iterations: {self.n_updates_per_iteration}')
        plt.grid()

        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        # if self.CBF:
        #     formatted_time = 'CBF_' + formatted_time
        # else:
        #     formatted_time = 'noCBF_' + formatted_time

        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots')
        plot_path = os.path.join(plot_dir, f'{formatted_time}.png')
        plt.savefig(plot_path) #save the plot with the current date and time
        print('Plot saved!')

        plt.show()

        plt.plot(self.logger['eps_rewards'])



    def save_model(self):
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
        os.makedirs(model_dir, exist_ok=True)

        torch.save(self.soft_q_net1.state_dict(), os.path.join(model_dir, 'Q1.pth'))
        torch.save(self.soft_q_net2.state_dict(), os.path.join(model_dir, 'Q2.pth'))
        torch.save(self.policy_net.state_dict(), os.path.join(model_dir, 'policy.pth'))
        print('Model saved!')


    def load_model(self, path):
        # self.soft_q_net1.load_state_dict(torch.load(path+'_q1'))
        # self.soft_q_net2.load_state_dict(torch.load(path+'_q2'))
        self.policy_net.load_state_dict(torch.load(path))

        # self.soft_q_net1.eval()
        # self.soft_q_net2.eval()
        self.policy_net.eval()


    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        i_so_far = self.logger['i_so_far']
        #avg_ep_lens = np.mean(self.logger['eps_lens'])
        ep_rew = self.logger['eps_rewards'][-1]
        actor_loss = self.logger['actor_losses'][-1] #take the last actor loss

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        #print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Episode Return: {ep_rew}", flush=True)
        print(f"Actor Loss: {actor_loss}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)
