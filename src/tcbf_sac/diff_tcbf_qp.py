import argparse
import numpy as np
import torch
#from rcbf_sac.dynamics import DYNAMICS_MODE
#from rcbf_sac.utils import to_tensor, to_numpy, prRed, get_polygon_normals, sort_vertices_cclockwise
from time import time
from qpth.qp import QPFunction

class CBFQPLayer:

    def __init__(self, env):
        """Constructor of CBFLayer.

        Parameters
        ----------
        env : gym.env
            Gym environment.
        gamma_b : float, optional
            gamma of control barrier certificate.
        k_d : float, optional
            confidence parameter desired (2.0 corresponds to ~95% for example).
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = env
        #self.u_min, self.u_max = self.get_control_bounds()

        # if self.env.dynamics_mode == 'Unicycle':
        #     self.k_d = k_d
        #     self.l_p = l_p


        self.action_dim = env.action_space.shape[0]
        # self.num_ineq_constraints = self.num_cbfs + 2 * self.action_dim

    def gamma(self, t, r=1.0):
        return max(r/2 + 10- 0.05 * t , r/2)

    def gamma_dt(self, t):
        return -0.05
    
    def get_tcbf_qp_constraints(self, state_batch, a_rl_batch, t, cbf_info):
        """
        Builds the QP inequality constraint matrices G and h for a batched time-varying CBF constraint.

        state_batch: (B, 3) tensor of [x, y, theta]
        a_rl_batch:  (B, 2) nominal control input (used for f_func)
        t: current time (scalar)
        centers: (B, 2) centers of obstacles
        radius: scalar radius of obstacle
        epsilon: float
        alpha: class K function gain
        """
        
        center = cbf_info['target_region_center']  # (2,) center of the target region
        radius = cbf_info['target_region_radius']  # scalar radius of the target region
        epsilon = cbf_info['epsilon']  # float, epsilon for the reference point
        alpha = cbf_info['alpha']  # float, alpha for the K function gain

        
        batch_size = state_batch.shape[0]

        x = state_batch[:, 0]
        y = state_batch[:, 1]
        theta = state_batch[:, 2]

        epsilon = torch.tensor(epsilon, device=theta.device)

        xc = center[0]
        yc = center[1]

        dx = x - xc
        dy = y - yc

        # Distance to the center
        dist = torch.sqrt(dx**2 + dy**2).clamp(min=1e-6) # Avoid division by zero
        b = self.gamma(t, radius) - dist  # scalar barrier function

        # ∇b = [-dx / dist, -dy / dist, 0]
        grad_b = torch.stack([-dx / dist, -dy / dist, torch.zeros_like(dist)], dim=1)  # (B, 3)

        # g(x) matrix: (B, 3, 2)
        g = torch.zeros((batch_size, 3, 2), device=state_batch.device)
        g[:, 0, 0] = torch.cos(theta)
        g[:, 0, 1] = -epsilon * torch.sin(theta)
        g[:, 1, 0] = torch.sin(theta)
        g[:, 1, 1] = epsilon * torch.cos(theta)
        g[:, 2, 1] = 1.0

        # f(x): compute nominal dynamics under a_rl_batch
        v = a_rl_batch[:, 0]
        w = a_rl_batch[:, 1]
        f = torch.zeros((batch_size, 3), device=state_batch.device)
        f[:, 0] = torch.cos(theta) * v - epsilon * torch.sin(theta) * w
        f[:, 1] = torch.sin(theta) * v + epsilon * torch.cos(theta) * w
        f[:, 2] = w

        # Lg b(x): (B, 2)
        Lgb = torch.bmm(grad_b.unsqueeze(1), g).squeeze(1)  # (B, 2)
        # Lf b(x): (B,)
        Lfb = torch.sum(grad_b * f, dim=1)

        # Right-hand side of constraint
        rhs = Lfb + self.gamma_dt(t) + alpha * b  # (B,)

        # G = -Lgb  # (B, 2)
        # h = rhs   # (B,)

        # Convert to inequality form: -Lgb * u <= rhs
        slack_column = -torch.ones((batch_size, 1), device=self.device) #add slack variables for the QP
        G = torch.cat([-Lgb, slack_column], dim=1).unsqueeze(1) 
        h = rhs.unsqueeze(1)

        return G, h
    


    def get_safe_action(self, state_batch, action_batch, t, modular=False, cbf_info=None):
        """
        Parameters
        ----------
        state_batch : torch.tensor or ndarray
        action_batch : torch.tensor or ndarray
            State batch
        mean_pred_batch : torch.tensor or ndarray
            Mean of disturbance
        sigma_batch : torch.tensor or ndarray
            Standard deviation of disturbance

        Returns
        -------
        final_action_batch : torch.tensor
            Safe actions to take in the environment.
        """

        # batch form if only a single data point is passed
        expand_dims = len(state_batch.shape) == 1
        if expand_dims:
            action_batch = action_batch.unsqueeze(0)
            state_batch = state_batch.unsqueeze(0)


        start_time = time()
        G, h = self.get_tcbf_qp_constraints(state_batch, action_batch, t, cbf_info)
        build_qp_time = time()
        safe_action_batch = self.solve_qp(G, h)
        # prCyan('Time to get constraints = {} - Time to solve QP = {} - time per qp = {} - batch_size = {} - device = {}'.format(build_qp_time - start_time, time() - build_qp_time, (time() - build_qp_time) / safe_action_batch.shape[0], Ps.shape[0], Ps.device))
        # The actual safe action is the cbf action + the nominal action
        #final_action = torch.clamp(action_batch + safe_action_batch, self.u_min.repeat(action_batch.shape[0], 1), self.u_max.repeat(action_batch.shape[0], 1))

        # print(action_batch.device)
        # print(safe_action_batch.device)        

        final_action = action_batch + safe_action_batch[:, :action_batch.shape[1]]

        return final_action if not expand_dims else final_action.squeeze(0)

    def solve_qp(self, G: torch.Tensor, h: torch.Tensor):
        """
        Solves:
            minimize_{u} 0.5 * u^T P u + q^T u
                subject to G u <= h

        Parameters
        ----------
        Ps : torch.Tensor
            (batch_size, n_u, n_u)
        qs : torch.Tensor
            (batch_size, n_u)
        G : torch.Tensor
            (batch_size, num_constraints, n_u)
        h : torch.Tensor
            (batch_size, num_constraints)

        Returns
        -------
        safe_action_batch : torch.Tensor
            The QP solution u (safe control).
        """

        # Normalize constraints for numerical stability
        Gh = torch.cat((G, h.unsqueeze(-1)), dim=2)
        Gh_norm = torch.max(torch.abs(Gh), dim=2, keepdim=True)[0]
        G /= Gh_norm
        h /= Gh_norm.squeeze(-1)

        # Solve QPs using your differentiable QP solver (cbf_layer)
        sol = self.cbf_layer(
            G, h,
            solver_args={
                "check_Q_spd": False,
                "maxIter": 100000,
                "notImprovedLim": 10,
                "eps": 1e-4
            }
        )

        return sol  # shape: (batch_size, n_u)


    def cbf_layer(self, G, h, A=None, b=None, solver_args=None):
        """
        Parameters
        ----------
        Qs : torch.Tensor
        ps : torch.Tensor
        G : torch.Tensor
            shape (batch_size, num_ineq_constraints, num_vars)
        h : torch.Tensor
            shape (batch_size, num_ineq_constraints)
        As : torch.Tensor, optional
        bs : torch.Tensor, optional
        solver_args : dict, optional

        Returns
        -------
        result : torch.Tensor
            Result of QP
        """
        #move to device
        G = G.to(self.device)
        h = h.to(self.device)

        num_vars = G.shape[-1]
        batch_size = G.shape[0]

        if solver_args is None:
            solver_args = {}

        if A is None or b is None:
            A = torch.Tensor().to(self.device).double()
            b = torch.Tensor().to(self.device).double()

        #Q = torch.eye(num_vars).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        slack_penalty = 100
        Q = torch.eye(num_vars).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        Q[:, -1, -1] = slack_penalty

        #p = torch.zeros(batch_size, num_vars).to(self.device)
        p = torch.zeros(batch_size, num_vars).to(self.device)


        result = QPFunction(verbose=0, **solver_args)(Q.double(), p.double(),G.double(), h.double(), A, b).float()

        if torch.any(torch.isnan(result)):
            print('QP Failed to solve - result is nan == {}!'.format(torch.any(torch.isnan(result))))
            raise Exception('QP Failed to solve')
        return result



    def get_control_bounds(self):
        """

        Returns
        -------
        u_min : torch.tensor
            min control input.
        u_max : torch.tensor
            max control input.
        """

        u_min = torch.tensor(self.env.safe_action_space.low).to(self.device)
        u_max = torch.tensor(self.env.safe_action_space.high).to(self.device)

        return u_min, u_max

if __name__ == "__main__": 
    import sys
    import os
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(parent_dir)


    from simulator import Continuous2DEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_region_center = [2.0, 2.0]
    target_region_radius = 1

    #config dictionary for the environment
    config = {
        'init_loc': [-1.0, -3.0, 0.0], #initial location of the agent (x, y, theta)
        "width": 8.0,
        "height": 8.0,
        "dt": 0.1,
        "render": True,
        'dt_render': 0.01,
        "goal_location": [8.0, 8.0],
        "goal_size": 1,
        "obstacle_location": [10.0, 10.0],
        "obstacle_size": 0.0,
        "target_region_center": target_region_center,
        "target_region_radius": target_region_radius,
        "randomize_loc": False #whether to randomize the agent location at the end of each episode
    }

    #CBF parameters
    #epsilon = 0.5 #for reference point

    env = Continuous2DEnv(config)
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).to(device)    

    target_reached = False

    action_rl = torch.tensor([2.0, 0.0], dtype=torch.float32).to(device)

    action = action_rl

    CBF = CBFQPLayer(env)

    CBF_parameters = {
    "target_region_center": target_region_center,
    "target_region_radius": target_region_radius,
    'epsilon' : 0.5, #for reference point,
    'alpha': 5, #weight for the CBF term
    }


    for t in range(400):
        if target_region_radius**2 - ((state[0] - target_region_center[0])**2 + (state[1] - target_region_center[1])**2) >= target_region_radius/2 - 0.2:
            target_reached = True
            print("Target region reached!")

        if target_reached:
            action = action_rl
        else:
            action = CBF.get_safe_action(state, action_rl, t, cbf_info = CBF_parameters)
            print("Action:", action)


        #print("Action:", action)
        action = action.cpu().numpy()
        state, reward, done = env.step(action, CBF.gamma(t))
        state = torch.tensor(state, dtype=torch.float32).to(device)
        #print(state)
        if done:
            break
