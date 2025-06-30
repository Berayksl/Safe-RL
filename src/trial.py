
import argparse
import numpy as np
import torch
#from rcbf_sac.dynamics import DYNAMICS_MODE
#from rcbf_sac.utils import to_tensor, to_numpy, prRed, get_polygon_normals, sort_vertices_cclockwise
from time import time
from qpth.qp import QPFunction


def gamma(t, r=1.0):
    return max(r/2 + 10- 0.05 * t , r/2)

def gamma_dt(t):
    return -0.05

def get_tcbf_qp_constraints(state_batch, a_rl_batch, t, center, radius, epsilon, alpha=5.0):
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
    b = gamma(t, radius) - dist  # (B,) shape

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
    rhs = Lfb + gamma_dt(t) + alpha * b  # (B,)

    # Convert to inequality form: -Lgb * u <= rhs
    G = -Lgb  # (B, 2)
    h = rhs   # (B,)

    return G, h

def solve_qp(Gs: torch.Tensor, hs: torch.Tensor):
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
    Gs : torch.Tensor
        (batch_size, num_constraints, n_u)
    hs : torch.Tensor
        (batch_size, num_constraints)

    Returns
    -------
    safe_action_batch : torch.Tensor
        The QP solution u (safe control).
    """

    # Normalize constraints for numerical stability
    Ghs = torch.cat((Gs, hs.unsqueeze(-1)), dim=1)
    Ghs_norm = torch.max(torch.abs(Ghs), dim=1, keepdim=True)[0]
    Gs /= Ghs_norm
    hs /= Ghs_norm.squeeze(-1)

    # Solve QPs using your differentiable QP solver (cbf_layer)
    sol = cbf_layer(
        Gs, hs,
        solver_args={
            "check_Q_spd": False,
            "maxIter": 100000,
            "notImprovedLim": 10,
            "eps": 1e-4
        }
    )

    return sol  # shape: (batch_size, n_u)




def cbf_layer(Gs, hs, As=None, bs=None, solver_args=None):
    """
    Parameters
    ----------
    Qs : torch.Tensor
    ps : torch.Tensor
    Gs : torch.Tensor
        shape (batch_size, num_ineq_constraints, num_vars)
    hs : torch.Tensor
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
    Gs = Gs.to(device)
    hs = hs.to(device)

    num_vars = Gs.shape[-1]
    batch_size = Gs.shape[0]

    if solver_args is None:
        solver_args = {}

    if As is None or bs is None:
        As = torch.Tensor().to(device).double()
        bs = torch.Tensor().to(device).double()

    Qs = torch.eye(num_vars).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    ps = torch.zeros(batch_size, num_vars).to(device)
    result = QPFunction(verbose=0, **solver_args)(Qs.double(), ps.double(),Gs.double(), hs.double(), As, bs).float()

    if torch.any(torch.isnan(result)):
        print('QP Failed to solve - result is nan == {}!'.format(torch.any(torch.isnan(result))))
        raise Exception('QP Failed to solve')
    return result




def get_safe_action(state_batch, action_batch, t, center, radius, epsilon, modular=False, cbf_info_batch=None):
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
        if cbf_info_batch is not None:
            cbf_info_batch = cbf_info_batch.unsqueeze(0)

    start_time = time()
    Gs, hs = get_tcbf_qp_constraints(state_batch, action_batch, t, center, radius, epsilon)
    build_qp_time = time()
    safe_action_batch = solve_qp(Gs, hs)
    # prCyan('Time to get constraints = {} - Time to solve QP = {} - time per qp = {} - batch_size = {} - device = {}'.format(build_qp_time - start_time, time() - build_qp_time, (time() - build_qp_time) / safe_action_batch.shape[0], Ps.shape[0], Ps.device))
    # The actual safe action is the cbf action + the nominal action
    #final_action = torch.clamp(action_batch + safe_action_batch, self.u_min.repeat(action_batch.shape[0], 1), self.u_max.repeat(action_batch.shape[0], 1))
    final_action = action_batch + safe_action_batch

    return final_action if not expand_dims else final_action.squeeze(0)





if __name__ == "__main__":
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
    epsilon = 0.5 #for reference point

    env = Continuous2DEnv(config)
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).to(device)    

    target_reached = False

    action_rl = torch.tensor([2.0, 0.0], dtype=torch.float32).to(device)

    action = action_rl

    for t in range(200):
        if target_region_radius**2 - ((state[0] - target_region_center[0])**2 + (state[1] - target_region_center[1])**2) >= target_region_radius/2 - 0.2:
            target_reached = True
            print("Target region reached!")

        if target_reached:
            action = action_rl
        else:
            action = get_safe_action(state, action_rl, t, target_region_center, target_region_radius, epsilon)
            print("Action:", action)


        #print("Action:", action)
        action = action.cpu().numpy()
        state, reward, done = env.step(action, gamma(t))
        state = torch.tensor(state, dtype=torch.float32).to(device)
        #print(state)
        if done:
            break
