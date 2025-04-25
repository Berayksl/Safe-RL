import numpy as np
import cvxpy as cp

def turning_circle_cbf(state, center, Rc, omega_max, v, k=5.0):
    """
    TC-CBF for unicycle with linear/angular velocity inputs.

    Parameters:
        state: np.array [x, y, theta]
        center: (xc, yc)
        Rc: radius of safe circular region
        omega_max: maximum angular velocity (rad/s)
        v: linear speed (used to compute turning radius)
        k: smoothing parameter for smooth min

    Returns:
        ht: CBF value
    """
    x, y, theta = state
    xc, yc = center

    R = v / omega_max  # turning radius

    # Turning circle centers
    xtr, ytr = x + R * np.cos(theta - np.pi/2), y + R * np.sin(theta - np.pi/2)
    xtl, ytl = x + R * np.cos(theta + np.pi/2), y + R * np.sin(theta + np.pi/2)

    # Distances to region center
    dr = np.hypot(xtr - xc, ytr - yc)
    dl = np.hypot(xtl - xc, ytl - yc)

    htr = Rc - dr - R
    htl = Rc - dl - R

    # Smooth min
    ht = - (1 / k) * np.log((np.exp(-k * htr) + np.exp(-k * htl)) / 2)
    return ht


def f_func(state, a_rl, epsilon):
    _, _, theta = state
    v, w = a_rl
    return np.array([
        [np.cos(theta) * v -epsilon * np.sin(theta) * w],
        [np.sin(theta) * v + epsilon * np.cos(theta) * w],
        [w]
    ])

def g_func(state, epsilon):
    _, _, theta = state
    return np.array([
        [np.cos(theta), -epsilon * np.sin(theta)],
        [np.sin(theta), epsilon * np.cos(theta)],
        [0, 1]
    ])


def solve_cbf_qp(h_func, f_func, g_func, state, center, Rc, omega_max, a_rl, epsilon, prev_v, alpha=1.0):
    """
    Solve QP to get control u satisfying the CBF condition.
    
    Parameters:
        h_func: function - returns scalar h(x)
        f_func: function - returns f(x) vector (np.array)
        g_func: function - returns g(x) matrix (np.array)
        state: np.array [x, y, theta]
        center: (xc, yc) - center of safe region
        Rc: float - safe region radius
        omega_max: float - max turning rate
        a_rl: np.array - control input from RL model (can be used as nominal input)
        epsilon: reference point parameter
        prev_v: float - previous linear velocity
        alpha: float - CBF class-K function coefficient

    Returns:
        u_sol: np.array - optimal control that satisfies the CBF
    """

    x = state
    u = cp.Variable(2)

    # CBF value
    k = 5.0 #smoothing parameter
    h_val = h_func(x, center, Rc, omega_max, prev_v, k)

    # Numerical gradient ∇h(x) via finite difference
    eps = 1e-5
    grad_h = np.zeros_like(x)
    for i in range(len(x)):
        x_perturb = x.copy()
        x_perturb[i] += eps
        h_plus = h_func(x_perturb, center, Rc, omega_max, prev_v, k)
        grad_h[i] = (h_plus - h_val) / eps

    # f(x), g(x)
    f = f_func(x, a_rl, epsilon)
    g = g_func(x, epsilon)

    # Lie derivatives
    Lf_h = grad_h @ f
    Lg_h = grad_h @ g


    u_min = np.array([-8, -4])    
    u_max = np.array([8, 4])#might need to change later!

    cbf_constraint = [
        Lf_h + Lg_h @ u + alpha * h_val >= 0,
        #control input constraints:
        u >= u_min,
        u <= u_max
    ]

    objective = cp.Minimize(cp.sum_squares(u))

    prob = cp.Problem(objective, cbf_constraint)
    prob.solve(solver=cp.ECOS)

    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        return u.value
    else:
        print("QP failed:", prob.status)
        return None



if __name__ == "__main__":
    from simulator import Continuous2DEnv

    safe_region_center = [0.0, 0.0]
    safe_region_radius = 5

    #config dictionary for the environment
    config = {
        'init_loc': [0.0, 0.0, 0.0], #initial location of the agent (x, y, theta)
        "width": 8.0,
        "height": 8.0,
        "dt": 0.1,
        "render": True,
        'dt_render': 0.1,
        "goal_location": [10.0, 10.0],
        "goal_size": 0.5,
        "obstacle_location": [10.0, 10.0],
        "obstacle_size": 0.0,
        "safe_region_center": safe_region_center,
        "safe_region_radius": safe_region_radius,
        "randomize_loc": False #whether to randomize the agent location at the end of each episode
    }

    #CBF parameters
    omega_max = 3 #max turning rate (turtlebot = 2.84 rad/s)
    alpha = 5
    
    epsilon = 0.5 #for reference point

    env = Continuous2DEnv(config)
    state = env.reset()


    action_rl = np.array([6.0, 0.0])
    prev_v = 1

    action = action_rl

    for _ in range(400):
        action = solve_cbf_qp(turning_circle_cbf, f_func, g_func, state, safe_region_center, safe_region_radius, omega_max, action_rl, epsilon, prev_v, alpha)
        action = (action[0]+action_rl[0] , action[1] + action_rl[1]) #a_rl + a_cbf
        print("Action:", action)
        state, reward, done = env.step(action)
        if safe_region_radius**2 - ((state[0] - safe_region_center[0])**2 + (state[1] - safe_region_center[1])**2) < 0:
            print('Safe region violated!')
        #prev_v = action[0]
        
        #print(state)
        if done:
            break

