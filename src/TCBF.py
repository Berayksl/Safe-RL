#Time-varying CBF for a circular target region
import numpy as np
import cvxpy as cp



def gamma(t, r=1.0):
    return max(r/2 + 10- 0.05 * t , r/2)

def CBF(state, center, t, radius, gamma_func = gamma):
    #find the distance to the center of the circle:0
    x, y, theta = state
    xc, yc = center
    d = np.sqrt((x - xc)**2 + (y - yc)**2)

    gamma = gamma_func(t, radius)

    cbf_value = gamma - d
    return cbf_value


# def gamma_dt(t):
#     if t <= 160:
#         return -0.03
#     else:
#         return 0.0


def gamma_dt(t):
    return -0.05

def g_func(state, epsilon):
    _, _, theta = state
    return np.array([
        [np.cos(theta), -epsilon * np.sin(theta)],
        [np.sin(theta), epsilon * np.cos(theta)],
        [0, 1]
    ])


def f_func(state, a_rl, epsilon):
    _, _, theta = state
    v, w = a_rl
    return np.array([
        [np.cos(theta) * v -epsilon * np.sin(theta) * w],
        [np.sin(theta) * v + epsilon * np.cos(theta) * w],
        [w]
    ])


def solve_cbf_qp(b_func, g_func, f_func, state, center, t, epsilon, a_rl, radius):
    """
    Solve QP to get control u satisfying the CBF condition.
    """

    x = state
    u = cp.Variable(2)

    x_c, y_c = center

    dx = x[0] - x_c
    dy = x[1] - y_c
    eps = 1e-6
    dist = max(np.sqrt(dx**2 + dy**2), eps)#dist will never be zero

    grad_b = np.array([-dx / dist, -dy / dist, 0.0]) #gradient of the CBF function w.r.t. state

    #find the g and f matrices:
    g = g_func(state, epsilon)
    f = f_func(state, a_rl, epsilon)

    Lgb = np.dot(grad_b, g)  # Lie derivative of b w.r.t. g
    Lfb = np.dot(grad_b, f)  # Lie derivative of b w.r.t. f

    alpha = 5.0

    u_min = np.array([-8, -5])    
    u_max = np.array([8, 5])#might need to change later!

    cbf_constraint = [Lfb + Lgb @ u + gamma_dt(t) + alpha * b_func(state, center, t, radius) >= 0, #CBF condition
        #control input constraints:
        u >= u_min,
        u <= u_max
    ]

    Q = np.eye(2)

    objective = cp.Minimize(cp.quad_form(u, Q))

    prob = cp.Problem(objective, cbf_constraint)
    prob.solve(solver=cp.ECOS)

    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        return u.value
    else:
        print("QP failed:", prob.status)
        return None



if __name__ == "__main__":
    from simulator import Continuous2DEnv

    target_region_center = [2.0, 2.0]
    target_region_radius = 1

    #config dictionary for the environment
    config = {
        'init_loc': [5.0, 2.0, 0.0], #initial location of the agent (x, y, theta)
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

    target_reached = False

    action_rl = np.array([2.0, 0.0])

    action = action_rl

    for t in range(200):
        if target_region_radius**2 - ((state[0] - target_region_center[0])**2 + (state[1] - target_region_center[1])**2) >= target_region_radius/2 - 0.2:
            target_reached = True
            print("Target region reached!")

        if target_reached:
            action = action_rl
        else:
            action = solve_cbf_qp(CBF, g_func, f_func, state, target_region_center, t, epsilon, action_rl, target_region_radius)
            action = (action[0]+action_rl[0] , action[1] + action_rl[1]) #a_rl + a_cbf
            #print("Action:", action)

        #print("Action:", action)
        state, reward, done = env.step(action, gamma(t))
        #print(state)
        if done:
            break

