import cvxpy as cp
import numpy as np


def cbf_qp(state, a_rl, x_c, y_c, r, epsilon):
    """Solves the CBF-QP to ensure safety within a circular area.
       x_c, y_c = center of the circle
       r = radius of the circle
       a_rl = control input from the RL model
    """
    x, y, theta = state[0], state[1], state[2] #state matrix

    av_rl = a_rl[0]
    a_theta_rl = a_rl[1]

    av = cp.Variable()
    a_theta = cp.Variable()
    u = np.array([av, a_theta]) #control matrix
    
    # Barrier function h(s) = r^2 - (x - x_c)^2 - (y - y_c)^2
    h = r**2 - (x - x_c)**2 - (y - y_c)**2
    dh_dx = -2 * (x - x_c)
    dh_dy = -2 * (y - y_c)
    dh_dtheta = 0
    
    d_h = np.array([dh_dx, dh_dy, dh_dtheta])

    # Compute f_hat for modified model
    f_hat_x = av_rl * np.cos(theta) - epsilon * a_theta_rl * np.sin(theta)
    f_hat_y = av_rl * np.sin(theta) + epsilon * a_theta_rl * np.cos(theta)
    f_hat_theta = a_theta_rl
    f_hat = np.array([f_hat_x, f_hat_y, f_hat_theta]) #f'(s) matrix

    g = np.array([[np.cos(theta), -epsilon*np.sin(theta)], [np.sin(theta), epsilon*np.cos(theta)], [0, 1]])

    # Constraint ensuring ḣ(s) >= -alpha * h(s) (CBF condition)
    alpha = 10
    #delta = cp.Variable(nonneg=True)  # slack variable
    #K = 100  # weight for the slack variable

    constraint = [
        d_h @ f_hat + d_h @ g @ u + alpha * h >= 0,
        #control input constraints:
        av >= -8.0, av <= 8.0,
        a_theta >= -1.0, a_theta <= 1.0
    ]
    
    # Quadratic objective: minimize deviation from desired control
    objective = cp.Minimize(av**2 + a_theta**2)
    
    # Solve QP
    problem = cp.Problem(objective, constraint)
    problem.solve()

    # print(f"Slack variable delta: {delta.value}")
    
    return av.value, a_theta.value





if __name__ == '__main__':
    # Example usage
    from simulator import Continuous2DEnv

    safe_region_center = [0.0, 0.0]
    safe_region_radius = 5

    #config dictionary for the environment
    config = {
        'init_loc': [0.0, 0.0], #initial location of the agent
        "width": 8.0,
        "height": 8.0,
        "dt": 0.1,
        "render": True,
        "goal_location": [10.0, 10.0],
        "goal_size": 0.5,
        "obstacle_location": [10.0, 10.0],
        "obstacle_size": 0.0,
        "safe_region_center": safe_region_center,
        "safe_region_radius": safe_region_radius,
        "randomize_loc": False #whether to randomize the agent location at the end of each episode
    }
    epsilon = 0.5 #for reference point

    env = Continuous2DEnv(config)
    state = env.reset()

    action_rl = (4.0, 0.5)

    action = action_rl

    for _ in range(400):
        state, reward, done = env.step(action)
        action = cbf_qp(state, action_rl, safe_region_center[0], safe_region_center[1], safe_region_radius, epsilon)
        action = (action[0]+action_rl[0] , action[1] + action_rl[1]) #a_rl + a_cbf

        print('action:', action)
        #print(state)
        if done:
            break

