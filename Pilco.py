import numpy as np
import scipy as sp
import GPy
import sys
import datetime

from numpy import pi


# constants
MIN_X_SAMPLES = 6
NUM_XU_SAMPLES = 100


def get_policy_params(fi):
    '''
        returns (A,b)
    '''
    temp = np.atleast_2d(fi)
    return temp[0:1, 0:4], temp[0:1, 4:5]


# calculate policy parameters
def cost_t(x, target_point):
    return np.exp(-np.sum((x - target_point) ** 2))


def cost(fi, gp, target_point, x, vx0):
    '''
        calculates cost for a given policy fi
        based on dynamics model gp
        and trajectory of system variables x
    '''


    temp_c = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        #		print( 'iteration: {0}'.format( i ) )

        if i == 0:
            mxt_1 = x[0:1, :].T
        else:
            mxt_1 = np.atleast_2d(np.mean(x[:i, :], axis=0)).T

        if i < MIN_X_SAMPLES:
            vx_1 = vx0
        else:
            vx_1 = np.cov(x[:i, :], rowvar=0)

        mxu = np.row_stack((mxt_1, A.dot(mxt_1) + b))
        vxu = np.vstack([np.hstack([vx_1, vx_1.dot(A.T)]), np.hstack([A.dot(vx_1), A.dot(vx_1).dot(A.T)])])

        xu = np.random.multivariate_normal(np.ravel(mxu), vxu, NUM_XU_SAMPLES)

        # apply dynamics
        delta = gp.predict(xu)[0]

        # recover xt
        xt_1, ut_1 = (xu[:, 0:4], xu[:, 4:5])
        xt = xt_1 + delta

        # integrate cost
        temp_c[i] = np.apply_along_axis(cost_t, 1, xt, target_point).sum()

    return temp_c.sum()


def run_on_robot(pbot, A, b, max_time, dt_pbot, dt_pilco):
    '''
        function runs pbot with a controller (A,b) for max_time
    '''

    num_samples = int(max_time / dt_pilco)

    x = np.zeros((num_samples + 2, 4))
    u = np.zeros((num_samples + 2, 1))

    t = np.linspace(0.0, dt_pilco / dt_pbot, dt_pbot)
    points = np.zeros((t.shape[0], 4))

    pbot.step(t)
    pbot.get_points(points)

    for i in range(int(max_time / dt_pbot)):
        if i == 0:
            x[0, :] = pbot.x[0, :]
            u[0, :] = pbot.u[0, :]
            x[1, :] = pbot.x[-1, :]
            u[1, :] = pbot.u[-1, :]
        elif i % int(dt_pilco / dt_pbot) == 0:
            x[i + 1, :] = pbot.x[-1, :]
            u[i + 1, :] = pbot.u[-1, :]

        pbot.step(t)
        pbot.get_points(points)

    return x, u

DIM = 6
# init
start_time = datetime.datetime.now()
kernel = GPy.kern.RBF(input_dim=DIM, useGPU=False)

# initialise pendubot
env = gym.make('DoubleCartPole-v0')
    observation = env.reset()
#[x],[theta],[phi],[x_dot],[theta_dot],[phi_dot]
start_state = np.array([0., -0.5*pi, 0., 0.,0.,0])
target_point = np.array([None,None,0., 0.5*pi, 0., 0.,])  # observing (x,y) coordinates of 2 ends

# init policy parameters
fi = np.random.normal(0.0, 1.0, size=(1, 5))

# sampling period for pbot
dt_pbot = 1./50.

MAX_TIME = 5
t_full = np.arange(0.0, MAX_TIME, dt_pbot)
t_dim = t_full.shape[0]
points = np.zeros((t_full.shape[0], 4))

# sampling period for PILCO
dt_pilco = 1. / 20.


# PILCO algo:
# get some random moves ( or initial training )
move_data = np.zeros(t_dim,DIM)
action_data = np.random.normal(0.0,1.0,size = (t_dim,1))
for t in range(t_dim):
    env.render()
    action = action_data[t][0]
    observation, reward, done, info = env.step(action)
    move_data[t] = observation

xu = np.concatenate((move_data,action_data),axis = 1)
xu = xu[:-1,:] #data used for prediction doesn't include last state
delta_x = move_data[1:,:] -move_data[:-1,:]

vx0 = np.eye(target_point.shape[0]) * 0.01
epoch = 0
loop_flag = True
while loop_flag:
    print('epoch: {0}'.format(epoch))
    # fit dynamics model: (xt-1,ut-1) -> delta t
    print('\tfitting dynamics')
    m = GPy.models.GPRegression(xu, delta_x, kernel)

    # minimise cost given policy
    print('\toptimising policy')
    args = (m, target_point, pbot.x, vx0)

    res = sp.optimize.minimize(cost, fi, args, method='bfgs', jac=False)
    if not res.success:
        print('\toptimizer failed: ' + res.message)
        break

    print('\toptmizer iterations: {0}'.format(res.nit))

    fi = res.x
    A, b = get_policy_params(fi)

    print('fi: ', fi)
    break

    # run on robot, records points
    pbot = Pendubot(start_state)
    points_x, points_u = run_on_robot(pbot, A, b)

    # check if we achived required precision
    loop_flag = abs(cost_t(points_x[-1, :], target_point)) < X_PRECISION
    epoch += 1

