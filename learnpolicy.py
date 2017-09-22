import numpy as np
from numpy import pi
import GPy
import gym


def madadame(*args):
    print("yet to be implemented")
    return 0




class Policy():
    def __init__(self, **kwargs,params):
        maxU = 1
        pass
        params = params

    def minimize(self,opt,model):
        if opt["model"] == "BFGS":
            self.BFGS()
        else:
            pass


    def BFGS(self):
        x = x0
        fx = fx0
        r = -dfx0
        H = np.eye(x0.shape[0])
        while i < abs(p.length):

    def set_plant(self,gpmodel):
        plant = gpmodel

    def value(self):
        for t in range(prams["max_step"]):
            plant.predict("")
            #caliculate_cost
            plant.





class Plant():
    def __init__(self, **kwargs):
        dt = kwargs["dt"]
        envname = kwargs["envname"]

    def dynamics(self):
        return 0


class Cost():
    def __init__(self, function):
        fcn = function


class Dynmodel():
    def __init__(self, fcn, train, ):
        fcn = fcn
        train = train





def minimize(X, F, p, *args):
    f = (F, X)
    fx, dfx = f(X)
    i = 0
    return X, fx, i


def calcCost(cost, M, S):
    H = M.shape[1]
    L = np.zeros((1, H))
    SL = np.zeros((1, H))
    for i in range(H):
        L[0, i], d1, d2, SL[0, i] = cost.fcn(cost, M[:, i], S[:, :, i])
    sL = np.sqrt(SL)
    return L, sL


def setup(envname):
    dt = 0.05
    T = 5
    H = np.ceil(T / dt)
    maxH = H
    nc = 200
    s = np.array([0.1, 0.1, 0.1, 0.1, 0.01, 0.01]) ** 2
    S0 = np.diag(s)
    mu0 = np.array([0, 0, 0, 0, pi, pi])
    N = 40
    J = 1
    K = 1
    env = gym.make(envname)
    state_dim = env.env.state.shape[0]
    env.reset()
    env_param = {
        "dt": dt,  # [s] sampling time
        "max_time": T,  # [s] prediction time
        "max_step": H,  # number of prediction step
        "maxH": maxH,  # max pred horizon
        "nc": nc,
        "s": s,  # initial state variances
        "S0": S0,
        "mu0": mu0,
        "num_iter": N,  # number of policy searches & GP
        "num_im": J,  # initial (random) trajectories, each of length H
        "K": K,  # number of initial states for which we optimize
        "action_dim": 1,  # original: Dimension of action space
        "state_dim": state_dim,
        "gp_dim": 10,
    }

    plant = Plant(envname=envname, dt=dt, )
    policy = Policy()
    cost = Cost(madadame)
    opt = {
        "length":150,
        "MFEPLS":30,
        "verbosity":1,
        "method":"BFGS",
    }
    return plant, policy, cost, opt, env_param, env


def gaussian(mean, S, n):
    return mean + np.linalg.cholesky(S) * np.random.rand(n)



"""
fantasy = {
    "mean" : np.zeros((1, params["N"])),
    "std" : np.zeros((1, params["N"]))
}
realCost =  np.zeros((1, params["N"]))
M = np.zeros((params["N"], 1))
Sigma = np.zeros((params["N"], 1))
#TODO latentがどこで定義されているかわからん
"""
# 1.initialization
envname = "DoubleCartPole-v0"
plant, policy, cost, opt,params, env = setup(envname)
kernel = GPy.kern.RBF(input_dim=params["action_dim"] + params["state_dim"])

# 2.initial rollout
x = np.zeros((params["max_step"], params["state_dim"]))
#uu = np.random.normal(np.zeros(params["action_dim"]), np.eye(params["action_dim"])) for i in range(["max_step"])
uu = np.random.rand(params["max_step"],params["action_dim"])
for t in range(params["max_step"]):
    xtemp, reward, _, _ = env.step(uu[t,:])
    x[t,:] = xtemp.ravel()
y = x[1:, :] - x[:-1, :]
x[:-1,:]
# 3. Controlled learning (N iterations)
for j in range(params["num_iter"]):
    model = GPy.models.GPRegression(x.T, y.T, kernel)
    opt["fh"] = 1
    policy.minimize(opt,model)

    M[j], Sigma[j] = pred(policy, plant, dynmodel, mu0Sim[:, 0], S0Sim, H)
    fantasy.mean[j], fantasy.std[j] = calc
