import numpy as np
from numpy import pi
import GPy
import gym


def madadame(*args):
    print("yet to be implemented")
    return 0

class Policy():
    p = np.exp

class Plant():
    def __init__(self,**kwargs):
        dt = kwargs["dt"]
        envname = kwargs["envname"]


    def dynamics(self):
        return 0

    def

class Cost():
    def __init__(self,function):
        fcn = function

class Dynmodel():
    def __init__(self,fcn,train,):
        fcn = fcn
        train = train

class Opt():
    def __init__(self):
        length = 150
        MFEPLS = 30
        verbosity = 1
        method = "BFGS"

class Env():




def minimize(X, F, p, *args):
    f = (F, X)
    fx, dfx = f(X)
    return X, fX, i


def calcCost(cost, M, S):
    H = M.shape[1]
    L = np.zeros((1, H))
    SL = np.zeros((1, H))
    for i in range(H):
        L[0, i], d1, d2, SL[0, i] = cost.fcn(cost, M[:, i], S[:, :, i])
    sL = np.sqrt(SL)
    return L, sL



def set_up(envname):
    dt =  0.05,
    T  =  5,
    H =  np.ceil(T/dt)
    maxH = H
    nc = 200
    s = np.array([0.1,0.1,0.1,0.1,0.01,0.01])**2
    S0 = np.diag(s)
    mu0 = np.array([0,0,0,0,pi,pi])
    N = 40
    J = 1
    K = 1
    env_param = {
        "dt" :dt, #[s] sampling time
        "max_step" :T,  # [s] prediction time
        "H" : H, # number of prediction step
        "maxH" :maxH, # max pred horizon
        "nc" : nc,
        "s":s,  #initial state variances
        "S0" :S0,
        "mu0": mu0,
        "num_ps" :N, #number of policy search
        "num_im" :J, #initial (random) trajectories, each of length H
        "K" :K, #number of initial states for which we optimize
        "action_dim": 1, #original: Dimension of action space
        "state_dim":1,
        "gp_dim":10,
    }

    plant = Plant(envname = envname,dt = dt,)
    policy = Policy()
    cost = Cost(madadame)
    dynmodel = Dynmodel()
    opt = Opt()
    return plant, policy, cost, dynmodel, opt,env_param

def gaussian(mean,S,n):
    return mean + np.linalg.cholesky(S) * np.random.rand(n)

def rollout(start,policy,H,plant,cost):
    for i in range(H):
        s = x(i,dyno).T

    return x, y, latent

x = []
y = []
fantasy = {
    "mean" : np.zeros((1, params["N"])),
    "std" : np.zeros((1, params["N"]))
}
realCost =  np.zeros((1, params["N"]))
M = np.zeros((params["N"], 1))
Sigma = np.zeros((params["N"], 1))
#TODO latentがどこで定義されているかわからん

#1.initialization
plant, policy, cost, dynmodel, params= setup("envname")
kernel = GPy.kern.RBF(input_dim = params["gp_dim"])
env = gym.make("DoubleCartPole-v0")
env.reset()
#2.initial rollout
xx = np.zeros((params["max_step"],params["state_dim"]))
uu = np.random.normal(np.zeros(params["action_dim"]),np.eye(np.zeros(param["action_dim"])))
for t in range(params["max_step"]):
    xx[t],reward,_,_ = env.step(uu[t])
yy = xx[1:,:] -x[:-1,:]
xx = np.concatenate((xx,uu),axis= 0)

x.append(xx)
y.append(yy)





opt.fh = 1
policy.p, fx3 = minimize(policy.p, value, opt, mu0Sim, S0Sim,
                         dynmodel, policy, plant, cost, H)

M[j], Sigma[j] = pred(policy, plant, dynmodel, mu0Sim[:, 0], S0Sim, H)
fantasy.mean[j], fantasy.std[j] = calc
