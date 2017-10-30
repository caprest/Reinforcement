import numpy as np
from numpy import pi
import GPy
import gym



def squashing(x):
    return (9*np.sin(x) + np.sin(3*x))/8



class saturating_cost(x):
    dim = 4
    mu = np.zeros(dim)
    t_inv = np.zeros()
    target = np.zeros(dim)
    a = 1
    dif_cost_mu = np.zeros()
    dif_cost_sigma = np.zeros()
    def __init__(self,mu,a):
        self.mu = mu
        self.a = a

    def distance(self,x):
        return np.sum(np.matmul((x-self.target).T,self.T_inv,x-self.target))

    def forward(self,x):
        return 1 - np.exp(- 0.5 * self.distance(x))

    def expectation(self,mu,Sigma):
        temp = np.eye(self.dim)+np.matmul(Sigma,self.T_inv)
        S1 = np.matmul(self.T_inv,temp)
        dif = mu - self.target
        return 1 - np.det(temp)*np.exp(-0.5 * np.matmul(dif.T,S1,dif))

    def calc_dif_cost(self,mu,Sigma):
        temp = np.eye(self.dim)+np.matmul(Sigma,self.T_inv)
        S1 = np.matmul(self.T_inv,temp)
        exp = self.expectation(mu,Sigma)
        dif = mu - self.target
        self.dif_mu = -1.0 * exp * self.matmul(dif.T,S1)
        self.dif_sigma = 0.5 * exp * np.matmul((np.matmul(S1,dif,dif.T)-np.eye(self.dim)),S1)



    def dif_cost(self,T,num_param):
        ret = np.zeros((T+1,1,num_param))
        for t in range(0,T+1):
            self.dif_exp(mu,Sigma)
            ret[t,0,:] = dif_cost_mu[t]@self.dif_mu[t] + dif_cost_sigma@self.dif_cost







class RBF():
    def __init__(self,input_dim,kernel_dim,w = None,mu =None,lam =None):
        kernel_dim = 5
        input_dim = 7
        if not w:
            w = np.ones((1,k))
        else:
            w = w
        if not mu:
            mu = np.zeros((kernel_dim,1,input_dim))
        else:
            mu = mu
        if not lam:
            lam = np.tile(np.eye(input_dim),(kernel_dim,1)).reshape(kernel_dim,input_dim,input_dim)
        else:
            lam = lam

    def prop(self,x):
        dif = self.mu - x.reshape(self.input_dim,1) #broadcasting
        return np.matmul(self.w,np.exp(np.matmul(dif.transpose(0,2,1),np.matmul(self.lam,dif)).reshape(-1)))


def propagate_with_d():
    """
     function [Mnext, Snext, dMdm, dSdm, dMds, dSds, dMdp, dSdp] = ...
     propagated(m, s, plant, dynmodel, policy)
    
     *Input arguments:*
    
       m                 mean of the state distribution at time t           [D x 1]
       s                 covariance of the state distribution at time t     [D x D]
       plant             plant structure
       dynmodel          dynamics model structure
       policy            policy structure
    
     *Output arguments:*
    
       Mnext             predicted mean at time t+1                         [E x 1]
       Snext             predicted covariance at time t+1                   [E x E]
       dMdm              output mean wrt input mean                         [E x D]
       dMds              output mean wrt input covariance matrix         [E  x D*D]
       dSdm              output covariance matrix wrt input mean        [E*E x  D ]
       dSds              output cov wrt input cov                       [E*E x D*D]
       dMdp              output mean wrt policy parameters                  [E x P]
       dSdp              output covariance matrix wrt policy parameters  [E*E x  P]    
    """




def gTrig(mean,cov,idx,scale_vec = None):
    """
    saturating function (limiting output like  $ u_{max}sin(\pi^\{tilde}(x))$
     m     mean vector of Gaussian                                    [ d       ]
    v     covariance matrix                                          [ d  x  d ]
    i     vector of indices of elements to augment                   [ I  x  1 ]
    たぶんなんかのフラグなのだが、よくわからん
    つまるところ角度を指定しなければならないところは、拡張する必要がある(sin,cos)で表現するため
    e     (optional) scale vector; default: 1                        [ I  x  1 ]
    ->そもそも thetaをsin\theta cos\thetaで書いているっぽい
    出力としてはp(sin\theta,cos\theta)をめざす？
    %   M     output means                                              [ 2I       ]
    %   V     output covariance matrix                                  [ 2I x  2I ]
    %   C     inv(v) times input-output covariance                      [ d  x  2I ]
    %   dMdm  derivatives of M w.r.t m                                  [ 2I x   d ]
    %   dVdm  derivatives of V w.r.t m                                  [4II x   d ]
    %   dCdm  derivatives of C w.r.t m                                  [2dI x   d ]
    %   dMdv  derivatives of M w.r.t v                                  [ 2I x d^2 ]
    %   dVdv  derivatives of V w.r.t v                                  [4II x d^2 ]
    %   dCdv  derivatives of C w.r.t v                                  [2dI x d^2 ]
    :return: 
    """
"""
d = length(m); I = length(i); Ic = 2*(1:I); Is = Ic-1;
if nargin == 3, e = ones(I,1); else e = e(:); end; 
ee = reshape([e e]',2*I,1);
mi(1:I,1) = m(i); vi = v(i,i); vii(1:I,1) = diag(vi);     % short-hand notation

M(Is,1) = e.*exp(-vii/2).*sin(mi); M(Ic,1) = e.*exp(-vii/2).*cos(mi);    % mean

lq = -bsxfun(@plus,vii,vii')/2;
 q = exp(lq);
U1 = (exp(lq+vi)-q).*sin(bsxfun(@minus,mi,mi'));
U2 = (exp(lq-vi)-q).*sin(bsxfun(@plus,mi,mi'));
U3 = (exp(lq+vi)-q).*cos(bsxfun(@minus,mi,mi'));
U4 = (exp(lq-vi)-q).*cos(bsxfun(@plus,mi,mi'));
V(Is,Is) = U3 - U4; V(Ic,Ic) = U3 + U4; V(Is,Ic) = U1 + U2; 
V(Ic,Is) = V(Is,Ic)'; V = ee*ee'.*V/2;                               % variance
"""

    d = mean.shape[0]
    I = len(idx)
    Ic = 2*np.arange(I)
    Is = Ic-1
    if not scale_vec:
        scalevec = np.ones(I,1)
    scale_vec2 = np.tile(scale_vec,(2,1)).reshape((2*I,1))
    idx_mean = mean[i,0]
    idx_cov = mean[i,:][:,i]
    idx_cov2 = np.diag(idx_cov)









class Policy():
    def __init__(self, **kwargs,env_params):
        maxU = 1
        pass
        params = env_params
        gamma = 1
        m0 = env_params["mu0"]
        S0 = env_params["S0"]
        k = 5 # kernel_dim
        self.w = np.ones((1,k))
        self.mu = np.zeros((,1,env_params["state_dim"]))
        self.lam = np.eye((1,env_params["state_dim"]))
        self.rbf = RBF(input_dim = env_params["state_dim"],kernel_dim  = k)
        self.num_param = 10 #temp

    def get_policy(self,x):
        dif = x -self.mu
        self.w * exp(dif.T * self.lam * dif)


    def minimize(self,opt,model):
        if opt["model"] == "BFGS":
            self.BFGS()
        else:
            pass
    def policy_grad(self,dif_cost):
        assert dif_cost.shape ==  (self.T+1,1,self.num_param)
        gradient = np.sum(dif_cost,axis=0)
        #\sum_t=0^T(d/d\phi(E(c_t))
        return gradient

    def BFGS(self):
        x = x0
        fx = fx0
        r = -dfx0
        H = np.eye(x0.shape[0])
        while i < abs(p.length):

    def set_plant(self,gpmodel):
        plant = gpmodel

    def value(self):
        m = self.m0
        S = self.S0
        for t in range(env_prams["max_step"]):
            plant.predict()
            #caliculate_cost





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
