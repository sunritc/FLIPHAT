import jax.numpy as jnp
import numpy as np
import jax
from SparseBandit.utils import *

class DP_SLCB():
    '''
    # arms = K
    context dimension = d
    privacy parameters: epsilon, delta
    additional parameters: 
        s - sparsity
        C - projection level
        eta - learning parameter
        M0 - for # steps M_l = M0 \log (1 + N_l)
        xmax - L_inf bound for contexts
        bmax - L_1 bound for beta
    '''
    def __init__(self, K, d, epsilon=0.1, delta=0.01, seed=10, **kwargs):
        self.K = K
        self.d = d
        self.epsilon = epsilon
        self.delta = delta
        self.t = 1
        self.beta = np.zeros(d)
        self.contexts = []
        self.rewards = []
        self.episode_length = 1
        self.key = jax.random.PRNGKey(seed)
        self.observed_rewards = []
        
        # check hyperparams else use default
        if 's' in kwargs:
            self.sparsity = kwargs['s']
        else:
            self.sparsity = 20
        if 'eta' in kwargs:
            self.eta_ = kwargs['eta']
        else:
            self.eta_ = 1e-4
        if 'M0' in kwargs:
            self.M0 = kwargs['M0']
        else:
            self.M0 = 0.1
        if 'xmax' in kwargs:
            self.xmax = kwargs['xmax']
        else:
            self.xmax = np.log(K * d)
        if 'bmax' in kwargs:
            self.bmax = kwargs['bmax']
        else:
            self.bmax = 1
        self.C = self.bmax
            
    def take_action(self, X):
        # X is (K,d)
        action = np.argmax(X @ self.beta)
        self.contexts.append(X[action])
        self.t += 1
        return action
        
    def observe_reward(self, y):
        self.observed_rewards.append(y)
        self.rewards.append(y)
        if len(self.rewards) == self.episode_length:
            self.update_beta() # update beta using N-IHT
            self.episode_length = 2 * self.episode_length # next episode length is twice
            self.contexts = [] # forget last episode
            self.rewards = []
        
    def update_beta(self):
        '''
        updates beta using X=contexts and y=rewards saved so far from last episode
        '''
    
        X = np.array(self.contexts)
        y = np.array(self.rewards)
        n = X.shape[0] # sample size = length of last episode
        #print(n)
        
        R_ = self.xmax * self.bmax
        R = R_ + jnp.sqrt(2 * jnp.log(1 + n))
        B = R_ + R
        M = self.M0 * jnp.log(1 + n * self.bmax**2)
        eta = self.eta_ / n
        
        self.beta = NIHT((X,y), self.key, 
                         sparsity=self.sparsity,
                         epsilon=self.epsilon,
                         delta=self.delta, 
                         M=jnp.int32(M), 
                         R=R, 
                         B=B, 
                         eta=eta, 
                         C=self.C)
        self.key, _ = jax.random.split(self.key)
    
    
def get_contexts_rewards(K, d, beta):
    # some function to generate the contexts and rewards
    X = np.random.normal(size=(K,d)) # replace by distribution of contexts
    y = X @ beta + np.random.normal(scale=0.1, size=K)
    return X,y


