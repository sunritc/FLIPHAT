import jax.numpy as jnp
import numpy as np
import jax
from scipy.stats import norm
from sklearn import linear_model
from tqdm import tqdm

class SALassoBandit:
    def __init__(self,lambda0,d):
        self.x=[]
        self.r=[]
        self.lambda0 = lambda0
        self.d=d
        self.beta=np.zeros(d)
        
    def choose_a(self,x): 
        est=np.dot(x,self.beta)
        self.action=np.argmax(est)
        self.x.append(x[self.action])
        return(self.action)            
             
    def store_reward(self, rwd):
        self.r.append(rwd)
        
    def update_beta(self,rwd,t):
        if t>5:
            lam_t=2*self.lambda0*np.sqrt((4*np.log(t)+2*np.log(self.d))/t) 
            lasso=linear_model.Lasso(alpha=lam_t, fit_intercept=False, tol=1e-3)
            lasso.fit(self.x,self.r)
            self.beta=lasso.coef_
            
class SA_Bandit():
    '''
    contexts -> (T,K,d)
    rewards -> (T,K)
    max_rewards -> (T,) using best optimal arm at each time 
    beta_star -> (d,) for computing regret
    additional parameters: 
        lambda0
    '''
    def __init__(self, contexts, rewards, max_rewards, beta_star, key, lambda0):
        self.all_contexts = contexts
        self.all_rewards = rewards
        self.T, self.K, self.d = contexts.shape
        self.max_rewards = jnp.max(rewards, axis=1)
        self.beta = jnp.zeros(self.d)
        self.beta_star = beta_star
        self.max_rewards = max_rewards
        self.key = key
        self.lambda0 = lambda0
        self.model = SALassoBandit(lambda0,self.d)
        
    
    
    def play(self):
        actions = jnp.zeros(self.T).astype(jnp.int32)
        chosen_mean_reward = jnp.zeros(self.T)
            
        for t in range(self.T):
            action = self.model.choose_a(self.all_contexts[t])
            reward = self.all_rewards[t, action]
            chosen_mean_reward = chosen_mean_reward.at[t].set(reward)
            actions = actions.at[t].set(action)
            self.model.store_reward(reward)
            if t%200 == 0:
                self.model.update_beta(reward,t+1)
        
        self.actions = actions
        self.mean_rewards = chosen_mean_reward
        self.regret = jnp.cumsum(self.max_rewards - self.mean_rewards)
        