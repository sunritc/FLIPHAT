import jax.numpy as jnp
import numpy as np
import jax
from SparseBandit.utils import *

# equicorrelated Sigma - cholesky decomposition 
# rho = 0 -> iid

def EQ(d, rho=0.0, sigma=1.0):
    # check rho
    if rho > 1 or rho < -1/(d-1):
        raise ValueError('rho incompatible for equicorrelated matrix')
    Sigma = np.ones((d,d)) * rho
    np.fill_diagonal(Sigma, 1)
    # cholesky
    return sigma * jnp.linalg.cholesky(Sigma)

# autoregressive Sigma - cholesky decomposition

def AR(d, phi=0.0, sigma=1.0):
    a = jnp.arange(d)
    Sigma = phi ** jnp.abs(a.reshape(-1,1) - a)
    return sigma * jnp.linalg.cholesky(Sigma)

def generate_contexts(key, T, K, d, cov_structure='autoregressive', cov_param=0.0, sigma=1.0):
    '''
    generates all contexts to be fed to bandit (iid across arms and time)
    T -> time horizon
    K -> number of arms
    d -> context dimension
    cov_structure -> 'autoregressive' or 'equicorrelated'
    cov_params -> rho (for EQ) or phi (for AR)
    sigma -> context scaling
    returns (T, K, d) array
    '''
    X = jax.random.normal(key, shape=(T,K,d))
    if cov_structure == 'autoregressive':
        phi = cov_param
        L = AR(d, phi, sigma)
    elif cov_structure == 'equicorrelated':
        rho = cov_param
        L = EQ(d, rho, sigma)
    contexts = X @ L
    return contexts
    
def generate_rewards(key, contexts, beta_star, noise_sigma=0.1):
    '''
    generate rewards to be fed to bandit using lienar model
    contexts and beta_star are input
    noise_sigma -> sd of the Gaussian noise
    
    returns (T,K) array of all possible rewards
    '''
    if contexts.shape[-1] != len(beta_star):
        raise ValueError('context dim incompatible with beta')
    rewards = contexts @ beta_star # (T,K,d)x(d,1) -> (T,K,)
    best_actions = jnp.argmax(rewards, axis=1)
    max_rewards = jnp.max(rewards, axis=1)
    rewards = rewards + jax.random.normal(key, shape=rewards.shape) * noise_sigma
    return rewards, max_rewards, best_actions

def generate_rewards_unif(key, contexts, beta_star, bound=0.5):
    '''
    generate rewards to be fed to bandit using lienar model
    contexts and beta_star are input
    U(-bound, bound) noise added
    
    returns (T,K) array of all possible rewards
    '''
    if contexts.shape[-1] != len(beta_star):
        raise ValueError('context dim incompatible with beta')
    rewards = contexts @ beta_star # (T,K,d)x(d,1) -> (T,K,)
    best_actions = jnp.argmax(rewards, axis=1)
    max_rewards = jnp.max(rewards, axis=1)
    rewards = rewards + jax.random.uniform(key, shape=rewards.shape, minval=-bound, maxval=bound)
    return rewards, max_rewards, best_actions

class SLCB():
    '''
    contexts -> (T,K,d)
    rewards -> (T,K)
    max_rewards -> (T,) using best optimal arm at each time 
    beta_star -> (d,) for computing regret
    additional parameters: 
        s - sparsity
        epsilon, delta - privacy
        C - projection level
        eta - learning parameter
        R - 
    '''
    def __init__(self, contexts, rewards, max_rewards, beta_star, key, **kwargs):
        self.all_contexts = contexts
        self.all_rewards = rewards
        self.T, self.K, self.d = contexts.shape
        self.max_rewards = jnp.max(rewards, axis=1)
        self.beta = jnp.zeros(self.d)
        self.beta_star = beta_star
        self.max_rewards = max_rewards
        bmax = jnp.sum(jnp.abs(beta_star))
        self.key = key
        xmax = jnp.sqrt(2*jnp.log(self.d))
        self.xmax = xmax
        self.bmax = bmax
        
        # check hyperparams else use default
        if 's' in kwargs:
            self.sparsity = kwargs['s']
        else:
            self.sparsity = 20
        if 'epsilon' in kwargs:
            self.epsilon = kwargs['epsilon']
        else:
            self.epsilon = 1.0
        if 'delta' in kwargs:
            self.delta = kwargs['delta']
        else:
            self.delta = 0.01
        if 'C' in kwargs:
            self.C = kwargs['C']
        else:
            self.C = bmax
        if 'eta' in kwargs:
            self.eta_ = kwargs['eta']
        else:
            self.eta_ = 1e-3
        self.R_ = xmax*bmax
        # use R = R_ + sqrt(2*log(n))
        # use B = R + R_
        self.key = key
        if 'M' in kwargs:
            self.M_ = kwargs['M']
        else:
            self.M_ = 1
        # use M = M_ * log(bmax^2 * n)
        if 'M_scale' in kwargs:
            self.M_scale = kwargs['M_scale']
        else:
            self.M_scale = 1.0
            
    def update_params(self, X, r):
        '''
        X, r -> chosen contexts and observed rewards over last episode
        updates beta
        '''
        n = X.shape[0] # sample size
        
        R = self.R_ + jnp.sqrt(2 * jnp.log(1 + n))
        B = self.R_ + R
        self.M_ = self.M_ * self.M_scale
        M = self.M_ * jnp.log(1 + n * self.bmax**2)
        eta = self.eta_ / n
        
        self.beta = NIHT((X,r), self.key, 
                         sparsity=self.sparsity,
                         epsilon=self.epsilon,
                         delta=self.delta, 
                         M=jnp.int32(M), 
                         R=R, 
                         B=B, 
                         eta=eta, 
                         C=self.C)
        self.key, _ = jax.random.split(self.key)
        self.l = self.l * 2
    
    
    def play(self):
        actions = jnp.zeros(self.T).astype(jnp.int32)
        chosen_mean_reward = jnp.zeros(self.T)
        L = jnp.int32(jnp.log2(self.T))
        if self.T <= 2**L - 1:
            L = L-1
        elif self.T > 2**(L+1)-2:
            L = L+1
        
        # first play episode 0 - random action
        self.l = 1
        t = 0
        a = jax.random.randint(self.key, minval=0, maxval=self.K-1, shape=(1,))[0]
        #print(a)
        actions = actions.at[t].set(a)
        chosen_contexts = self.all_contexts[t, a].reshape((1, self.d))
        obs_rewards = self.all_rewards[t,a].reshape(1,)
        chosen_mean_reward = chosen_mean_reward.at[t].set(jnp.dot(self.all_contexts[t, a], self.beta_star))
        self.update_params(chosen_contexts, obs_rewards)
        
        
        def play_episode(l, val):
            # episode l t = 2^l-1 to min(2^{l+1}-2, T) starting at l=1 (i.e., t=1,2)
            # last episode was t = 2^{l-1}-1 to 2^l-2
            # l goes from 1 to L where L is max s.t. 2^L-1 < T <= 2^{L+1}-2
            chosen_mean_reward, actions = val
            chosen_contexts = jnp.zeros((self.l,self.d))
            obs_rewards = jnp.zeros((self.l,))
            beta = self.beta
            
            def inside_episode(s, val):
                # s indexes from 0 to 2**l-1 within the episode
                # corresponds to true t = 2**l-1+s
                chosen_contexts, obs_rewards, chosen_mean_reward, actions, beta, episode = val
                t = 2**episode - 1 + s
                a = jnp.argmax(self.all_contexts[t] @ beta)
                actions = actions.at[t].set(a)
                chosen_mean_reward = chosen_mean_reward.at[t].set(jnp.dot(self.all_contexts[t, a], self.beta_star))
                chosen_contexts = chosen_contexts.at[s].set(self.all_contexts[t, a])
                obs_rewards = obs_rewards.at[s].set(self.all_rewards[t, a])
                
                return chosen_contexts, obs_rewards, chosen_mean_reward, actions, beta, episode
            
            chosen_contexts, obs_rewards, chosen_mean_reward, actions, beta, l = jax.lax.fori_loop(0, 2**l-1, inside_episode, (chosen_contexts,obs_rewards,chosen_mean_reward,actions,beta,l))
            self.update_params(chosen_contexts, obs_rewards)
            return chosen_mean_reward, actions
        
        for l in range(1, L+1):
            chosen_mean_reward, actions = play_episode(l, (chosen_mean_reward, actions))

        self.actions = actions
        self.mean_rewards = chosen_mean_reward
        self.regret = jnp.cumsum(self.max_rewards - self.mean_rewards)
        
    