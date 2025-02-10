import jax.numpy as jnp
import jax
import numpy as np
from time import time
from tqdm import tqdm
from joblib import Parallel, delayed
from SparseBandit.fliphat_simulation_utils import *
from SparseBandit.sparsity_agnostic import *


'''
     *****   SETTING 1   *****
'''
key = jax.random.PRNGKey(20)
T = 20000
K = 3
d = 400
s = 5
S = jnp.arange(s).astype(jnp.int16)
beta_star = jnp.zeros(d)
a = jax.random.uniform(key, minval=0.5, maxval=1.0, shape=(s,))
signs = jax.random.rademacher(key, shape=(s,))
beta_star = beta_star.at[S].set(a*signs)

eps = [0.5, 1.0, 2.0, 5.0, 10.0]
n_reps = 60

regret_vs_time_eps = np.zeros((n_reps, len(eps)+2, T))

for rep in tqdm(range(n_reps)):
    key, _ = jax.random.split(key)
    
    # draw full contexts and rewards
    contexts = generate_contexts(key, T=T, K=K, d=d, cov_structure='autoregressive', cov_param=0.1, sigma=1.0)
    key, _ = jax.random.split(key)
    rewards, max_rewards, best_actions = generate_rewards(key, contexts=contexts, beta_star=beta_star, noise_sigma=0.1)
    
    # random actions
    random_actions = jax.random.randint(key, minval=0, maxval=K-1, shape=(T,))
    rand_contexts = jnp.array([contexts[i,random_actions[i],:] for i in range(len(random_actions))])
    rand_mean_reward = rand_contexts @ beta_star
    rand_regret = jnp.cumsum(max_rewards - rand_mean_reward)
    regret_vs_time_eps[rep, 0] = rand_regret
    
    # sparsity agnostic
    sa_model = SA_Bandit(contexts, rewards, max_rewards, beta_star, key, lambda0=1.)
    sa_model.play()
    regret_vs_time_eps[rep, 1] = rand_regret = sa_model.regret
    
    # NIHT based SLCB
    key, _ = jax.random.split(key)
    def f(i):
        bandit = SLCB(contexts, rewards, max_rewards, beta_star, key, s=10, epsilon=eps[i], delta=0.01, eta=1e-4, M=0.16)
        bandit.play()
        return bandit.regret

    results = Parallel(n_jobs=6)(delayed(f)(i) for i in range(5))
    results = np.array(results)
    regret_vs_time_eps[rep, 2:] = results
    
# matrix -> axis=0 is repetitions (60), axis=1 is algo (random, SA, eps0.5, 1.0, 2.0, 5.0, 10.0), axis=2 is time horizon T
np.save('simulations/result_AR1e-1-d400-s5-T40000-K3.npy', regret_vs_time_eps) 


'''
     *****   SETTING 2   *****
'''
d_list = np.linspace(400, 4000, 10).astype(np.int32)
eps = [0.5, 1.0, 2.0, 5.0]
T = 10000
K = 3
s = 5
key = jax.random.PRNGKey(20)

S = jnp.arange(s).astype(jnp.int16)
a = jax.random.uniform(key, minval=0.5, maxval=1., shape=(s,))
signs = jax.random.rademacher(key, shape=(s,))

n_reps = 50
regret_vs_dim_eps = np.zeros((len(d_list), n_reps, len(eps)))

for j, d in enumerate(d_list):
    print(f'------ NOW WORKING d={d}-------')

    # draw beta
    key = jax.random.PRNGKey(20)

    beta_star = jnp.zeros(d)
    beta_star = beta_star.at[S].set(a*signs)

    for rep in tqdm(range(n_reps)):
        key, _ = jax.random.split(key)
        # draw full contexts and rewards
        contexts = generate_contexts(key, T=T, K=K, d=d, cov_structure='autoregressive', cov_param=0.1, sigma=1.0)
        key, _ = jax.random.split(key)
        rewards, max_rewards, best_actions = generate_rewards(key, contexts=contexts, beta_star=beta_star, noise_sigma=0.1)

        key, _ = jax.random.split(key)
        def f(i):
            bandit = SLCB(contexts, rewards, max_rewards, beta_star, key, s=10, epsilon=eps[i], delta=0.01, eta=1e-4, M=0.15)
            bandit.play()
            return bandit.regret

        results = Parallel(n_jobs=4)(delayed(f)(i) for i in range(4))
        results = np.array(results)[:,-1]
        regret_vs_dim_eps[j, rep] = results # len-3 vector
    np.save('simulations/result_AR1e-1-s5-T40000-K3-effect_d.npy', regret_vs_dim_eps)
    
    
'''
     *****   SETTING 3   *****
'''
key = jax.random.PRNGKey(20)
T = 20000
K = 3
d = 400
s = 5
S = jnp.arange(s).astype(jnp.int16)

beta_star = jnp.zeros(d)
a = jax.random.uniform(key, minval=0.5, maxval=1.0, shape=(s,))
signs = jax.random.rademacher(key, shape=(s,))
beta_star = beta_star.at[S].set(a*signs)

eps = [0.5, 1.0, 2.0, 5.0, 10.0]
n_reps = 60

regret_vs_time_eps = np.zeros((n_reps, len(eps)+2, T))

for rep in tqdm(range(n_reps)):
    key, _ = jax.random.split(key)
    
    # draw full contexts and rewards
    contexts = generate_contexts(key, T=T, K=K, d=d, cov_structure='autoregressive', cov_param=0.1, sigma=1.0)
    key, _ = jax.random.split(key)
    rewards, max_rewards, best_actions = generate_rewards_unif(key, contexts=contexts, beta_star=beta_star, bound=0.1)
    
    # random actions
    random_actions = jax.random.randint(key, minval=0, maxval=K-1, shape=(T,))
    rand_contexts = jnp.array([contexts[i,random_actions[i],:] for i in range(len(random_actions))])
    rand_mean_reward = rand_contexts @ beta_star
    rand_regret = jnp.cumsum(max_rewards - rand_mean_reward)
    regret_vs_time_eps[rep, 0] = rand_regret
    
    # sparsity agnostic
    sa_model = SA_Bandit(contexts, rewards, max_rewards, beta_star, key, lambda0=1.)
    sa_model.play()
    regret_vs_time_eps[rep, 1] = rand_regret = sa_model.regret
    
    # NIHT based SLCB
    key, _ = jax.random.split(key)
    def f(i):
        bandit = SLCB(contexts, rewards, max_rewards, beta_star, key, s=10, epsilon=eps[i], delta=0.01, eta=1e-4, M=0.16)
        bandit.play()
        return bandit.regret

    results = Parallel(n_jobs=6)(delayed(f)(i) for i in range(5))
    results = np.array(results)
    regret_vs_time_eps[rep, 2:] = results
    
# matrix -> axis=0 is repetitions (60), axis=1 is algo (random, SA, eps0.5, 1.0, 2.0, 5.0, 10.0), axis=2 is time horizon T
np.save('simulations/result_AR1e-1-d400-s5-T40000-K3-unif_noise.npy', regret_vs_time_eps) 



'''
     *****   SETTING 4   *****
'''
key = jax.random.PRNGKey(20)
T = 10000
K = 3
d = 400
s = 10
S = jnp.arange(s).astype(jnp.int16)

beta_star = jnp.zeros(d)
a = jax.random.uniform(key, minval=0.5, maxval=1.0, shape=(s,))
signs = jax.random.rademacher(key, shape=(s,))
beta_star = beta_star.at[S].set(a*signs)
eps = [0.5, 1.0, 2.0, 5.0]
s_list = (np.arange(12, dtype=np.int16)+1)*5
n_reps = 50

regret_vs_s_eps = np.zeros((n_reps, len(eps), len(s_list)))


for rep in tqdm(range(n_reps)):
    key, _ = jax.random.split(key)

    # draw full contexts and rewards
    contexts = generate_contexts(key, T=T, K=K, d=d, cov_structure='autoregressive', cov_param=0.1, sigma=1.0)
    key, _ = jax.random.split(key)
    rewards, max_rewards, best_actions = generate_rewards(key, contexts=contexts, beta_star=beta_star, noise_sigma=0.1)

    # NIHT based SLCB
    for j, eps_ in enumerate(eps):
        key, _ = jax.random.split(key)
        def f(i):
            bandit = SLCB(contexts, rewards, max_rewards, beta_star, key, s=s_list[i], epsilon=eps_, delta=0.01, eta=1e-4, M=0.16)
            bandit.play()
            return bandit.regret[-1]

        results = Parallel(n_jobs=6)(delayed(f)(i) for i in range(12))
        regret_vs_s_eps[rep, j, :] = np.array(results)
    np.save('simulations/result_tuning_s_new.npy', regret_vs_s_eps)
    
    
'''
     *****   SETTING 5   *****
'''

key = jax.random.PRNGKey(20)
T = 10000
d = 400
s = 5
S = jnp.arange(s).astype(jnp.int16)
K_list = (np.arange(12, dtype=np.int16)+1)*2

beta_star = jnp.zeros(d)
a = jax.random.uniform(key, minval=0.5, maxval=1.0, shape=(s,))
signs = jax.random.rademacher(key, shape=(s,))
beta_star = beta_star.at[S].set(a*signs)
eps = [0.5, 1.0, 2.0, 5.0]
n_reps = 50

regret_vs_K_eps = np.zeros((n_reps, len(eps), len(K_list)))


for rep in tqdm(range(n_reps)):
    key, _ = jax.random.split(key)
    
    for i, K in enumerate(K_list):
        # draw full contexts and rewards
        contexts = generate_contexts(key, T=T, K=K, d=d, cov_structure='autoregressive', cov_param=0.1, sigma=1.0)
        key, _ = jax.random.split(key)
        rewards, max_rewards, best_actions = generate_rewards(key, contexts=contexts, beta_star=beta_star, noise_sigma=0.1)

        # NIHT based SLCB
        def f(j):
            bandit = SLCB(contexts, rewards, max_rewards, beta_star, key, s=10, epsilon=eps[j], delta=0.01, eta=1e-4, M=0.16)
            bandit.play()
            return bandit.regret[-1]

        results = Parallel(n_jobs=6)(delayed(f)(j) for j in range(4))
        regret_vs_K_eps[rep, :, i] = np.array(results)
    np.save('simulations/result_effect_K.npy', regret_vs_K_eps)