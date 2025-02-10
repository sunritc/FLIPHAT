import jax.numpy as jnp
import jax

def sparsify(v, S):
    '''
    given vector v in R^d and S subset of [d]
    returns u in R^d, which matches v exactly on S
    and 0 elsewhere
    '''
    u = jnp.zeros_like(v).astype(v.dtype)
    u = u.at[S].set(v[S])
    return u

def peel(v, key, sparsity, epsilon, delta, lmbda):
    '''
    perform peeling on v - private top s selection
    key -> control randomness (jax)
    sparsity -> parameter s (upper bound on true sparsity)
    epsilon, delta -> DP parameters
    lmbda -> Laplace noise scale
    returns: a vector (same dim as v) with privatized top sparsity coordinates (others 0)
    '''
    xi = lmbda * (2 * jnp.sqrt(3*sparsity*jnp.log(1/delta))) / epsilon
    S = jnp.zeros(sparsity).astype(jnp.int32)
    abs_v = jnp.abs(v)
    def f(i, val):
        S, abs_v, key = val
        w = xi * jax.random.laplace(key, shape=(len(v),))
        key, _ = jax.random.split(key)
        j_star = jnp.argmax(abs_v + w)
        S = S.at[i].set(j_star)
        abs_v = abs_v.at[j_star].set(-jnp.inf)
        return (S, abs_v, key)
    S, _, key = jax.lax.fori_loop(0, sparsity, f, (S, abs_v, key))
    u = sparsify(v, S)
    u = u.at[S].set(u[S] + xi * jax.random.laplace(key, shape=(sparsity,)))
    return u

def L1_proj(v, C):
    '''
    returns Euclidean projection of v on L1 ball of radius C
    given v and C, return u = argmin ||v-u|| over u s.t. |u|_1 <= C
    '''
    abs_v = jnp.abs(v)
    def true_fn(v, abs_v, C):
        return v
    def false_fn(v, abs_v, C):
        abs_v_sorted = -jnp.sort(-abs_v)
        partial_sum = jnp.cumsum(abs_v_sorted)
        a = 1 * ((abs_v_sorted - (partial_sum - C)/jnp.arange(1, len(v)+1)) > 0)
        rho = jnp.max(jnp.nonzero(a, size=len(a))[0])
        tau = (partial_sum[rho] - C) / (rho + 1)
        s = 2 * (v>0) - 1
        return jnp.maximum(abs_v - tau, 0) * s
    proj_v = jax.lax.cond(jnp.sum(abs_v) <= C, true_fn, false_fn, v, abs_v, C)
    return proj_v

def grad_sq_loss(XtX, Xty, theta):
    '''
    L = ||X theta - Y||^2
    grad = 2 X.T (X theta - Y) = 2 * (XtX theta - Xty)
    '''
    return 2*(XtX @ theta - Xty)

def clip(x, R):
    return x * jnp.minimum(1, R/jnp.abs(x))

def NIHT(data, 
         key,
         sparsity,
         epsilon, 
         delta,
         M, R, B, eta, C):
    '''
    Noisy Iterative Hard Thresholding
    data -> (X, y) X is (n,d), y is (n,)
    key -> control randomness (in peeling subroutine)
    sparsity -> s
    epsilon, delta -> DP parameters
    M -> number of iterations, should scale like log n
    R -> truncation level (for y), typically sigma sqrt(2 log n)
    B -> noise scale, typically B = R + xmax*bmax
    eta -> step size (tricky to tune, start small) - in bandit, pass eta_0 / n
    C -> L1 projection radius (for theta)
    
    note: for other loss function, change the grad function above and use it here
    for squared error loss, easier to compute XtX and Xty once and use it everytime
    '''
    X, y = data
    y_clipped = clip(y, R)
    A1 = X.T @ X # (d,d)
    A2 = X.T @ y_clipped # (d,)
    n, d = X.shape
    theta = jnp.zeros(d)
    def f(i, val):
        theta, key = val
        theta = theta - eta * grad_sq_loss(A1, A2, theta)
        theta = peel(theta, key, sparsity, epsilon/M, delta/M, eta*B)
        key, _ = jax.random.split(key)
        theta = L1_proj(theta, C)
        return (theta, key)
    theta, _ = jax.lax.fori_loop(0, M, f, (theta, key))
    return theta

def NIHT_np(data, 
         sparsity,
         epsilon, 
         delta,
         M, R, B, eta, C):
    '''
    Noisy Iterative Hard Thresholding
    data -> (X, y) X is (n,d), y is (n,)
    sparsity -> s
    epsilon, delta -> DP parameters
    M -> number of iterations, should scale like log n
    R -> truncation level (for y), typically sigma sqrt(2 log n)
    B -> noise scale, typically B = R + xmax*bmax
    eta -> step size (tricky to tune, start small) - in bandit, pass eta_0 / n
    C -> L1 projection radius (for theta)
    
    note: for other loss function, change the grad function above and use it here
    for squared error loss, easier to compute XtX and Xty once and use it everytime
    '''
    X, y = data
    y_clipped = clip(y, R)
    A1 = X.T @ X # (d,d)
    A2 = X.T @ y_clipped # (d,)
    n, d = X.shape
    theta = jnp.zeros(d)
    def f(i, val):
        theta, key = val
        theta = theta - eta * grad_sq_loss(A1, A2, theta)
        theta = peel(theta, key, sparsity, epsilon/M, delta/M, eta*B)
        key, _ = jax.random.split(key)
        theta = L1_proj(theta, C)
        return (theta, key)
    theta, _ = jax.lax.fori_loop(0, M, f, (theta, key))
    return theta