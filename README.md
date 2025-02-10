# FLIPHAT

Implementation of FLIPHAT algorithm for Jointly Differentially Private Sparse Linear Contextual Bandits - see [paper](https://arxiv.org/abs/2405.14038) (Sunrit Chakraborty, Saptarshi Roy, Debabrota Basu)

The main codes are in `SparseBandit` (see `example.ipynb` for a brief demo). Check `simulation_results.ipynb` which recreates the plots in the paper.

Required packages: (`Python` 3.9.18)
1. `jax` (0.4.23)
2. `numpy` (1.26.3)
3. `sklearn` (1.3.0), `scipy` (1.11.4) - for sparsity agnostic bandit
4. `matplotlib` (3.8.4), `tqdm` (4.66.2) - for graphics
