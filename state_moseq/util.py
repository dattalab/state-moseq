import jax.numpy as jnp
import jax.random as jr
import jax
import optax
import tqdm
import h5py
import numpy as np
from functools import partial
from tensorflow_probability.substrates.jax import distributions as tfd
from jaxtyping import Array, Float, Int, PyTree, Bool
from typing import Tuple, Union, Callable, Dict, Optional, List
from scipy.optimize import linear_sum_assignment
from dynamax.utils.optimize import run_gradient_descent
from jax.scipy.linalg import cho_factor, cho_solve

na = jnp.newaxis


@jax.vmap
def logits_to_probs(
    logits: Float[Array, "n_categories-1"]
) -> Float[Array, "n_categories"]:
    """Convert logits to probabilities."""
    logits = jnp.concatenate([logits, jnp.zeros(1)])
    return jax.nn.softmax(logits)


@jax.vmap
def probs_to_logits(
    probs: Float[Array, "n_categories"],
    pseudo_count: Float = 1e-8,
) -> Float[Array, "n_categories-1"]:
    """Convert probabilities to logits."""
    log_probs = jnp.log(probs + pseudo_count)
    return log_probs[:-1] - log_probs[-1]


def normal_inverse_gamma_posterior(
    seed: Float[Array, "2"],
    mean: Float,
    sigmasq: Float,
    n: Int,
    lambda_: Float,
    alpha: Float,
    beta: Float,
) -> Tuple[Float, Float]:
    """
    Sample posterior mean and variance given normal-inverse gamma prior.

    Args:
        seed: random seed
        mean: sample mean
        sigmasq: sample variance
        n: number of data points
        lambda_: strength of prior
        alpha: inverse gamma shape parameter
        beta: inverse gamma rate parameter

    Returns:
        mu: posterior mean
        sigma: posterior variance
    """
    seeds = jr.split(seed, 2)
    mean = jnp.nan_to_num(mean)
    sigmasq = jnp.nan_to_num(sigmasq)
    lambda_n = lambda_ + n
    alpha_n = alpha + n / 2
    beta_n = beta + 0.5 * n * sigmasq + 0.5 * n * lambda_ * (mean**2) / lambda_n
    sigma = sample_inv_gamma(seeds[0], alpha_n, beta_n)
    mu = jr.normal(seeds[1]) * jnp.sqrt(sigmasq / lambda_n) + mean
    return mu, sigma


def center_embedding(n: int) -> Float[Array, "n n-1"]:
    """Generate an orthonormal matrix that embeds R^(n-1) into the space of 0-sum vectors in R^n."""
    # using numpy.linalg.svd because jax version crashes on windows
    X = jnp.tril(jnp.ones((n, n)), k=-1)[1:]
    X = jnp.eye(n)[1:] - X / X.sum(1)[:, na]
    X = X / jnp.sqrt((X**2).sum(1))[:, na]
    return X.T


def lower_dim(arr, axis=0):
    """Lower dimension in specified axis by projecting onto the space of 0-sum vectors."""
    arr = jnp.moveaxis(arr, axis, 0)
    k, *shape = arr.shape
    arr = arr.reshape(k, -1)
    arr = center_embedding(k).T @ arr
    arr = arr.reshape(k - 1, *shape)
    arr = jnp.moveaxis(arr, 0, axis)
    return arr


def raise_dim(arr, axis=0):
    """Raise dimension in specified axis by embedding into the space of 0-sum vectors."""
    arr = jnp.moveaxis(arr, axis, 0)
    k, *shape = arr.shape
    arr = arr.reshape(k, -1)
    arr = center_embedding(k + 1) @ arr
    arr = arr.reshape(k + 1, *shape)
    arr = jnp.moveaxis(arr, 0, axis)
    return arr


def sample_multinomial(
    seed: Float[Array, "2"],
    n: Int,
    p: Float[jnp.ndarray, "n_categories"],
) -> Int[Array, "n_categories"]:
    return tfd.Multinomial(n, probs=p).sample(seed=seed)


def sample_gamma(
    seed: Float[Array, "2"],
    a: Float,
    b: Float,
) -> Float:
    return jr.gamma(seed, a) / b


def sample_inv_gamma(
    seed: Float[Array, "2"],
    a: Float,
    b: Float,
) -> Float:
    return 1.0 / sample_gamma(seed, a, b)

@partial(jax.jit, static_argnames=["n_timesteps"])
def simulate_markov_chain(
    seed: Float[Array, "2"],
    trans_probs: Union[
        Float[Array, "n_states n_states"], Float[Array, "n_timesteps n_states n_states"]
    ],
    n_timesteps: Int,
) -> Int[Array, "n_timesteps"]:
    """Simulate a state sequence from in Markov chain.

    Args:
        seed: random seed
        trans_probs: transition probabilities between states
        n_timesteps: number of timesteps to simulate

    Returns:
        states: simulated state sequence
    """
    seeds = jr.split(seed, n_timesteps + 1)
    n_states = trans_probs.shape[0]
    log_trans_probs = jnp.log(trans_probs)
    init_state = jr.categorical(seeds[0], jnp.ones(n_states) / n_states)

    if trans_probs.ndim == 2:
        def step(state, seed):
            next_state = jr.categorical(seed, logT[state])
            return next_state, next_state
        _, states = jax.lax.scan(step, init_state, seeds[1:])

    else:
        def step(state, args):
            seed, logT = args
            next_state = jr.categorical(seed, logT[state])
            return next_state, next_state
        _, states = jax.lax.scan(step, init_state, (seeds[1:], log_trans_probs))
        
    return states


def count_transitions(
    states: Int[Array, "n_timesteps"],
    mask: Int[Array, "n_timesteps"],
    n_states: int,
) -> Float[Array, "n_states n_states"]:
    """Count transitions between states.

    Args:
        states: discrete state sequence
        mask: mask of valid observations
        n_states: number of discrete states

    Returns:
        trans_counts: transition counts
    """
    trans_counts = (
        jnp.zeros((n_states, n_states))
        .at[states[:-1], states[1:]]
        .add(mask[:-1])
    )
    return trans_counts


def compare_states(
    true_states: Int[Array, "n_sessions n_timesteps"],
    pred_states: Int[Array, "n_sessions n_timesteps"],
    n_states: Int,
) -> Tuple[Int[Array, "n_states n_states"], Int[Array, "n_states"], Float]:
    """Compare true and predicted states.

    Args:
        true_states: true state sequences
        pred_states: predicted state sequences
        n_states: number of states

    Returns:
        confusion_matrix: confusion matrix
        optimal_permutation: optimal permutation of predicted states
        accuracy: proportion of correct labels (after optimal permutation)
    """
    confusion = jnp.zeros((n_states, n_states)).at[pred_states, true_states].add(1)
    optimal_perm = linear_sum_assignment(-confusion.T)[1]
    accuracy = confusion[optimal_perm, jnp.arange(n_states)].sum() / true_states.size
    confusion = confusion / confusion.sum(axis=1, keepdims=True)
    return confusion, optimal_perm, accuracy


def sample_hmc(
    seed: Float[Array, "2"],
    log_prob_fn: Callable,
    init_params: PyTree,
    num_leapfrog_steps: Int = 3,
    step_size: Float = 0.001,
    num_results: Int = 1,
    num_burnin_steps: Int = 100,
) -> Tuple[PyTree, PyTree]:
    """Sample using Hamiltonian Monte Carlo."""
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
    )
    params, _, kernel_state = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=init_params,
        kernel=hmc_kernel,
        seed=seed,
        trace_fn=None,
        return_final_kernel_results=True,
    )
    return params, kernel_state


def sample_laplace(
    seed: Float[Array, "2"],
    log_prob_fn: Callable,
    init_params: PyTree,
    gradient_descent_iters: Int = 200,
    gradient_descent_lr: Float = 0.01,
) -> Tuple[PyTree, Float[Array, "gradient_descent_iters"]]:
    """Sample using Laplace approximation. Uses gradient descent to find mode of posterior.

    Args:
        seed: random seed
        log_prob_fn: log probability function
        init_params: initial parameters
        gradient_descent_iters: number of gradient descent iterations
        gradient_descent_lr: gradient descent learning rate

    Returns:
        params: sampled parameters
        losses: loss history
    """
    # find the mode of the posterior
    mode, _, losses = run_gradient_descent(
        lambda x: -log_prob_fn(x),
        init_params,
        num_mstep_iters=gradient_descent_iters,
        optimizer=optax.adam(gradient_descent_lr),
    )
    # calculate covariance matrix from hessian at mode
    mode, unravel_fn = jax.flatten_util.ravel_pytree(mode)
    ll_fn = lambda x: log_prob_fn(unravel_fn(x))
    hessian_at_mode = jax.hessian(ll_fn)(mode)
    covariance_matrix = psd_inv(-hessian_at_mode, diagonal_boost=1e-2)

    # sample from laplace approximation
    x = jr.multivariate_normal(seed, mean=mode, cov=covariance_matrix)
    return unravel_fn(x), losses


def symmetrize(A: Float[Array, "n n"]) -> Float[Array, "n n"]:
    """Symmetrize a matrix by averaging it with its transpose."""
    return (A + A.swapaxes(-1, -2)) / 2


def psd_solve(
    A: Float[Array, "n n"], 
    B: Float[Array, "n m"], 
    diagonal_boost: float = 1e-6
) -> Float[Array, "n m"]:
    """Solve the linear system Ax = B, where A is a positive semi-definite matrix.
    Args:
        A: positive semi-definite matrix
        B: right-hand side matrix
        diagonal_boost: boost to diagonal to ensure positive definiteness
    Returns:
        x: solution to the linear system
    """
    A = symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1])
    L, lower = cho_factor(A, lower=True)
    x = cho_solve((L, lower), B)
    return x


def psd_inv(A : Float[Array, "n n"], diagonal_boost: float = 1e-6) -> Float[Array, "n n"]:
    """Compute the inverse of a positive semi-definite matrix using Cholesky decomposition.
    Args:
        A: positive semi-definite matrix
        diagonal_boost: boost to diagonal to ensure positive definiteness
    Returns:
        Ainv: inverse of the matrix
    """
    Ainv = psd_solve(A, jnp.eye(A.shape[-1]), diagonal_boost=diagonal_boost)
    return symmetrize(Ainv)


def cross_sequence_mutual_information(
    sequence1: Int[Array, "n_timesteps"],
    sequence2: Int[Array, "n_timesteps"],
    mask: Bool[Array, "n_timesteps"],
    n_categories: int,
    pseudo_count: float = 1e-8,
) -> Float:
    """Compute cross-sequence mutual information.

    Args:
        sequence1: first sequence
        sequence2: second sequence
        mask: mask for valid timesteps
        n_categories: number of categories
        pseudo_count: pseudo count to add to probabilities

    Returns:
        mi: mutual information
    """
    counts = (
        (jnp.ones((n_categories, n_categories)) * pseudo_count)
        .at[sequence1, sequence2]
        .add(mask)
    )
    probs = counts / counts.sum()
    mi = (probs * jnp.log(probs / (probs.sum(1)[:, na] * probs.sum(0)[na, :]))).sum()
    return mi


def lagged_mutual_information(
    sequences: Int[Array, "n_sequences n_timesteps"],
    mask: Bool[Array, "n_sequences n_timesteps"],
    lags: Int[Array, "n_lags"],
    pseudo_count: float = 1e-8,
) -> Float[Array, "n_sequences n_lags"]:
    """Compute mutual information at a range of lags for real data and equivalent Markov chains.
    
    Args:
        sequences: sequences from which to compute mutual information
        mask: mask indicating valid timesteps
        lags: array of temporal lags
        pseudo_count: pseudo count to use when computing mutual information

    Returns:
        real_mi: mutual information for each sequence at each lag
        markov_mi: mutual information for equivalent Markov chains at each lag
        shuff_mi: mutual information across randomly paired sequences at each lag
    """
    # get dimensions and number of categories
    n_sequences, n_timesteps = sequences.shape
    n_lags = len(lags)
    n_categories = jnp.where(mask, sequences, 0).max().item() + 1

    # simulate Markov chains
    seeds = jr.split(jr.PRNGKey(0), n_sequences)
    trans_counts = jax.vmap(count_transitions, in_axes=(0, 0, None))(sequences, mask, n_categories)
    trans_probs = trans_counts / trans_counts.sum(axis=2, keepdims=True)
    markov_seqs = jnp.zeros((n_sequences, n_timesteps), dtype=int)
    for i in tqdm.trange(n_sequences, desc="Simulating Markov chains"):
        markov_seqs = markov_seqs.at[i].set(
            simulate_markov_chain(seeds[i], trans_probs[i], n_timesteps)
        )

    # compute mutual information 
    cross_mi = jax.vmap(
        jax.jit(cross_sequence_mutual_information, static_argnums=(3, 4)),
        in_axes=(0, 0, 0, None, None),
    )
    real_mi = jnp.zeros((n_sequences, n_lags))
    markov_mi = jnp.zeros((n_sequences, n_lags))
    shuff_mi = jnp.zeros((n_sequences, n_lags))

    shuff_seqs = jnp.roll(sequences, 1, axis=0)
    for i, lag in tqdm.tqdm(enumerate(lags), total=n_lags, desc="Computing MI"):
        lagged_real_seqs = jnp.roll(sequences, lag, axis=1)
        lagged_markov_seqs = jnp.roll(markov_seqs, lag, axis=1)
        lagged_mask = mask.at[:, :lag].set(0)

        real_mi = real_mi.at[:, i].set(
            cross_mi(sequences, lagged_real_seqs, lagged_mask, n_categories, pseudo_count)
        )
        markov_mi = markov_mi.at[:, i].set(
            cross_mi(markov_seqs, lagged_markov_seqs, lagged_mask, n_categories, pseudo_count)
        )
        shuff_mi = shuff_mi.at[:, i].set(
            cross_mi(shuff_seqs, lagged_real_seqs, lagged_mask, n_categories, pseudo_count)
        )
    return real_mi, markov_mi, shuff_mi


def save_hdf5(
    filepath: str,
    save_dict: Dict[str, PyTree],
    datapath: Optional[str] = None,
    overwrite_results: bool = False,
) -> None:
    """Save a dict of pytrees to an hdf5 file. The leaves of the pytrees must
    be numpy arrays, scalars, or strings.

    Args:
        filepath: Path of the hdf5 file to create.
        save_dict: Dictionary where the values are pytrees.
        datapath: Path within hdf5 file to save the data. If None, data are saved at the root.
    """
    with h5py.File(filepath, "a") as f:
        if datapath is not None:
            _savetree_hdf5(jax.device_get(save_dict), f, datapath)
        else:
            for k, tree in save_dict.items():
                _savetree_hdf5(jax.device_get(tree), f, k)


def load_hdf5(
    filepath: str,
    datapath: Optional[str] = None,
) -> Dict[str, PyTree]:
    """Load a dict of pytrees from an hdf5 file.

    Args:
        filepath: Path of the hdf5 file to load.
        datapath: Path within hdf5 file to load data from. If None, loads from the root.

    Returns:
        save_dict: Dictionary where the values are pytrees.
    """
    with h5py.File(filepath, "r") as f:
        if datapath is None:
            return {k: _loadtree_hdf5(f[k]) for k in f}
        else:
            return _loadtree_hdf5(f[datapath])


def _savetree_hdf5(tree: PyTree, group: h5py.Group, name: str) -> None:
    """Recursively save a pytree to an h5 file group."""
    if name in group:
        del group[name]
    if isinstance(tree, np.ndarray):
        if tree.dtype.kind == "U":
            dt = h5py.special_dtype(vlen=str)
            group.create_dataset(name, data=tree.astype(object), dtype=dt)
        else:
            group.create_dataset(name, data=tree)
    elif isinstance(tree, (float, int, str)):
        group.create_dataset(name, data=tree)
    else:
        subgroup = group.create_group(name)
        subgroup.attrs["type"] = type(tree).__name__

        if isinstance(tree, (tuple, list)):
            for k, subtree in enumerate(tree):
                _savetree_hdf5(subtree, subgroup, f"arr{k}")
        elif isinstance(tree, dict):
            for k, subtree in tree.items():
                _savetree_hdf5(subtree, subgroup, k)
        else:
            raise ValueError(f"Unrecognized type {type(tree)}")


def _loadtree_hdf5(leaf: Union[h5py.Dataset, h5py.Group]) -> PyTree:
    """Recursively load a pytree from an h5 file group."""
    if isinstance(leaf, h5py.Dataset):
        data = np.array(leaf[()])
        if h5py.check_dtype(vlen=data.dtype) == str:
            data = np.array([item.decode("utf-8") for item in data])
        elif data.dtype.kind == "S":
            data = data.item().decode("utf-8")
        elif data.shape == ():
            data = data.item()
        return data
    else:
        leaf_type = leaf.attrs["type"]
        values = map(_loadtree_hdf5, leaf.values())
        if leaf_type == "dict":
            return dict(zip(leaf.keys(), values))
        elif leaf_type == "list":
            return list(values)
        elif leaf_type == "tuple":
            return tuple(values)
        else:
            raise ValueError(f"Unrecognized type {leaf_type}")


def unbatch(
    data: Array,
    keys: Union[list[str], Array],
    bounds: Int[Array, "n_segs 2"]
) -> Dict[str, Array]:
    """Invert :py:func:`state_moseq.util.batch`

    Args:
        data: Stack of segmented time-series, shape (n_segs, seg_length, ...).
        keys: Name of the time-series that each segment came from
        bounds: Start and end indices for each segment.

    Returns:
        data_dict: Dictionary mapping names to reconstructed time-series.
    """
    data_dict = {}
    for key in set(list(keys)):
        length = bounds[keys == key, 1].max()
        seq = np.zeros((int(length), *data.shape[2:]), dtype=data.dtype)
        for (s, e), d in zip(bounds[keys == key], data[keys == key]):
            seq[s:e] = d[: e - s]
        data_dict[key] = seq
    return data_dict


def batch(
    data_dict: Dict[str, Array],
    keys: Optional[list[str]] = None,
    seg_length: Optional[int] = None,
    seg_overlap: int = 30,
) -> Tuple[Array, Int[Array, "N seg_length"], Tuple[list[str], Int[Array, "N 2"]]]:
    """Stack time-series data of different lengths into a single array for batch
    processing, optionally breaking up the data into fixed length segments. The
    data is padded so that the stacked array isn't ragged. The padding
    repeats the last frame of each time-series until the end of the segment.

    Args:
        data_dict: Dictionary of time-series, each of shape (T, ...).
        keys: Optional list of keys to control order and inclusion of time-series.
        seg_length: Length of each segment. Defaults to max sequence length.
        seg_overlap: Overlap between segments in frames.

    Returns:
        data: Stacked data array, shape (N, seg_length, ...).
        mask: Binary mask for valid data (1 = valid, 0 = padding), shape (N, seg_length).
        metadata: Tuple (keys, bounds), identifying sources and segment positions.
    """
    if keys is None:
        keys = sorted(data_dict.keys())
    Ns = [len(data_dict[key]) for key in keys]
    if seg_length is None:
        seg_length = np.max(Ns)

    stack, mask, keys_out, bounds = [], [], [], []
    for key, N in zip(keys, Ns):
        for start in range(0, N, seg_length):
            arr = data_dict[key]
            end = min(start + seg_length + seg_overlap, N)
            pad_length = seg_length + seg_overlap - (end - start)
            padding = np.repeat(arr[end - 1 : end], pad_length, axis=0)
            mask.append(np.hstack([np.ones(end - start), np.zeros(pad_length)]))
            stack.append(np.concatenate([arr[start:end], padding], axis=0))
            keys_out.append(key)
            bounds.append((start, end))

    stack = np.stack(stack)
    mask = np.stack(mask)
    metadata = (np.array(keys_out), np.array(bounds))
    return stack, mask, metadata


def get_durations(
    states_dict: Dict[str, Int[Array, "n_timesteps"]]
) -> Int[Array, "n_durations"]:
    """Get durations of high-level states.

    Args:
        states_dict: Dictionary of high-level state sequences.

    Returns:
        durations: Times between high-level state transitions (across all sequences).

    Examples:
        >>> states_dict = {
        ...     'name1': np.array([1, 1, 2, 2, 2, 3]),
        ...     'name2': np.array([0, 0, 0, 1]),
        ... }
        >>> get_durations(states_dict)
        array([2, 3, 1, 3, 1])
    """
    stateseq_flat = np.hstack(list(states_dict.values()))
    stateseq_padded = np.hstack([[-1], stateseq_flat, [-1]])
    changepoints = np.diff(stateseq_padded).nonzero()[0]
    return changepoints[1:] - changepoints[:-1]


def sample_instances(
    states_dict: Dict[str, Int[Array, "n_timesteps"]],
    num_instances: int,
) -> Dict[int, List[Tuple[str, int, int]]]:
    """Randomly sample instances of each state.

    Args:
        states_dict: Dictionary of state sequences.
        num_instances: Number of instances per state.

    Returns:
        sampled_instances: Dictionary mapping state index to instances.
    """
    state_ixs = np.unique(np.hstack(list(states_dict.values())))
    all_instances = {state_ix: [] for state_ix in state_ixs}

    for key, stateseq in states_dict.items():
        transitions = np.nonzero(stateseq[1:] != stateseq[:-1])[0] + 1
        starts = np.insert(transitions, 0, 0)
        ends = np.append(transitions, len(stateseq))
        for s, e, state in zip(starts, ends, stateseq[starts]):
            all_instances[state].append((key, s, e))

    sampled_instances = {}
    for state_ix, instances in all_instances.items():
        subset = np.random.permutation(len(instances))[:num_instances]
        sampled_instances[state_ix] = [instances[i] for i in subset]

    return sampled_instances
