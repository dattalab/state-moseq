import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import norm, dirichlet
from jaxtyping import Array, Float, Int
from jax.nn import softmax
import tqdm
from typing import Tuple
from functools import partial
from dynamax.hidden_markov_model import (
    hmm_filter,
    hmm_smoother,
    hmm_posterior_mode,
    hmm_posterior_sample,
    parallel_hmm_posterior_sample,
)

from .util import (
    simulate_markov_chain,
    lower_dim,
    raise_dim,
    count_transitions,
)

na = jnp.newaxis

"""
data = {
    "syllables": (n_sequences, n_timesteps, n_syllables),
    "mask": (n_sequences, n_timesteps),
}

states: (n_sequences, n_timesteps)

params = {
    "emissions": (n_states, n_syllables, n_syllables),
    "trans_probs": (n_states, n_states),
}

hypparams = {
    "n_states": (,),
    "emission_beta": (,),
    "trans_beta": (,),
    "trans_kappa": (,),
    "n_syllables"
}
"""


def obs_log_likelihoods(
    data: dict,
    params: dict,
) -> Float[Array, "n_sequences n_timesteps n_states"]:
    """Compute log likelihoods of observations for each hidden state."""

    n_sequences = data["syllables"].shape[0]
    n_states = params["trans_probs"].shape[0]

    log_syllable_trans_probs = jnp.log(params["emissions"])

    log_likelihoods = jax.vmap(
        lambda T: T[data["syllables"][:, :-1], data["syllables"][:, 1:]]
    )(log_syllable_trans_probs)

    log_likelihoods = jnp.concatenate(
        [jnp.zeros((n_states, n_sequences, 1)), log_likelihoods], axis=2
    )
    return log_likelihoods.transpose((1, 2, 0)) * data["mask"][:, :, na]


def log_params_prob(
    params: dict,
    hypparams: dict,
) -> Float:
    """Compute the log probability of the parameters based on their priors."""

    n_states, n_syllables = params["emissions"].shape[:2]

    # prior on emission base parameters
    emissions_log_prob = jax.vmap(dirichlet.logpdf, in_axes=(0, None))(
        params["emissions"].reshape(-1, n_syllables),
        jnp.ones(n_syllables) * hypparams["emissions_beta"],
    ).sum()

    # prior on transition parameters
    trans_probs_log_prob = jax.vmap(dirichlet.logpdf)(
        params["trans_probs"],
        jnp.eye(n_states) * hypparams["trans_kappa"] + hypparams["trans_beta"],
    ).sum()

    return emissions_log_prob + trans_probs_log_prob


def log_joint_prob(
    data: dict,
    params: dict,
    hypparams: dict,
) -> Float:
    """Compute the log joint probability of the data and parameters."""
    return marginal_loglik(data, params) + log_params_prob(params, hypparams)


@partial(jax.jit, static_argnums=(3,))
def resample_states(
    seed: Float[Array, "2"],
    data: dict,
    params: dict,
    parallel: bool = False,
) -> Tuple[Int[Array, "n_sequences n_timesteps"], Float]:
    """Resample hidden states from their posterior distribution.

    Args:
        seed: random seed
        data: data dictionary
        params: parameters dictionary
        parallel: whether to use parallel message passing

    Returns:
        states: resampled hidden states
        marginal_loglik: marginal log likelihood of the data
    """
    n_states = params["trans_probs"].shape[0]
    seeds = jr.split(seed, data["syllables"].shape[0])

    sample_fn = parallel_hmm_posterior_sample if parallel else hmm_posterior_sample
    marginal_logliks, states = jax.vmap(sample_fn, in_axes=(0, None, None, 0))(
        seeds,
        jnp.ones(n_states) / n_states,
        params["trans_probs"],
        obs_log_likelihoods(data, params),
    )
    return states, marginal_logliks.sum()


def fit_gibbs(
    data: dict,
    hypparams: dict,
    init_params: dict,
    init_states: Int[Array, "n_sequences n_timesteps"] = None,
    seed: Float[Array, "2"] = jr.PRNGKey(0),
    num_iters: Int = 100,
    parallel: bool = False,
) -> Tuple[dict, Float[Array, "num_iters"]]:
    """Fit a model using Gibbs sampling.

    Args:
        data: data dictionary
        hypparams: hyperparameters dictionary
        init_params: initial parameters directionary
        init_states: initial hidden states (optional)
        seed: random seed
        num_iters: number of iterations
        parallel: whether to use parallel message passing

    Returns:
        params: fitted parameters dictionary
        log_joints: log joint probability of the data and parameters recorded at each iteration
    """
    if init_states is None:
        states, _ = resample_states(seed, data, init_params, parallel)

    log_joints = []
    params = init_params
    for _ in tqdm.trange(num_iters):
        seed, subseed = jr.split(seed)
        params = resample_params(subseed, data, params, states, hypparams)
        states, marginal_loglik = resample_states(seed, data, params, parallel)
        log_joints.append(marginal_loglik + log_params_prob(params, hypparams))
    return params, states, jnp.array(log_joints)


def initialize_params(
    data: dict,
    hypparams: dict,
    states: Int[Array, "n_sequences n_timesteps"] = None,
    seed: Float[Array, "2"] = jr.PRNGKey(0),
) -> dict:
    """Initialize parameters by sampling from their prior distribution or using
    provided states.

    Args:
        data: data dictionary
        hypparams: hyperparameters dictionary
        states: states used for initializing the parameters (optional)
        seed: random seed
    """
    if states is not None:
        params = resample_params(seed, data, states, hypparams)
    else:
        params = random_params(seed, hypparams)
    return params


def marginal_loglik(
    data: dict,
    params: dict,
) -> Float[Array, "n_sequences n_timesteps n_states"]:
    """Estimate marginal log likelihood of the data"""
    n_states = params["trans_probs"].shape[0]
    mll = jax.vmap(hmm_filter, in_axes=(None, None, 0))(
        jnp.ones(n_states) / n_states,
        params["trans_probs"],
        obs_log_likelihoods(data, params),
    ).marginal_loglik.sum()
    return mll


def smoothed_states(
    data: dict,
    params: dict,
) -> Float[Array, "n_sequences n_timesteps n_states"]:
    """Estimate marginals of hidden states using forward-backward algorithm."""
    n_states = params["trans_probs"].shape[0]
    return jax.vmap(hmm_smoother, in_axes=(None, None, 0))(
        jnp.ones(n_states) / n_states,
        params["trans_probs"],
        obs_log_likelihoods(data, params),
    ).smoothed_probs


def filtered_states(
    data: dict,
    params: dict,
) -> Float[Array, "n_sequences n_timesteps n_states"]:
    """Estimate marginals of hidden states using forward-backward algorithm."""
    n_states = params["trans_probs"].shape[0]
    return jax.vmap(hmm_filter, in_axes=(None, None, 0))(
        jnp.ones(n_states) / n_states,
        params["trans_probs"],
        obs_log_likelihoods(data, params),
    ).filtered_probs


def predicted_states(
    data: dict,
    params: dict,
) -> Float[Array, "n_sequences n_timesteps n_states"]:
    """Predict hidden states using Viterbi algorithm."""
    n_states = params["trans_probs"].shape[0]
    return jax.vmap(hmm_posterior_mode, in_axes=(None, None, 0))(
        jnp.ones(n_states) / n_states,
        params["trans_probs"],
        obs_log_likelihoods(data, params),
    )


def random_params(
    seed: Float[Array, "2"],
    hypparams: dict,
) -> dict:
    """Generate random model parameters.

    emissions ~ Dirichlet(emissions_beta) (for each state and syllable)
    trans_probs ~ Dirichlet(trans_beta + trans_kappa * I)

    Args:
        seed: random seed
        hypparams: hyperparameters dictionary

    Returns:
        params: parameters dictionary
    """
    n_syllables = hypparams["n_syllables"]
    n_states = hypparams["n_states"]
    seeds = jr.split(seed, 2)

    emissions = jr.dirichlet(
        seeds[0],
        jnp.ones((n_states, n_syllables, n_syllables)) * hypparams["emissions_beta"],
    )

    trans_probs = jax.vmap(jr.dirichlet)(
        jr.split(seeds[2], n_states),
        jnp.eye(n_states) * hypparams["trans_kappa"] + hypparams["trans_beta"],
    )

    return {
        "emissions": emissions,
        "trans_probs": trans_probs,
    }


def simulate(
    seed: Float[Array, "2"],
    params: dict,
    n_timesteps: Int,
    n_sequences: Int,
) -> Tuple[
    Int[Array, "n_sequences n_timesteps"], Int[Array, "n_sequences n_timesteps"]
]:
    """Simulate data from the model.

    Args:
        seed: random seed
        params: parameters dictionary
        n_timesteps: number of timesteps to simulate
        n_sequences: number of sessions to simulate

    Returns:
        states: simulated states
        syllables: simulated syllables
    """
    seeds = jr.split(seed, 3)

    states = jax.vmap(simulate_markov_chain, in_axes=(0, None, None))(
        jr.split(seeds[0], n_sequences),
        params["trans_probs"],
        n_timesteps,
    )

    syllable_trans_probs = params["emissions"][states]

    syllables = jax.vmap(simulate_markov_chain, in_axes=(0, 0, None))(
        jr.split(seeds[1], n_sequences), syllable_trans_probs, n_timesteps
    )
    return states, syllables


def resample_params(
    seed: Float[Array, "2"],
    data: dict,
    params: dict,
    states: Int[Array, "n_sequences n_timesteps"],
    hypparams: dict,
) -> dict:
    """Resample parameters from their posterior distribution.

    Args:
        seed: random seed
        data: data dictionary
        params: parameters dictionary
        states: hidden states
        hypparams: hyperparameters dictionary

    Returns:
        params: parameters dictionary
    """
    seeds = jr.split(seed, 2)

    emissions = resample_emission_params(
        seeds[1],
        data["syllables"],
        data["mask"],
        states,
        hypparams["n_states"],
        hypparams["n_syllables"],
        hypparams["emissions_beta"],
    )
    trans_probs = resample_trans_probs(
        seeds[2],
        data["mask"],
        states,
        hypparams["n_states"],
        hypparams["trans_beta"],
        hypparams["trans_kappa"],
    )
    params = {
        "emissions": emissions,
        "trans_probs": trans_probs,
    }
    return params

    emissions = resample_emission_params(
        seeds[1],
        data["syllables"],
        data["mask"],
        states,
        hypparams["n_states"],
        hypparams["n_syllables"],
        hypparams["emissions_beta"],
    )


@partial(jax.jit, static_argnums=(4, 5))
def resample_emission_params(
    seed: Float[Array, "2"],
    syllables: Int[Array, "n_sequences n_timesteps"],
    mask: Int[Array, "n_sequences n_timesteps"],
    states: Int[Array, "n_sequences n_timesteps"],
    n_states: int,
    n_syllables: int,
    emissions_beta: Float,
) -> Float[Array, "n_states n_syllables n_syllables"]:
    """Resample emission parameters from their posterior distribution.

    Args:
        seed: random seed
        syllables: syllable observations
        mask: mask of valid observations
        states: hidden states
        n_states: number of hidden states
        n_syllables: number of syllables
        emission_base_sigma: emission base standard deviation
        emission_biases_sigma: emission biases standard deviation

    Returns:
        emissions: syllable transition probabilities for each state
    """
    sufficient_stats = (
        jnp.zeros((n_states, n_syllables, n_syllables))
        .at[states[:, 1:], syllables[:, :-1], syllables[:, 1:]]
        .add(mask[:, 1:])
    )
    emissions = jax.vmap(jr.dirichlet)(
        jr.split(seed, n_states * n_syllables),
        sufficient_stats.reshape(-1, n_syllables) + emissions_beta,
    ).reshape(n_states, n_syllables, n_syllables)

    return emissions


@partial(jax.jit, static_argnums=(3,))
def resample_trans_probs(
    seed: Float[Array, "2"],
    mask: Int[Array, "n_sequences n_timesteps"],
    states: Int[Array, "n_sequences n_timesteps"],
    n_states: int,
    beta: Float,
    kappa: Float,
) -> Float[Array, "n_states n_states"]:
    """Resample transition probabilities from their posterior distribution.

    Args:
        seed: random seed
        mask: mask of valid observations
        states: hidden states
        n_states: number of hidden states
        beta: Dirichlet concentration parameter
        kappa: Dirichlet concentration parameter

    Returns:
        trans_probs: posterior transition probabilities
    """
    trans_counts = jax.vmap(count_transitions, in_axes=(0, 0, None))(states, mask, n_states).sum(0)
    trans_probs = jax.vmap(jr.dirichlet)(
        jr.split(seed, n_states), trans_counts + beta + jnp.eye(n_states) * kappa
    )
    return trans_probs
