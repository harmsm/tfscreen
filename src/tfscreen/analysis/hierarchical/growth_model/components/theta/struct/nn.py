"""
Per-structure MLP that predicts per-mutation ΔΔG from structural features.

MLP weights are treated as deterministic learned parameters via
``pyro.param`` rather than sampled latent variables.  This is an empirical-
Bayes approach: the NN produces prior means for the ΔΔG latent variables,
and the data update those latent variables away from the NN prediction via
a per-structure trust scale σ_s (see ``prior.py``).

Zero-initialising all weights ensures the model starts in the perturbative
(WT) regime — all mutation effects are zero at initialization and the data
drive them away from zero as needed.

The raw scalar output for structure s is multiplied by n_chains[s] to
account for chain stoichiometry: in a symmetric homodimer the same mutation
is present on both chains and contributes twice to the free energy change.

Architecture (per structure):
    input  → (feat_dim, H) → ReLU → (H, 1) → scalar
    shape:   (M, 60)       → (M, H)         → (M,)
"""

import jax
import jax.numpy as jnp
import numpyro as pyro

_DEFAULT_HIDDEN_SIZE = 16


def compute_nn_predictions(name, features, struct_names, n_chains,
                           hidden_size=_DEFAULT_HIDDEN_SIZE):
    """
    Predict per-mutation ΔΔG for each structure via independent per-structure
    two-layer MLPs with learned (``pyro.param``) weights.

    Parameters
    ----------
    name : str
        Prefix for all ``pyro.param`` site names.
    features : jnp.ndarray, shape (M, S, 60)
        Per-mutation per-structure feature vectors.
    struct_names : sequence of str, length S
        Structure identifiers.  Each structure gets its own independent MLP;
        names are used to form unique ``pyro.param`` site names.
    n_chains : array-like, shape (S,) int
        Number of chains bearing the mutation in each structure.  The scalar
        MLP output is multiplied by this value.
    hidden_size : int
        Hidden layer width H.

    Returns
    -------
    jnp.ndarray, shape (M, S)
        NN-predicted ΔΔG for each mutation and structure.
    """
    _, S, feat_dim = features.shape
    H = hidden_size
    n_chains_f = jnp.array(n_chains, dtype=jnp.float32)  # (S,)

    outputs = []
    for s_idx, sname in enumerate(struct_names):
        feats_s = features[:, s_idx, :]    # (M, feat_dim)
        prefix  = f"{name}_nn_{sname}"

        W1 = pyro.param(f"{prefix}_W1", jnp.zeros((feat_dim, H)))  # (feat_dim, H)
        b1 = pyro.param(f"{prefix}_b1", jnp.zeros(H))              # (H,)
        W2 = pyro.param(f"{prefix}_W2", jnp.zeros((H, 1)))         # (H, 1)
        b2 = pyro.param(f"{prefix}_b2", jnp.zeros(1))              # (1,)

        h   = jax.nn.relu(feats_s @ W1 + b1)   # (M, H)
        out = (h @ W2 + b2)[:, 0]              # (M,)

        outputs.append(out * n_chains_f[s_idx])

    return jnp.stack(outputs, axis=1)   # (M, S)
