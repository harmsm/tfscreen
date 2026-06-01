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

Memory note
-----------
``_nn_forward`` is decorated with ``jax.checkpoint`` so that the (M, H)
hidden activations are **not cached** during the forward pass.  They are
recomputed on demand during backpropagation.  This trades a small amount
of extra compute for reduced peak GPU memory, which matters when a large
dense mutation-genotype matrix is also live at the same time.
"""

import jax
import jax.numpy as jnp
import numpyro as pyro

_DEFAULT_HIDDEN_SIZE = 16


# ---------------------------------------------------------------------------
# Checkpointed pure forward pass (no side effects — safe to checkpoint)
# ---------------------------------------------------------------------------

@jax.checkpoint
def _nn_forward(features, W1_all, b1_all, W2_all, b2_all, n_chains_f):
    """
    Vectorised forward pass over all structures.

    Parameters
    ----------
    features  : (M, S, feat_dim)
    W1_all    : (S, feat_dim, H)
    b1_all    : (S, H)
    W2_all    : (S, H, 1)
    b2_all    : (S, 1)
    n_chains_f: (S,)  float32

    Returns
    -------
    (M, S)  NN-predicted ΔΔG for each mutation × structure.
    """
    def one_struct(feats_s, W1, b1, W2, b2):
        """Single-structure two-layer MLP: (M, feat_dim) → (M,)."""
        h   = jax.nn.relu(feats_s @ W1 + b1)   # (M, H)
        return (h @ W2 + b2)[:, 0]              # (M,)

    # vmap over the structure axis (axis-1 of features, axis-0 of weights)
    out_SM = jax.vmap(one_struct, in_axes=(1, 0, 0, 0, 0))(
        features, W1_all, b1_all, W2_all, b2_all,
    )   # (S, M)

    return (out_SM * n_chains_f[:, None]).T   # (M, S)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

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

    Notes
    -----
    ``pyro.param`` registration (the side-effectful part) happens first in a
    plain Python loop, before ``_nn_forward`` is called.  ``_nn_forward`` is
    a pure JAX function decorated with ``jax.checkpoint``, which prevents its
    hidden activations from being cached during forward and instead recomputes
    them during backpropagation.  This reduces peak GPU memory at the cost of
    one extra forward pass per training step.
    """
    _, S, feat_dim = features.shape
    H = hidden_size
    n_chains_f = jnp.array(n_chains, dtype=jnp.float32)   # (S,)

    # ------------------------------------------------------------------
    # Register all parameters first.
    # pyro.param calls are SIDE EFFECTS and must remain OUTSIDE the
    # checkpointed function.  We collect them into stacked arrays so
    # that _nn_forward can vmap over the structure dimension cleanly.
    # ------------------------------------------------------------------
    W1_all = jnp.stack(
        [pyro.param(f"{name}_nn_{sname}_W1", jnp.zeros((feat_dim, H)))
         for sname in struct_names], axis=0)   # (S, feat_dim, H)
    b1_all = jnp.stack(
        [pyro.param(f"{name}_nn_{sname}_b1", jnp.zeros(H))
         for sname in struct_names], axis=0)   # (S, H)
    W2_all = jnp.stack(
        [pyro.param(f"{name}_nn_{sname}_W2", jnp.zeros((H, 1)))
         for sname in struct_names], axis=0)   # (S, H, 1)
    b2_all = jnp.stack(
        [pyro.param(f"{name}_nn_{sname}_b2", jnp.zeros(1))
         for sname in struct_names], axis=0)   # (S, 1)

    # ------------------------------------------------------------------
    # Checkpointed forward pass — no activations cached during forward.
    # ------------------------------------------------------------------
    return _nn_forward(features, W1_all, b1_all, W2_all, b2_all, n_chains_f)
