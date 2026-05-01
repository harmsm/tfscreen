"""
K-assembly via MLP on LigandMPNN structural features.

The four input structures represent distinct thermodynamic states:
    H    = free TF (head/operator-binding conformation)
    HD   = TF:operator complex
    L    = free TF (linker/allosteric conformation)
    LE2  = TF:effector complex (IPTG-bound)

The MLP maps per-mutation ΔlogP features (shape: num_mutation × 4) to
per-state ΔΔG values, which are projected onto Δln_K via the fixed
thermodynamic matrix:

    Δln_K_op = ΔΔG_H  - ΔΔG_HD      (operator binding)
    Δln_K_HL = ΔΔG_H  - ΔΔG_L       (head-linker allosteric)
    Δln_K_E  = ΔΔG_L  - ΔΔG_LE2     (effector binding)

Mutation effects are additive at the genotype level; pairwise epistasis is
not modeled (structural features are intrinsically per-mutation). K_E is
applied uniformly across effector species (no T-dim heterogeneity in the
mutation effect).

Thermodynamic functions are imported from thermo.py.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import numpyro as pyro
import numpyro.distributions as dist
import pandas as pd
from flax.struct import dataclass, field
from typing import Dict, Any, Union

from tfscreen.analysis.hierarchical.growth_model.data_class import GrowthData, BindingData
from tfscreen.analysis.hierarchical.growth_model.components.theta.lac_dimer.thermo import (
    ThetaParam,
    _compute_theta,
    _population_moments,
    run_model,
    get_population_moments,
)

# Fixed thermodynamic projection: rows = (K_op, K_HL, K_E), cols = (H, HD, L, LE2).
# Δln_K = ddG @ _PROJ.T   where ddG is (M, 4) → result (M, 3)
_PROJ = jnp.array([[1., -1.,  0.,  0.],
                    [1.,  0., -1.,  0.],
                    [0.,  0.,  1., -1.]])  # (3, 4)

# ---------------------------------------------------------------------------
# Feature extraction from NPZ
# ---------------------------------------------------------------------------

# Column order matches _PROJ: H=col0, HD=col1, L=col2, LE2=col3.
STRUCTURE_KEYS = ('H', 'HD', 'L', 'LE2')

# LigandMPNN amino-acid alphabet (ProteinMPNN standard; index 20 = X/unknown,
# excluded here — we use only the first 20 positions).
_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
_AA_TO_IDX = {aa: i for i, aa in enumerate(_ALPHABET)}

_MUT_PATTERN = re.compile(r'^([A-Z])(\d+)([A-Z])$')


def build_ligandmpnn_features(npz_path, mut_labels):
    """
    Build a (num_mutation, 4) ΔlogP feature matrix from a LigandMPNN NPZ file.

    The NPZ must contain, for each name in STRUCTURE_KEYS:
      ``{name}``              — float32 (L, 20): mean log-probabilities
      ``{name}_residue_nums`` — int32   (L,):    PDB residue numbers per row

    ΔlogP[m, s] = logP(mut_aa, pos, structure_s) − logP(wt_aa, pos, structure_s)

    Parameters
    ----------
    npz_path : str
        Path to the NPZ feature file (from generate_ligandmpnn_features.py).
    mut_labels : list of str
        Mutation labels in "A42G" format (wt_aa + 1-indexed PDB resnum + mut_aa).

    Returns
    -------
    np.ndarray, shape (num_mutation, 4)
    """
    data = np.load(npz_path)

    missing = [k for k in STRUCTURE_KEYS if k not in data]
    if missing:
        raise KeyError(
            f"NPZ file is missing structure key(s) {missing}. "
            f"Found: {sorted(data.keys())}. Required: {list(STRUCTURE_KEYS)}"
        )

    num_mut = len(mut_labels)
    features = np.zeros((num_mut, len(STRUCTURE_KEYS)), dtype=np.float32)

    for m_idx, label in enumerate(mut_labels):
        match = _MUT_PATTERN.match(label)
        if match is None:
            raise ValueError(f"Cannot parse mutation label: {label!r}")
        wt_aa, pos_1idx, mut_aa = match.group(1), int(match.group(2)), match.group(3)

        wt_idx  = _AA_TO_IDX.get(wt_aa)
        mut_idx = _AA_TO_IDX.get(mut_aa)
        if wt_idx is None:
            raise ValueError(f"Unknown amino acid {wt_aa!r} in mutation label {label!r}")
        if mut_idx is None:
            raise ValueError(f"Unknown amino acid {mut_aa!r} in mutation label {label!r}")

        for s_idx, key in enumerate(STRUCTURE_KEYS):
            logP     = data[key]                    # (L, 20)
            res_nums = data[f"{key}_residue_nums"]  # (L,)

            hits = np.where(res_nums == pos_1idx)[0]
            if hits.size == 0:
                raise ValueError(
                    f"Residue {pos_1idx} not found in structure '{key}'. "
                    f"Available: {sorted(set(res_nums.tolist()))}"
                )
            arr_idx = hits[0]
            features[m_idx, s_idx] = logP[arr_idx, mut_idx] - logP[arr_idx, wt_idx]

    return features

# Default hidden size — must match get_hyperparameters() default.
_DEFAULT_HIDDEN_SIZE = 8


# ---------------------------------------------------------------------------
# Priors pytree
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPriors:
    """Hyperparameters for the LigandMPNN-based ln-K lac-dimer theta model."""

    # WT priors (Normal in ln-K space)
    theta_ln_K_op_wt_loc: float
    theta_ln_K_op_wt_scale: float
    theta_ln_K_HL_wt_loc: float
    theta_ln_K_HL_wt_scale: float
    theta_ln_K_E_wt_loc: float
    theta_ln_K_E_wt_scale: float

    # Physical concentrations (M); Sochor, PeerJ 2014
    theta_tf_total_M: float
    theta_op_total_M: float

    # MLP architecture and weight prior scale
    # pytree_node=False keeps this as a concrete Python int during JIT tracing
    # (it determines tensor shapes and cannot be an abstract tracer).
    theta_nn_hidden_size: int = field(pytree_node=False)
    theta_nn_w_scale: float


# ---------------------------------------------------------------------------
# MLP forward pass
# ---------------------------------------------------------------------------

def _mlp_forward(features, W1, b1, W2, b2):
    """Single-hidden-layer MLP with ReLU: (M, 4) → (M, 4)."""
    h = jax.nn.relu(features @ W1 + b1)  # (M, H)
    return h @ W2 + b2                    # (M, 4)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def define_model(name: str,
                 data: Union[GrowthData, BindingData],
                 priors: ModelPriors) -> ThetaParam:
    """
    Define the LigandMPNN-based lac-dimer theta model.

    The MLP maps per-mutation ΔlogP features to per-state ΔΔG values, which
    are projected onto Δln_K via the fixed thermodynamic matrix and assembled
    into per-genotype equilibrium constants.

    Parameters
    ----------
    name : str
        Prefix for all Numpyro sample sites.
    data : GrowthData or BindingData
        Must have ``ligandmpnn_features`` (num_mutation × 4),
        ``mut_geno_matrix`` (num_mutation × G), and ``num_mutation``.
    priors : ModelPriors
    """
    T = data.num_titrant_name
    M_mat    = jnp.array(data.mut_geno_matrix)        # (M, G)
    features = jnp.array(data.ligandmpnn_features)    # (M, 4)
    H        = int(priors.theta_nn_hidden_size)
    w_scale  = priors.theta_nn_w_scale
    tf_total = priors.theta_tf_total_M
    op_total = priors.theta_op_total_M
    titrant_conc_M = data.titrant_conc / 1000.0       # mM → M

    # ------------------------------------------------------------------
    # WT equilibrium constants
    # ------------------------------------------------------------------
    ln_K_op_wt = pyro.sample(
        f"{name}_ln_K_op_wt",
        dist.Normal(priors.theta_ln_K_op_wt_loc, priors.theta_ln_K_op_wt_scale))
    ln_K_HL_wt = pyro.sample(
        f"{name}_ln_K_HL_wt",
        dist.Normal(priors.theta_ln_K_HL_wt_loc, priors.theta_ln_K_HL_wt_scale))
    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        ln_K_E_wt = pyro.sample(
            f"{name}_ln_K_E_wt",
            dist.Normal(priors.theta_ln_K_E_wt_loc, priors.theta_ln_K_E_wt_scale))

    # ------------------------------------------------------------------
    # MLP weights — zero-centered Normal prior; guide zero-inits locs so
    # the model starts in the perturbative (WT) regime.
    # ------------------------------------------------------------------
    W1 = pyro.sample(f"{name}_nn_W1",
        dist.Normal(jnp.zeros((4, H)), w_scale).to_event(2))
    b1 = pyro.sample(f"{name}_nn_b1",
        dist.Normal(jnp.zeros(H), w_scale).to_event(1))
    W2 = pyro.sample(f"{name}_nn_W2",
        dist.Normal(jnp.zeros((H, 4)), w_scale).to_event(2))
    b2 = pyro.sample(f"{name}_nn_b2",
        dist.Normal(jnp.zeros(4), w_scale).to_event(1))

    # ------------------------------------------------------------------
    # Forward: ΔlogP → ΔΔG per state → Δln_K
    # ------------------------------------------------------------------
    ddG       = _mlp_forward(features, W1, b1, W2, b2)  # (M, 4)
    delta_lnK = ddG @ _PROJ.T                            # (M, 3)

    d_ln_K_op = delta_lnK[:, 0]   # (M,)
    d_ln_K_HL = delta_lnK[:, 1]   # (M,)
    d_ln_K_E  = delta_lnK[:, 2]   # (M,)

    pyro.deterministic(f"{name}_d_ln_K_op", d_ln_K_op)
    pyro.deterministic(f"{name}_d_ln_K_HL", d_ln_K_HL)
    pyro.deterministic(f"{name}_d_ln_K_E",  d_ln_K_E)

    # ------------------------------------------------------------------
    # Assemble per-genotype equilibrium constants
    # K_op, K_HL: scalar WT + additive mutation effects
    # K_E: T-dim WT + scalar mutation effect (uniform across effectors)
    # ------------------------------------------------------------------
    ln_K_op = ln_K_op_wt + d_ln_K_op @ M_mat                          # (G,)
    ln_K_HL = ln_K_HL_wt + d_ln_K_HL @ M_mat                          # (G,)
    ln_K_E  = ln_K_E_wt[:, None] + (d_ln_K_E @ M_mat)[None, :]        # (T, G)

    pyro.deterministic(f"{name}_ln_K_op", ln_K_op)
    pyro.deterministic(f"{name}_ln_K_HL", ln_K_HL)
    pyro.deterministic(f"{name}_ln_K_E",  ln_K_E)

    # ------------------------------------------------------------------
    # Population moments for transformation model
    # ------------------------------------------------------------------
    theta_for_moments = _compute_theta(ln_K_op, ln_K_HL, ln_K_E,
                                       titrant_conc_M, tf_total, op_total)
    mu, sigma = _population_moments(theta_for_moments, data)

    return ThetaParam(ln_K_op=ln_K_op, ln_K_HL=ln_K_HL, ln_K_E=ln_K_E,
                      tf_total=tf_total, op_total=op_total, mu=mu, sigma=sigma)


# ---------------------------------------------------------------------------
# Guide
# ---------------------------------------------------------------------------

def guide(name: str,
          data: Union[GrowthData, BindingData],
          priors: ModelPriors) -> ThetaParam:
    """Variational guide for the LigandMPNN-based lac-dimer theta model."""

    T        = data.num_titrant_name
    M_mat    = jnp.array(data.mut_geno_matrix)
    features = jnp.array(data.ligandmpnn_features)
    H        = int(priors.theta_nn_hidden_size)
    tf_total = priors.theta_tf_total_M
    op_total = priors.theta_op_total_M
    titrant_conc_M = data.titrant_conc / 1000.0       # mM → M

    # ------------------------------------------------------------------
    # Variational parameters for WT K values
    # ------------------------------------------------------------------
    ln_K_op_wt_loc = pyro.param(
        f"{name}_ln_K_op_wt_loc", jnp.array(priors.theta_ln_K_op_wt_loc))
    ln_K_op_wt_scale = pyro.param(
        f"{name}_ln_K_op_wt_scale", jnp.array(1.0),
        constraint=dist.constraints.positive)

    ln_K_HL_wt_loc = pyro.param(
        f"{name}_ln_K_HL_wt_loc", jnp.array(priors.theta_ln_K_HL_wt_loc))
    ln_K_HL_wt_scale = pyro.param(
        f"{name}_ln_K_HL_wt_scale", jnp.array(1.0),
        constraint=dist.constraints.positive)

    ln_K_E_wt_locs = pyro.param(
        f"{name}_ln_K_E_wt_locs", jnp.full(T, priors.theta_ln_K_E_wt_loc))
    ln_K_E_wt_scales = pyro.param(
        f"{name}_ln_K_E_wt_scales", jnp.ones(T),
        constraint=dist.constraints.positive)

    # ------------------------------------------------------------------
    # Variational parameters for MLP weights (zero-init → perturbative start)
    # ------------------------------------------------------------------
    W1_loc   = pyro.param(f"{name}_nn_W1_loc",   jnp.zeros((4, H)))
    W1_scale = pyro.param(f"{name}_nn_W1_scale",  jnp.full((4, H), 0.1),
                          constraint=dist.constraints.positive)
    b1_loc   = pyro.param(f"{name}_nn_b1_loc",   jnp.zeros(H))
    b1_scale = pyro.param(f"{name}_nn_b1_scale",  jnp.full(H, 0.1),
                          constraint=dist.constraints.positive)
    W2_loc   = pyro.param(f"{name}_nn_W2_loc",   jnp.zeros((H, 4)))
    W2_scale = pyro.param(f"{name}_nn_W2_scale",  jnp.full((H, 4), 0.1),
                          constraint=dist.constraints.positive)
    b2_loc   = pyro.param(f"{name}_nn_b2_loc",   jnp.zeros(4))
    b2_scale = pyro.param(f"{name}_nn_b2_scale",  jnp.full(4, 0.1),
                          constraint=dist.constraints.positive)

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------
    ln_K_op_wt = pyro.sample(
        f"{name}_ln_K_op_wt", dist.Normal(ln_K_op_wt_loc, ln_K_op_wt_scale))
    ln_K_HL_wt = pyro.sample(
        f"{name}_ln_K_HL_wt", dist.Normal(ln_K_HL_wt_loc, ln_K_HL_wt_scale))
    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        ln_K_E_wt = pyro.sample(
            f"{name}_ln_K_E_wt", dist.Normal(ln_K_E_wt_locs, ln_K_E_wt_scales))

    W1 = pyro.sample(f"{name}_nn_W1", dist.Normal(W1_loc, W1_scale).to_event(2))
    b1 = pyro.sample(f"{name}_nn_b1", dist.Normal(b1_loc, b1_scale).to_event(1))
    W2 = pyro.sample(f"{name}_nn_W2", dist.Normal(W2_loc, W2_scale).to_event(2))
    b2 = pyro.sample(f"{name}_nn_b2", dist.Normal(b2_loc, b2_scale).to_event(1))

    # ------------------------------------------------------------------
    # Forward pass and assembly (mirrors define_model)
    # ------------------------------------------------------------------
    ddG       = _mlp_forward(features, W1, b1, W2, b2)
    delta_lnK = ddG @ _PROJ.T

    d_ln_K_op = delta_lnK[:, 0]
    d_ln_K_HL = delta_lnK[:, 1]
    d_ln_K_E  = delta_lnK[:, 2]

    ln_K_op = ln_K_op_wt + d_ln_K_op @ M_mat
    ln_K_HL = ln_K_HL_wt + d_ln_K_HL @ M_mat
    ln_K_E  = ln_K_E_wt[:, None] + (d_ln_K_E @ M_mat)[None, :]

    theta_for_moments = _compute_theta(ln_K_op, ln_K_HL, ln_K_E,
                                       titrant_conc_M, tf_total, op_total)
    mu, sigma = _population_moments(theta_for_moments, data)

    return ThetaParam(ln_K_op=ln_K_op, ln_K_HL=ln_K_HL, ln_K_E=ln_K_E,
                      tf_total=tf_total, op_total=op_total, mu=mu, sigma=sigma)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def get_hyperparameters() -> Dict[str, Any]:
    p = {}
    p["theta_ln_K_op_wt_loc"]   = 23.0   # 2.3 (nM⁻¹) + ln(1e9) ≈ 23.0 (M⁻¹)
    p["theta_ln_K_op_wt_scale"] = 2.0
    p["theta_ln_K_HL_wt_loc"]   = -9.0   # dimensionless, unchanged
    p["theta_ln_K_HL_wt_scale"] = 3.0
    p["theta_ln_K_E_wt_loc"]    = 33.4   # -8.0 (nM⁻²) + ln(1e18) ≈ 33.4 (M⁻²)
    p["theta_ln_K_E_wt_scale"]  = 3.0
    p["theta_tf_total_M"]       = 6.5e-7  # 650 nM
    p["theta_op_total_M"]       = 2.5e-8  # 25 nM
    p["theta_nn_hidden_size"]   = _DEFAULT_HIDDEN_SIZE
    p["theta_nn_w_scale"]       = 1.0
    return p


def get_guesses(name: str, data: Union[GrowthData, BindingData]) -> Dict[str, Any]:
    T = data.num_titrant_name
    H = _DEFAULT_HIDDEN_SIZE
    g = {}
    g[f"{name}_ln_K_op_wt"] = jnp.array(23.0)
    g[f"{name}_ln_K_HL_wt"] = jnp.array(-9.0)
    g[f"{name}_ln_K_E_wt"]  = jnp.full(T, 33.4)
    g[f"{name}_nn_W1"]      = jnp.zeros((4, H))
    g[f"{name}_nn_b1"]      = jnp.zeros(H)
    g[f"{name}_nn_W2"]      = jnp.zeros((H, 4))
    g[f"{name}_nn_b2"]      = jnp.zeros(4)
    return g


def get_priors() -> ModelPriors:
    return ModelPriors(**get_hyperparameters())


def get_extract_specs(ctx):
    geno_dim = ctx.growth_tm.tensor_dim_names.index("genotype")
    num_genotype = len(ctx.growth_tm.tensor_dim_labels[geno_dim])
    num_mut = len(ctx.mut_labels)

    geno_df = (ctx.growth_tm.df[["genotype", "genotype_idx"]]
               .drop_duplicates().copy())
    geno_df["map_geno"] = geno_df["genotype_idx"]
    specs = [dict(
        input_df=geno_df,
        params_to_get=["ln_K_op", "ln_K_HL"],
        map_column="map_geno",
        get_columns=["genotype"],
        in_run_prefix="theta_",
    )]

    theta_KE_df = (ctx.growth_tm.df[["genotype", "titrant_name",
                                     "genotype_idx", "titrant_name_idx"]]
                   .drop_duplicates().copy())
    titrant_dim = ctx.growth_tm.tensor_dim_names.index("titrant_name")
    num_titrant = len(ctx.growth_tm.tensor_dim_labels[titrant_dim])
    theta_KE_df["map_theta_KE"] = (theta_KE_df["titrant_name_idx"] * num_genotype
                                   + theta_KE_df["genotype_idx"])
    specs.append(dict(
        input_df=theta_KE_df,
        params_to_get=["ln_K_E"],
        map_column="map_theta_KE",
        get_columns=["genotype", "titrant_name"],
        in_run_prefix="theta_",
    ))

    mut_df = pd.DataFrame({
        "mutation": ctx.mut_labels,
        "map_mut": range(num_mut),
    })
    specs.append(dict(
        input_df=mut_df,
        params_to_get=["d_ln_K_op", "d_ln_K_HL", "d_ln_K_E"],
        map_column="map_mut",
        get_columns=["mutation"],
        in_run_prefix="theta_",
    ))

    return specs
