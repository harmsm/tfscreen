"""
Relative Hill curves: a wt-relative growth variable X instead of theta.

Roadmap step 5 (``planning/analysis-roadmap.md``). Growth data alone fix an
occupancy-like variable only up to an affine map (``X -> a X + b`` is absorbed
by each condition's ``k`` and ``m``), so this component does not claim
absolute theta. Each genotype's X follows hill_geno's curve,

    X_g(c) = X_low_g + (X_high_g - X_low_g) * sigmoid(n_g (ln c - ln K_g)),

with real-valued baselines (a mutant may repress more than wt, X > 1, or
less than wt does at saturating IPTG, X < 0). The gauge is fixed on wt (D4):

    X_wt(c_lo) = 1,   X_wt(c_hi) = 0,

at the two gauge concentrations ``data.theta_gauge_log_conc`` (an
orchestrator setting, ``theta_gauge_conc``; default the lowest and highest
measured concentration). wt keeps its K and n free; its baselines follow from
them. Then ``k_c`` is wt's growth at ``c_hi`` and ``m_c`` wt's growth change
between the two, in every condition, and X is shared by all conditions (D5).

Only valid with activity 'fixed', theta_rescale 'passthrough', linear
condition growth, no theta noise, no binding data and, under the congression
mixture, the 'max' theta rule (C1); ModelOrchestrator enforces these.

Outputs are on the X scale (``THETA_SCALE = "X"``): the deterministic sites
are ``theta_X_low``/``theta_X_high`` and tfs-predict-theta labels its rows.
"""
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass, field

from tfscreen.tfmodel.data_class import DataClass

# What run_model returns; read by tfs-predict-theta and the X-scale checks.
THETA_SCALE = "X"

# Floor on wt's occupancy change between the gauge concentrations. The gauge
# divides by it; a wt curve that barely moves across the range would
# otherwise send wt's baselines to infinity.
_MIN_GAUGE_SPAN = 1e-3


@dataclass(frozen=True)
class ModelPriors:
    """
    Hyperpriors for the relative Hill model.

    ``X_low`` is X at no titrant, ``X_delta`` = ``X_high - X_low``. The
    defaults are centered on wt under the gauge (1 and -1).
    """

    theta_X_low_hyper_loc_loc: float
    theta_X_low_hyper_loc_scale: float
    theta_X_low_hyper_scale: float
    theta_X_delta_hyper_loc_loc: float
    theta_X_delta_hyper_loc_scale: float
    theta_X_delta_hyper_scale: float

    theta_log_hill_K_hyper_loc_loc: float
    theta_log_hill_K_hyper_loc_scale: float
    theta_log_hill_K_hyper_scale: float

    theta_log_hill_n_hyper_loc_loc: float
    theta_log_hill_n_hyper_loc_scale: float
    theta_log_hill_n_hyper_scale: float
    # > 0: hold the population SD of log(hill_n) here (hill_geno's field of
    # the same name explains why)
    theta_log_hill_n_hyper_scale_fixed: float = field(pytree_node=False,
                                                      default=0.0)


@dataclass(frozen=True)
class ThetaParam:
    """
    Per-genotype relative Hill parameters, each shape ``(num_titrant_name,
    num_genotype)``, library-ordered. wt's ``X_low``/``X_high`` are set by
    the gauge.
    """

    X_low: jnp.ndarray
    X_high: jnp.ndarray
    log_hill_K: jnp.ndarray
    hill_n: jnp.ndarray


def gauge_baselines(log_hill_K, hill_n, gauge_log_conc):
    """
    wt's X baselines that put ``X = 1`` at ``c_lo`` and ``X = 0`` at ``c_hi``.

    Parameters
    ----------
    log_hill_K, hill_n : jnp.ndarray
        wt's Hill constant (log) and coefficient, any matching shape.
    gauge_log_conc : jnp.ndarray
        ``(ln c_lo, ln c_hi)``.

    Returns
    -------
    X_low, X_high : jnp.ndarray
    """
    occ_lo = jax.nn.sigmoid(hill_n * (gauge_log_conc[0] - log_hill_K))
    occ_hi = jax.nn.sigmoid(hill_n * (gauge_log_conc[1] - log_hill_K))
    span = jnp.maximum(occ_hi - occ_lo, _MIN_GAUGE_SPAN)
    X_low = 1.0 + occ_lo / span
    X_high = X_low - 1.0 / span
    return X_low, X_high


def _assemble(hyper, offsets, data):
    """Per-genotype parameters from hyperparameters and offsets, wt gauged."""
    (low_loc, low_scale, delta_loc, delta_scale,
     K_loc, K_scale, n_loc, n_scale) = hyper
    low_off, delta_off, K_off, n_off = offsets

    X_low = low_loc[:, None] + low_off * low_scale[:, None]
    X_delta = delta_loc[:, None] + delta_off * delta_scale[:, None]
    log_hill_K = K_loc[:, None] + K_off * K_scale[:, None]
    hill_n = jnp.exp(n_loc[:, None] + n_off * n_scale[:, None])

    wt = data.wt_indexes[0]
    wt_low, wt_high = gauge_baselines(log_hill_K[:, wt], hill_n[:, wt],
                                      data.theta_gauge_log_conc)
    is_wt = jnp.zeros(data.num_genotype, dtype=bool).at[data.wt_indexes].set(True)
    X_low = jnp.where(is_wt[None, :], wt_low[:, None], X_low)
    X_high = jnp.where(is_wt[None, :], wt_high[:, None], X_low + X_delta)

    return ThetaParam(X_low=X_low, X_high=X_high,
                      log_hill_K=log_hill_K, hill_n=hill_n)


_HYPER = ("X_low", "X_delta", "log_hill_K", "log_hill_n")
_OFFSETS = ("X_low_offset", "X_delta_offset", "log_hill_K_offset",
            "log_hill_n_offset")


def _n_fixed(priors):
    return float(getattr(priors, "theta_log_hill_n_hyper_scale_fixed", 0.0))


def define_model(name: str,
                 data: DataClass,
                 priors: ModelPriors) -> ThetaParam:
    """
    Hierarchical relative Hill model with per-titrant hyperpriors and
    library-sized per-genotype offsets (``run_model`` slices by
    ``data.batch_idx[data.geno_theta_idx]``).
    """
    T = data.num_titrant_name

    hyper = []
    with pyro.plate(f"{name}_hyper_plate", T, dim=-1):
        for h in _HYPER:
            hyper.append(pyro.sample(
                f"{name}_{h}_hyper_loc",
                dist.Normal(getattr(priors, f"theta_{h}_hyper_loc_loc"),
                            getattr(priors, f"theta_{h}_hyper_loc_scale"))))
            if h == "log_hill_n" and _n_fixed(priors) > 0:
                hyper.append(pyro.deterministic(f"{name}_{h}_hyper_scale",
                                                jnp.full(T, _n_fixed(priors))))
            else:
                hyper.append(pyro.sample(
                    f"{name}_{h}_hyper_scale",
                    dist.HalfNormal(getattr(priors, f"theta_{h}_hyper_scale"))))

    with pyro.plate(f"{name}_titrant_name_plate", T, dim=-2):
        with pyro.plate(f"{name}_genotype_plate", data.num_genotype, dim=-1):
            offsets = [pyro.sample(f"{name}_{o}", dist.Normal(0.0, 1.0))
                       for o in _OFFSETS]

    theta_param = _assemble(hyper, offsets, data)

    pyro.deterministic(f"{name}_X_low", theta_param.X_low)
    pyro.deterministic(f"{name}_X_high", theta_param.X_high)
    pyro.deterministic(f"{name}_log_hill_K", theta_param.log_hill_K)
    pyro.deterministic(f"{name}_hill_n", theta_param.hill_n)

    return theta_param


def guide(name: str,
          data: DataClass,
          priors: ModelPriors) -> ThetaParam:
    """Mean-field guide mirroring ``define_model``."""
    T = data.num_titrant_name
    G = data.num_genotype

    hyper = []
    with pyro.plate(f"{name}_hyper_plate", T, dim=-1):
        for h in _HYPER:
            loc_loc = pyro.param(
                f"{name}_{h}_hyper_loc_loc",
                jnp.full(T, getattr(priors, f"theta_{h}_hyper_loc_loc")))
            loc_scale = pyro.param(
                f"{name}_{h}_hyper_loc_scale",
                jnp.full(T, getattr(priors, f"theta_{h}_hyper_loc_scale")),
                constraint=dist.constraints.greater_than(1e-4))
            hyper.append(pyro.sample(f"{name}_{h}_hyper_loc",
                                     dist.Normal(loc_loc, loc_scale)))
            if h == "log_hill_n" and _n_fixed(priors) > 0:
                hyper.append(jnp.full(T, _n_fixed(priors)))
                continue
            scale_loc = pyro.param(f"{name}_{h}_hyper_scale_loc",
                                   jnp.full(T, -1.0))
            scale_scale = pyro.param(
                f"{name}_{h}_hyper_scale_scale", jnp.full(T, 0.1),
                constraint=dist.constraints.greater_than(1e-4))
            hyper.append(pyro.sample(f"{name}_{h}_hyper_scale",
                                     dist.LogNormal(scale_loc, scale_scale)))

    locs = [pyro.param(f"{name}_{o}_locs", jnp.zeros((T, G), dtype=float))
            for o in _OFFSETS]
    scales = [pyro.param(f"{name}_{o}_scales", jnp.ones((T, G), dtype=float),
                         constraint=dist.constraints.positive)
              for o in _OFFSETS]

    with pyro.plate(f"{name}_titrant_name_plate", T, dim=-2):
        with pyro.plate(f"{name}_genotype_plate", G, dim=-1):
            offsets = [pyro.sample(f"{name}_{o}", dist.Normal(l, s))
                       for o, l, s in zip(_OFFSETS, locs, scales)]

    return _assemble(hyper, offsets, data)


def run_model(theta_param: ThetaParam, data: DataClass) -> jnp.ndarray:
    """
    X at every titrant concentration for the batch's genotypes.

    Returns shape ``(T, C, G_subset)``, or ``(1, 1, 1, 1, T, C, G_subset)``
    when ``data.scatter_theta == 1`` (the hill_geno contract).
    """
    geno_idx = data.batch_idx[data.geno_theta_idx]
    X_low = theta_param.X_low[:, None, geno_idx]
    X_high = theta_param.X_high[:, None, geno_idx]
    log_hill_K = theta_param.log_hill_K[:, None, geno_idx]
    hill_n = theta_param.hill_n[:, None, geno_idx]

    log_titrant = data.log_titrant_conc[None, :, None]
    occupancy = jax.nn.sigmoid(hill_n * (log_titrant - log_hill_K))
    X = X_low + (X_high - X_low) * occupancy

    if data.scatter_theta == 1:
        X = X[None, None, None, None, :, :, :]

    return X


def get_population_moments(theta_param: ThetaParam, data: DataClass) -> tuple:
    """No logit-space moments: X is not an occupancy (theta noise is refused)."""
    return None, None


def get_hyperparameters() -> Dict[str, Any]:
    """Default hyperparameters (X centered on wt's gauge; K/n as hill_geno)."""
    return {
        "theta_X_low_hyper_loc_loc": 1.0,
        "theta_X_low_hyper_loc_scale": 0.5,
        "theta_X_low_hyper_scale": 1.0,
        "theta_X_delta_hyper_loc_loc": -1.0,
        "theta_X_delta_hyper_loc_scale": 0.5,
        "theta_X_delta_hyper_scale": 1.0,
        "theta_log_hill_K_hyper_loc_loc": -4.1,
        "theta_log_hill_K_hyper_loc_scale": 1.0,
        "theta_log_hill_K_hyper_scale": 0.1,
        "theta_log_hill_n_hyper_loc_loc": 0.7,
        "theta_log_hill_n_hyper_loc_scale": 0.5,
        "theta_log_hill_n_hyper_scale": 1.0,
        "theta_log_hill_n_hyper_scale_fixed": 0.0,
    }


def get_priors() -> ModelPriors:
    return ModelPriors(**get_hyperparameters())


def get_guesses(name: str, data: DataClass) -> Dict[str, Any]:
    """Starting values: wt-like baselines, K at the median concentration."""
    log_conc = np.array(data.log_titrant_conc)
    finite = log_conc[np.isfinite(log_conc)]
    log_K_guess = float(np.median(finite)) if len(finite) > 0 else -4.1

    T = data.num_titrant_name
    G = data.num_genotype

    guesses = {
        f"{name}_X_low_hyper_loc": jnp.full(T, 1.0),
        f"{name}_X_low_hyper_scale": jnp.full(T, 0.5),
        f"{name}_X_delta_hyper_loc": jnp.full(T, -1.0),
        f"{name}_X_delta_hyper_scale": jnp.full(T, 0.5),
        f"{name}_log_hill_K_hyper_loc": jnp.full(T, log_K_guess),
        f"{name}_log_hill_K_hyper_scale": jnp.full(T, 1.0),
        f"{name}_log_hill_n_hyper_loc": jnp.full(T, 0.7),
        f"{name}_log_hill_n_hyper_scale": jnp.full(T, 0.3),
    }
    for o in _OFFSETS:
        guesses[f"{name}_{o}"] = jnp.zeros((T, G), dtype=float)
    return guesses


def get_extract_specs(ctx):
    return [dict(
        input_df=ctx.growth_tm.df,
        params_to_get=["hill_n", "log_hill_K", "X_high", "X_low"],
        map_column="map_theta_group",
        get_columns=["genotype", "titrant_name"],
        in_run_prefix="theta_",
    )]


_ZERO_CONC_VALUE = 1e-20


def _log_conc(conc):
    conc = np.asarray(conc, dtype=float).copy()
    conc[conc == 0] = _ZERO_CONC_VALUE
    return np.log(conc)


def build_calc_df(model, manual_titrant_df):
    """Concentration grid for X extraction (same grid as hill_geno)."""
    from tfscreen.tfmodel.generative.components.theta.hill_geno import (
        build_calc_df as _hill_build_calc_df,
    )
    return _hill_build_calc_df(model, manual_titrant_df)


def compute_theta_samples(calc_df, param_posteriors):
    """Posterior X samples, shape ``(S, len(calc_df))``."""
    from tfscreen.tfmodel.inference.posteriors import get_posterior_samples

    indices = calc_df["map_theta_group"].values.astype(int)
    log_titrant = _log_conc(calc_df["titrant_conc"].values)[None, :]

    def _load_flat(key):
        v = get_posterior_samples(param_posteriors, key)
        if hasattr(v, "shape") and not hasattr(v, "reshape"):
            v = v[:]
        return v.reshape(v.shape[0], -1)

    h_n = _load_flat("theta_hill_n")[:, indices]
    l_K = _load_flat("theta_log_hill_K")[:, indices]
    x_h = _load_flat("theta_X_high")[:, indices]
    x_l = _load_flat("theta_X_low")[:, indices]

    occupancy = 1.0 / (1.0 + np.exp(-h_n * (log_titrant - l_K)))
    return x_l + (x_h - x_l) * occupancy


def predict_unmeasured(target_genotypes,
                       titrant_names,
                       manual_titrant_df,
                       mut_labels,
                       pair_labels,
                       param_posteriors,
                       q_to_get):
    """
    Population-mean X for genotypes not seen in training (no per-mutation
    structure, as for hill_geno).
    """
    from tfscreen.tfmodel.inference.posteriors import get_posterior_samples
    from tfscreen.tfmodel.analysis.predict_unmeasured import (
        _build_prediction_grid,
    )

    calc_df, _, titrant_idx = _build_prediction_grid(
        list(target_genotypes), titrant_names, manual_titrant_df)

    def _load(key):
        v = get_posterior_samples(param_posteriors, key)
        if hasattr(v, "shape") and not hasattr(v, "reshape"):
            v = v[:]
        return np.array(v)

    X_low = _load("theta_X_low_hyper_loc")[:, titrant_idx]
    X_high = X_low + _load("theta_X_delta_hyper_loc")[:, titrant_idx]
    l_K = _load("theta_log_hill_K_hyper_loc")[:, titrant_idx]
    h_n = np.exp(_load("theta_log_hill_n_hyper_loc"))[:, titrant_idx]

    log_conc = _log_conc(calc_df["titrant_conc"].values)[None, :]
    occupancy = 1.0 / (1.0 + np.exp(-h_n * (log_conc - l_K)))
    samples = X_low + (X_high - X_low) * occupancy

    result_df = calc_df[["genotype", "titrant_name", "titrant_conc"]].copy()
    for q_name, q_val in q_to_get.items():
        result_df[q_name] = np.quantile(samples, q_val, axis=0)
    return result_df


def x_scale_truth(theta, theta_wt_lo, theta_wt_hi,
                  activity=1.0, activity_wt=1.0):
    """
    Map simulated occupancy onto the X gauge (roadmap C8).

    With growth linear in ``activity * theta`` and the gauge ``X_wt(c_lo) =
    1``, ``X_wt(c_hi) = 0``:

        X = (A theta - A_wt theta_wt(c_hi)) / (A_wt (theta_wt(c_lo) - theta_wt(c_hi)))

    Parameters are arrays or scalars that broadcast together.
    """
    theta = np.asarray(theta, dtype=float)
    span = activity_wt * (np.asarray(theta_wt_lo) - np.asarray(theta_wt_hi))
    return (activity * theta - activity_wt * np.asarray(theta_wt_hi)) / span


def x_scale_growth_truth(k, m, theta_wt_lo, theta_wt_hi, activity_wt=1.0):
    """
    A condition's (k, m) on the X gauge, from its (k, m) on the theta scale.

    ``g = k + m A theta`` with ``A theta = A_wt theta_wt(c_hi) + X * span``
    gives ``k_X = k + m A_wt theta_wt(c_hi)`` and ``m_X = m * span``.
    """
    span = activity_wt * (theta_wt_lo - theta_wt_hi)
    return k + m * activity_wt * theta_wt_hi, m * span
