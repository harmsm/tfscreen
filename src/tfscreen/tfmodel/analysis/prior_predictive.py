import numpy as np
import jax
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive

from tfscreen.tfmodel.analysis.prediction import predict


def draw_prior(orchestrator, rng_key=0, num_draws=1):
    """
    Draw samples from the model prior (no data conditioning).

    Runs ``Predictive`` with ``posterior_samples=None`` to obtain joint draws
    from the prior.  Observed sites (those with ``obs=`` in the model) are
    excluded from the return because they hold the training data, not new
    samples.

    Parameters
    ----------
    orchestrator : ModelOrchestrator
        Fully initialised model, as returned by ``read_configuration``.
    rng_key : int or jax.random.PRNGKey, optional
        Random seed.  An ``int`` is converted to a PRNGKey.  Default 0.
    num_draws : int, optional
        Number of independent joint draws from the prior.  Default 1.

    Returns
    -------
    predictions : dict
        Deterministic model sites keyed by name (e.g. ``'growth_pred'``,
        ``'theta_growth_pred'``).  Each value is a ``numpy.ndarray`` with
        shape ``(num_draws, *site_shape)``.
    latent_params : dict
        All non-observed stochastic sites drawn from the prior, in the same
        format as posterior sample dicts produced by
        ``tfs-sample-posterior``.  Each value has shape
        ``(num_draws, *param_shape)``.  This dict can be passed directly to
        ``predict()`` or ``tfs-predict-growth``.
    """
    if isinstance(rng_key, int):
        rng_key = jax.random.PRNGKey(rng_key)

    # Trace the model once to classify all sites.
    seeded = seed(orchestrator.jax_model, rng_seed=0)
    tr = trace(seeded).get_trace(data=orchestrator.data, priors=orchestrator.priors)

    observed = {
        name for name, site in tr.items()
        if site["type"] == "sample" and site.get("is_observed", False)
    }
    deterministic = {
        name for name, site in tr.items()
        if site["type"] == "deterministic"
    }

    return_sites = sorted(set(tr.keys()) - observed)

    predictive = Predictive(
        orchestrator.jax_model,
        posterior_samples=None,
        num_samples=num_draws,
        return_sites=return_sites,
    )
    raw = predictive(rng_key, data=orchestrator.data, priors=orchestrator.priors)

    predictions = {k: np.array(v) for k, v in raw.items() if k in deterministic}
    latent_params = {k: np.array(v) for k, v in raw.items() if k not in deterministic}

    return predictions, latent_params


def growth_df_from_prior(orchestrator, latent_params, draw_idx=0, noise_rng=None):
    """
    Build a synthetic growth DataFrame from one prior draw.

    Maps ``latent_params`` through the forward model to obtain the
    deterministic ``growth_pred`` values, then writes them into a copy of
    ``orchestrator.growth_df``.  The result has the same rows and structure
    as the original DataFrame and is ready for ``tfs-fit-model``.

    Noise is added using ``orchestrator.growth_df['ln_cfu_std']`` as the
    per-observation standard deviation.  This preserves the original
    measurement-error structure while replacing the signal with synthetic
    predictions.

    Parameters
    ----------
    orchestrator : ModelOrchestrator
    latent_params : dict
        Output of ``draw_prior``.  Values have shape ``(num_draws, ...)``.
    draw_idx : int, optional
        Which draw to use.  Default 0.
    noise_rng : numpy.random.Generator or None, optional
        If provided, adds ``Normal(0, ln_cfu_std)`` noise to each row.
        If ``None``, returns the deterministic predictions.

    Returns
    -------
    pandas.DataFrame
        Copy of ``orchestrator.growth_df`` with ``ln_cfu`` replaced by
        synthetic values.
    """
    # Slice to single draw so predict() treats it as one "posterior sample".
    single_draw = {k: v[draw_idx : draw_idx + 1] for k, v in latent_params.items()}

    merge_keys = [
        "replicate", "condition_pre", "condition_sel",
        "titrant_name", "genotype", "t_pre", "t_sel", "titrant_conc",
    ]

    pred_df = predict(
        orchestrator,
        single_draw,
        predict_sites=["growth_pred"],
        q_to_get=[0.5],
        num_samples=None,
    )

    # pred_df may contain extra rows (expanded grid); merge to keep original rows.
    out_df = orchestrator.growth_df.drop(columns=["ln_cfu"]).merge(
        pred_df[merge_keys + ["q0.5"]],
        on=merge_keys,
        how="left",
    )
    out_df = out_df.rename(columns={"q0.5": "ln_cfu"})

    if noise_rng is not None:
        out_df["ln_cfu"] = (
            out_df["ln_cfu"].values
            + noise_rng.normal(0.0, out_df["ln_cfu_std"].values)
        )

    return out_df
