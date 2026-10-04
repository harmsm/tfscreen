"""
A MAP fit with level tube offsets, run in three stages.

A cold MAP with ``sample_offset: level`` can settle in a mode where the tube
offsets carry the population's growth: Adam moves every parameter by about
one step size per step, so in the first window it moves the offsets and the
growth rates k and m (prior SD 0.002 per minute) by whole units, and the
step-size cuts then freeze that mode. On the dev-data fit the offsets landed
near ±2.8; starting from a MAP without offsets plus each tube's best offset
was 1.2e5 nats better (``planning/dev-data/real_fit``, 2026-10-01). The
stages do that start automatically:

1. **No offsets.** The MAP with every tube offset held at 0.
2. **Offsets alone.** Every other site held at stage 1's MAP; the offsets
   are the only latents. Given the rest, the tubes do not couple, so this is
   each tube's best offset (what ``small_offsets.py`` computed by hand).
3. **Joint.** The full MAP from stage 1's point plus stage 2's offsets, at a
   small step size so the first window cannot carry the start away.

While the offsets are held, a learned offset SD (``sigma_fixed`` = 0) is
held too, at the prior's scale: with every offset at 0 the SD's MAP is 0 and
the density is unbounded.

Each stage is an ordinary MAP on a model conditioned on its held values
(``HeldModel``), so it writes its own checkpoint, convergence record and
``_params.npz`` under ``{out_prefix}_stage1`` and ``{out_prefix}_stage2``;
stage 3 writes the run's own ``{out_prefix}_*`` files. A finished stage
(its ``_params.npz`` exists) is reused; an interrupted one resumes from its
checkpoint.
"""

import os

import jax.numpy as jnp
import numpy as np
from numpyro import handlers

STAGE_CHOICES = ("auto", "on", "off")

# Stage 3's starting step size: about one step size of movement per Adam
# step, small against the offsets' SD (0.17) and the growth rates' prior SD.
DEFAULT_STAGED_STEP_SIZE = 1e-4


class HeldModel:
    """
    A model with some sample sites held at fixed values.

    Wraps a ``ModelOrchestrator`` (or anything ``RunInference`` accepts):
    ``jax_model`` is the wrapped model conditioned on ``held``, so an
    autoguide built on it has no parameters for those sites and the loss
    counts their (constant) log density at the held values. Every other
    attribute is the wrapped model's, including ``jax_model_guide``, which
    ``RunInference.site_values`` reads only for parameter names: fit a
    HeldModel with an autoguide (MAP), never with the component guide.

    Parameters
    ----------
    model : object
        The model to wrap.
    held : dict
        Site name to constrained value. Per-genotype values must be
        library-sized, as a MAP's ``_params.npz`` stores them.
    """

    def __init__(self, model, held):
        self._model = model
        # jax arrays: the model indexes per-genotype values with traced
        # batch indices, which a numpy array cannot take
        self._held = {k: jnp.asarray(v) for k, v in held.items()}
        self._jax_model = handlers.condition(model.jax_model, data=self._held)

    @property
    def held(self):
        return dict(self._held)

    @property
    def jax_model(self):
        return self._jax_model

    def __getattr__(self, name):
        return getattr(self._model, name)


def offset_sites(orchestrator):
    """
    The level offset's site names and the value its SD is held at.

    Returns
    -------
    offset_site, sigma_site : str
    sigma_hold : float or None
        The SD to hold while the offsets are held, or None if the SD is
        already fixed (``sigma_fixed`` > 0, a deterministic site).
    """
    priors = orchestrator.priors.growth.sample_offset
    sigma_hold = None if float(priors.sigma_fixed) > 0 else float(priors.sigma_prior_scale)
    return "sample_offset_offset", "sample_offset_sigma", sigma_hold


def use_stages(stage_offsets, orchestrator, analysis_method,
               checkpoint_file=None, init_from=None):
    """
    Whether ``tfs-fit-model`` runs the staged MAP.

    ``auto`` stages a fresh MAP (no checkpoint to resume, no ``init_from``
    start) of a model with level tube offsets. ``on`` requires such a fit;
    ``off`` never stages.
    """
    if stage_offsets not in STAGE_CHOICES:
        raise ValueError(f"stage_offsets must be one of {STAGE_CHOICES}; got "
                         f"{stage_offsets!r}.")
    if stage_offsets == "off":
        return False
    level = orchestrator.settings.get("sample_offset") == "level"
    fresh_map = (analysis_method == "map" and checkpoint_file is None
                 and init_from is None)
    if stage_offsets == "auto":
        return level and fresh_map
    if not level:
        raise ValueError("stage_offsets='on' needs sample_offset 'level'.")
    if not fresh_map:
        raise ValueError("stage_offsets='on' needs analysis_method 'map' with "
                         "no checkpoint_file and no init_from.")
    return True


def read_params(path):
    """A MAP ``_params.npz`` as {site: constrained value}."""
    with np.load(path) as z:
        return {k[:-len("_auto_loc")]: np.asarray(z[k])
                for k in z.files if k.endswith("_auto_loc")}


def _stage(run_map, make_ri, model, held, init_values, prefix, label, **kwargs):
    """Run one stage, or reuse it if its params file is already there."""
    params_file = f"{prefix}_params.npz"
    if os.path.exists(params_file):
        print(f"{label}: reusing {params_file}", flush=True)
        return read_params(params_file)
    checkpoint = f"{prefix}_checkpoint.pkl"
    resume = checkpoint if os.path.exists(checkpoint) else None
    print(f"{label}: {'resuming ' + checkpoint if resume else 'starting'} "
          f"({len(held)} site(s) held)", flush=True)
    ri = make_ri(HeldModel(model, held))
    run_map(ri,
            init_values=None if resume else ri.site_values(init_values),
            checkpoint_file=resume,
            out_prefix=prefix,
            label=label,
            **kwargs)
    return read_params(params_file)


def run_staged_map(orchestrator, make_ri, run_map, guesses, out_prefix,
                   map_kwargs, stage_map_kwargs=None,
                   staged_step_size=DEFAULT_STAGED_STEP_SIZE):
    """
    Run the three-stage MAP (see the module docstring).

    Parameters
    ----------
    orchestrator : ModelOrchestrator
        The model, with ``sample_offset: level``.
    make_ri : callable
        ``make_ri(model) -> RunInference`` for a (possibly held) model.
    run_map : callable
        ``tfs-fit-model``'s ``_run_map``.
    guesses : dict
        The configured guesses (any form ``RunInference.site_values`` takes).
    out_prefix : str
        The run's prefix. Stages 1 and 2 write ``{out_prefix}_stage1_*`` and
        ``{out_prefix}_stage2_*``; stage 3 writes ``{out_prefix}_*``.
    map_kwargs : dict
        ``_run_map`` keyword arguments for stage 3 (optimizer and
        convergence settings); its ``adam_step_size`` is replaced by
        ``staged_step_size``.
    stage_map_kwargs : dict or None
        ``_run_map`` keyword arguments for stages 1 and 2 (default
        ``map_kwargs``).
    staged_step_size : float
        Stage 3's starting step size.

    Returns
    -------
    The stage-3 ``_run_map`` result: ``(svi_state, params, converged)``.
    """
    if stage_map_kwargs is None:
        stage_map_kwargs = map_kwargs
    offset_site, sigma_site, sigma_hold = offset_sites(orchestrator)

    # Stage 1: offsets (and a learned SD) held.
    held1 = {offset_site: _offset_zeros(orchestrator)}
    if sigma_hold is not None:
        held1[sigma_site] = np.asarray(sigma_hold, dtype=float)
    stage1 = _stage(run_map, make_ri, orchestrator, held1, guesses,
                    f"{out_prefix}_stage1", "Stage 1 (no tube offsets)",
                    **stage_map_kwargs)

    # Stage 2: every other site held at stage 1; the offsets start at 0.
    held2 = {k: v for k, v in stage1.items() if k != offset_site}
    if sigma_hold is not None:
        held2[sigma_site] = np.asarray(sigma_hold, dtype=float)
    stage2 = _stage(run_map, make_ri, orchestrator, held2,
                    {offset_site: held1[offset_site]},
                    f"{out_prefix}_stage2", "Stage 2 (tube offsets alone)",
                    **stage_map_kwargs)
    off = np.asarray(stage2[offset_site])
    print(f"  stage 2 offsets: SD {off.std():.3f}, range {off.min():+.2f} to "
          f"{off.max():+.2f}", flush=True)

    # Stage 3: joint, from stage 1 + stage 2's offsets.
    start = {**guesses, **{f"{k}_auto_loc": v for k, v in stage1.items()},
             f"{offset_site}_auto_loc": off}
    ri = make_ri(orchestrator)
    kwargs = dict(map_kwargs)
    kwargs["adam_step_size"] = staged_step_size
    print(f"Stage 3 (joint, step size {staged_step_size:g})", flush=True)
    return run_map(ri,
                   init_values=ri.site_values(start),
                   out_prefix=out_prefix,
                   label="Staged MAP",
                   **kwargs)


def _offset_zeros(orchestrator):
    """Zeros of the per-tube offset's shape (one per tube of the grid)."""
    d = orchestrator.data.growth
    n = (d.num_replicate * d.num_time * d.num_condition_pre
         * d.num_condition_sel * d.num_titrant_name * d.num_titrant_conc)
    return np.zeros(n, dtype=float)
