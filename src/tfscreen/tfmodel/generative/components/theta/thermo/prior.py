"""
Per-mutation per-structure ΔΔG sampling around structural prior means.

This module provides the MODEL side of the ΔΔG prior, a centered two-level
hierarchy per structure s:

    σ_s[s]    ~ LogNormal(sigma_loc, sigma_scale)     (one per structure)
    ΔΔG[m, s] ~ Normal(prior_mean[m, s], σ_s[s])      (the sampled site)

``prior_mean`` is the ΔΔG table for PddG and the structure MLP's output for
PnnC.  σ_s is the SD of the true ΔΔG about the prior mean: how far the
structural prediction is from the truth.  σ_s → 0 pins ΔΔG to the prior
means; a large σ_s lets the data alone set ΔΔG.

Why centered, and why σ_s is a sample site
------------------------------------------
Until 2026-10-11 σ_s was a ``pyro.param`` with no prior and ΔΔG was written
non-centered, ``prior_mean + σ_s · offset`` with ``offset ~ N(0, 1)``.  Under
MAP that is degenerate: the likelihood sees only ΔΔG, so raising σ_s while
shrinking the offsets lowers the offset penalty at no cost.  In a pilot
σ_s ran to 4-37 kT whatever the true noise (0-3.4 kT), the prior went flat,
and PddG recovered every constant worse than the no-prior PK model.  In the
centered form the density of ΔΔG carries −M·log σ_s − Σ_m r²/(2σ_s²), with
r = ΔΔG − prior_mean, which for fixed ΔΔG peaks at σ_s = RMS(r): σ_s tracks
the scatter of the fitted ΔΔG about the prior means.

Prior on σ_s.  σ_s is a positive scale whose plausible range spans two
decades (a structural prediction good to 0.1 kT, or one off by 10 kT), so a
prior symmetric in log σ_s is the natural weak choice.  The defaults
(``theta_ddG_sigma_loc = 0``, median 1 kT, and ``theta_ddG_sigma_scale =
1.2``) put 95% of the mass on 0.095-10.5 kT.  With hundreds of mutations per
structure the −M·log σ_s term outweighs this prior, so it mainly matters
for structures the data barely inform.  A HalfNormal was not used: its mode
is σ_s = 0, the wrong place to push a scale under MAP.

Known limits of joint MAP.  The joint density is unbounded as any σ_s → 0
with that structure's residuals → 0, so MAP finds an interior σ_s only as a
local optimum, and two things can push it to the boundary.  (1) Weak data:
for a ΔΔG direction the likelihood measures with standard error e, the MAP
fixed point of σ_s is nonzero only when the true σ_s exceeds about √3·e.
Below that σ_s drifts toward 0 (the LogNormal slows but does not stop it)
and ΔΔG goes to the prior means.  (2) Unidentified directions: in the U0
models (no unfolded state) a uniform shift of every structure's ΔΔG changes
no constant, so the fit can zero one structure's residuals with that shift
and drive its σ_s to 0, adding the shift to the other structures.  In a
simulated C4_U0 MAP (300 mutations, true σ_s = 0.3 kT, Δln K SE 0.05) one
structure's σ_s went to 0.001 and the ΔΔG error rose from 0.18 (old form)
to 0.28 kT; at true σ_s = 1 kT, and in the U1 models, which see ΔΔG_H
through K_U, σ_s came back within 20% of the truth.  A σ_s near 0 from MAP
means "not resolved", not "no noise".  SVI with the component guide
averages over ΔΔG and held σ_s near the truth in the same tests, except for
the structure that absorbs the U0 shift.

No non-centered option is offered.  For NUTS on weakly informed problems
the non-centered form mixes better; ``numpyro.infer.reparam.
LocScaleReparam(0)`` on ``{name}_ddG`` gives it without changing this
module.

Recorded sites
--------------
``{name}_ddG_sigma`` (S,) and ``{name}_ddG`` (M, S) are sample sites, and
``{name}_ddG_prior_mean`` (M, S) is a deterministic site (the CSV values for
PddG, the MLP output for PnnC).  The structure axis is in the order of the
``struct_names`` passed in.  Every PddG/PnnC model passes its
``STRUCTURE_NAMES`` order (PnnC C4/U1 requires the data in that order; the
others permute the data's columns into it), the order the simulator labels
its ``ddG_{structure}`` truth with.  :func:`guide_ddG` is the matching
component-guide block, :func:`ddG_guesses` the starting point, and
:func:`ddG_extract_specs` turns the sites into ``tfs-extract-params``
tables.
"""

import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints

# Default LogNormal prior on σ_s (kT): median exp(0) = 1 kT, 95% of the mass
# on exp(±1.96 · 1.2) = 0.095-10.5 kT.  The models' get_hyperparameters
# expose these as theta_ddG_sigma_loc / theta_ddG_sigma_scale.
DEFAULT_DDG_SIGMA_LOC = 0.0
DEFAULT_DDG_SIGMA_SCALE = 1.2

# Starting SD of the component guide's LogNormal on σ_s (log units).
_GUIDE_SIGMA_SCALE_INIT = 0.1


def ddG_sigma_site(name="theta"):
    """Name of the per-structure σ_s sample site."""
    return f"{name}_ddG_sigma"


def sample_ddG(name, struct_names, num_mut, prior_means,
               sigma_loc=DEFAULT_DDG_SIGMA_LOC,
               sigma_scale=DEFAULT_DDG_SIGMA_SCALE):
    """
    Sample per-structure σ_s and per-mutation per-structure ΔΔG (centered).

    Parameters
    ----------
    name : str
        Prefix for all Numpyro sites.
    struct_names : sequence of str, length S
        Structure identifiers (used only to determine S).
    num_mut : int
        Number of mutations M.
    prior_means : jnp.ndarray, shape (M, S)
        Structural prior means (the ΔΔG table or the MLP output).
    sigma_loc, sigma_scale : float
        LogNormal prior on σ_s, in log kT (the models' priors
        ``theta_ddG_sigma_loc`` and ``theta_ddG_sigma_scale``).

    Returns
    -------
    jnp.ndarray, shape (M, S)
        Sampled ΔΔG values.
    """
    S = len(struct_names)

    with pyro.plate(f"{name}_struct_plate", S, dim=-1):
        sigma_s = pyro.sample(ddG_sigma_site(name),
                              dist.LogNormal(sigma_loc, sigma_scale))   # (S,)

        # A wt-only library (num_mut == 0, e.g. a stratified binding pool)
        # skips the site, since numpyro.plate needs size > 0; the guides
        # skip it the same way.
        # validate_args=False lets the simulator condition σ_s to exactly 0
        # (truth = table); a fit only ever sees σ_s > 0 from the LogNormal.
        if num_mut > 0:
            with pyro.plate(f"{name}_mut_plate", num_mut, dim=-2):
                ddG = pyro.sample(f"{name}_ddG",
                                  dist.Normal(prior_means, sigma_s,
                                              validate_args=False))     # (M, S)
        else:
            ddG = jnp.zeros((0, S))

    pyro.deterministic(f"{name}_ddG_prior_mean", prior_means)
    return ddG


def guide_ddG(name, num_mut, num_struct, loc_init):
    """
    Component-guide block matching :func:`sample_ddG`.

    Mean field: σ_s ~ LogNormal(``{name}_ddG_sigma_locs``,
    ``{name}_ddG_sigma_scales``) per structure, and ΔΔG ~
    Normal(``{name}_ddG_locs``, ``{name}_ddG_scales``) per mutation and
    structure.  The names follow the ``{site}_loc(s)``/``{site}_scale(s)``
    convention that ``RunInference.component_guide_start`` maps a MAP onto.

    Parameters
    ----------
    name : str
        Site-name prefix.
    num_mut, num_struct : int
        M and S.
    loc_init : array, shape (M, S)
        Starting ΔΔG locations (the prior means at the start of the fit).

    Returns
    -------
    jnp.ndarray, shape (M, S)
        The guide's ΔΔG draw.
    """
    S = num_struct
    sigma_locs = pyro.param(f"{name}_ddG_sigma_locs", jnp.zeros(S))
    sigma_scales = pyro.param(f"{name}_ddG_sigma_scales",
                              jnp.full(S, _GUIDE_SIGMA_SCALE_INIT),
                              constraint=constraints.positive)
    with pyro.plate(f"{name}_struct_plate", S, dim=-1):
        pyro.sample(ddG_sigma_site(name),
                    dist.LogNormal(sigma_locs, sigma_scales))
        if num_mut == 0:
            return jnp.zeros((0, S))
        ddG_locs = pyro.param(f"{name}_ddG_locs",
                              jnp.asarray(loc_init, dtype=jnp.float32))
        ddG_scales = pyro.param(f"{name}_ddG_scales",
                                jnp.ones((num_mut, S)),
                                constraint=constraints.positive)
        with pyro.plate(f"{name}_mut_plate", num_mut, dim=-2):
            return pyro.sample(f"{name}_ddG",
                               dist.Normal(ddG_locs, ddG_scales))


def ddG_guesses(name, prior_means):
    """
    Starting values for the ΔΔG sites: ΔΔG at the prior means, σ_s at 1 kT.

    Parameters
    ----------
    name : str
        Site-name prefix.
    prior_means : array, shape (M, S)
        The prior means at the start of the fit (the ΔΔG table for PddG;
        zeros for PnnC, whose MLP output starts at 0).
    """
    prior_means = jnp.asarray(prior_means, dtype=jnp.float32)
    return {f"{name}_ddG": prior_means,
            ddG_sigma_site(name): jnp.ones(prior_means.shape[1])}


def ddG_extract_specs(ctx, struct_names, name="theta"):
    """
    ``get_extract_specs`` entries for the per-structure ΔΔG sites.

    The (M, S) sites ``{name}_ddG`` and ``{name}_ddG_prior_mean`` become one
    table per structure, ``ddG_{structure}`` and
    ``ddG_prior_mean_{structure}`` (``tfs-extract-params`` writes
    ``{out_prefix}_ddG_H.csv`` and so on), keyed by ``mutation``.  The names
    match the simulator's ``ddG_{structure}`` truth columns in
    ``*_sim_mutation_parameters.csv``, which ``tfs-summarize-fit`` joins on
    ``mutation``.  The (S,) site ``{name}_ddG_sigma`` becomes one table,
    ``ddG_sigma``, with one row per structure keyed by ``structure``; it
    matches the simulator's ``*_sim_ddG_sigma.csv`` truth.

    Parameters
    ----------
    ctx : ExtractionContext
        Supplies ``mut_labels``.
    struct_names : sequence of str
        The structure axis of the sites, i.e. the model's
        ``STRUCTURE_NAMES`` (the order every PddG/PnnC model samples in).
    name : str, default "theta"
        Site-name prefix.
    """
    import pandas as pd

    mut_labels = list(ctx.mut_labels)
    num_mut, S = len(mut_labels), len(struct_names)
    specs = []
    for s, struct in enumerate(struct_names):
        # Row-major flattening of (M, S): element (m, s) is m * S + s.
        mut_df = pd.DataFrame({
            "mutation": mut_labels,
            "map_ddG": [m * S + s for m in range(num_mut)],
        })
        # Separate specs, so a posterior file written before the prior-mean
        # site existed still gives the ddG tables.
        for site in ("ddG", "ddG_prior_mean"):
            specs.append(dict(
                input_df=mut_df,
                params_to_get=[site],
                out_names=[f"{site}_{struct}"],
                map_column="map_ddG",
                get_columns=["mutation"],
                in_run_prefix=f"{name}_",
            ))
    specs.append(dict(
        input_df=pd.DataFrame({"structure": list(struct_names),
                               "map_struct": range(S)}),
        params_to_get=["ddG_sigma"],
        out_names=["ddG_sigma"],
        map_column="map_struct",
        get_columns=["structure"],
        in_run_prefix=f"{name}_",
    ))
    return specs
