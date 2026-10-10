
import tfscreen

from tfscreen.simulate import (
    build_sample_dataframes,
    design_conditions,
    read_design,
    thermo_to_growth,
)
from tfscreen.simulate.sim_data_class import build_sim_data
from tfscreen.simulate.sample_theta import (
    ddG_truth_sigma_s,
    sample_theta_prior,
    sample_theta_stratified,
)
from tfscreen.simulate.binding_params import (
    SUPPORTED_COMPONENTS as _BINDING_PARAMS_SUPPORTED,
    read_binding_genotype_params,
    build_theta_gc_override_hill_geno,
    build_theta_gc_override_hill_mut,
    build_binding_theta_from_params,
    _wt_params_from_sim_priors,
)
from tfscreen.simulate.empirical.resample import (
    make_empirical_overrides,
    build_empirical_binding_theta,
)
from tfscreen.simulate.empirical.population import PopulationModel

from tfscreen.genetics import library_manager
from tfscreen.genetics import standardize_genotypes

import jax
import numpy as np
import pandas as pd

import os
import warnings
from typing import Any, Dict, Union
from pathlib import Path


def _is_file_choice(choose_by):
    """A ``choose_by`` value is a params file when it is not a builtin keyword."""
    return choose_by not in ("stratified", "random")


def _resolve_phenotype_model_path(path):
    """Resolve a ``phenotype_model`` value to the saved model JSON file.

    ``tfs-build-empirical`` writes the model as a single self-contained
    ``<out_prefix>_phenotype_model.json`` (and prints its absolute path).  This
    accepts that path with or without the ``.json`` extension, and — as a
    convenience if you point at the bare pipeline ``<out_prefix>`` — appends
    ``_phenotype_model.json``.  Returns a path that exists.
    """
    path = str(path)
    candidates = [path, f"{path}.json", f"{path}_phenotype_model.json"]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"No phenotype model file found for phenotype_model='{path}'. Looked "
        f"for: {candidates}. tfs-build-empirical prints the exact path "
        f"to use at the end of its run (a '<out_prefix>_phenotype_model.json' "
        f"file); use that -- an absolute path resolves from any working "
        f"directory, so you never need to copy the file.")


def _load_phenotype_model(cf):
    """Load the saved Stage-2 PopulationModel for ``phenotype_source: empirical``."""
    emp = cf.get("empirical")
    if not emp or not emp.get("phenotype_model"):
        raise ValueError(
            "phenotype_source: empirical requires an 'empirical' block with a "
            "'phenotype_model' path (a saved PopulationModel from "
            "tfs-build-empirical).")
    return PopulationModel.load(
        _resolve_phenotype_model_path(emp["phenotype_model"]))


def _validate_binding_config(binding_cfg, spiked_names, theta_component):
    """Fail-fast validation of the binding_data spiked_binding/library_binding blocks."""
    spiked_set = set(spiked_names)
    for block, must_be_spiked in (("spiked_binding", True),
                                  ("library_binding", False)):
        cfg = binding_cfg.get(block)
        if cfg is None:
            continue
        choose_by = cfg.get("choose_by")
        if choose_by is None:
            raise ValueError(f"binding_data.{block} requires a 'choose_by' key.")
        if _is_file_choice(choose_by):
            if "num" in cfg:
                raise ValueError(
                    f"binding_data.{block}: 'num' is incompatible with a choose_by "
                    f"file ('{choose_by}') -- the file defines the genotype set.")
            if theta_component not in _BINDING_PARAMS_SUPPORTED:
                raise ValueError(
                    f"binding_data.{block}: a choose_by file requires a Hill theta "
                    f"component {sorted(_BINDING_PARAMS_SUPPORTED)}; got "
                    f"'{theta_component}'.")
            file_genos = set(standardize_genotypes(
                list(read_binding_genotype_params(choose_by).keys())))
            if must_be_spiked:
                bad = file_genos - spiked_set
                if bad:
                    raise ValueError(
                        f"binding_data.spiked_binding genotypes must be spiked (in "
                        f"spiked_seqs); not spiked: {sorted(bad)}")
            else:
                bad = file_genos & spiked_set
                if bad:
                    raise ValueError(
                        f"binding_data.library_binding genotypes must NOT be spiked "
                        f"(they must carry congression); spiked: {sorted(bad)}")
        elif block == "spiked_binding":
            num = cfg.get("num")
            if num is not None and not (1 <= int(num) <= len(spiked_names)):
                raise ValueError(
                    f"binding_data.spiked_binding.num must be in "
                    f"[1, {len(spiked_names)}]; got {num}.")
        else:  # library_binding stratified/random
            if cfg.get("num") is None:
                raise ValueError(
                    f"binding_data.library_binding requires 'num' for choose_by "
                    f"'{choose_by}'.")


def _measured_params_override(params_file, cf, sim_data, library_df, rng,
                              binding_concs, titrant_name):
    """Build theta overrides + noise-free binding rows from a measured Hill-params file.

    Returns ``(theta_gc_override, theta_params_override, binding_rows)``.  Shared
    by spiked_binding (file) and library_binding (file); callers that measure the
    binding post-sim simply ignore ``binding_rows``.
    """
    theta_component = cf['theta_component']
    from tfscreen.tfmodel.generative.registry import model_registry
    theta_module = model_registry["theta"][theta_component]
    sim_params = theta_module.get_sim_hyperparameters()
    if cf.get('theta_sim_priors'):
        sim_params.update(cf['theta_sim_priors'])
    sim_priors = theta_module.SimPriors(**sim_params)
    wt_params = _wt_params_from_sim_priors(sim_priors)

    params_dict = read_binding_genotype_params(params_file)
    log_conc_growth = np.array(sim_data.log_titrant_conc)

    if theta_component == "hill_geno":
        gc_over, p_over = build_theta_gc_override_hill_geno(
            params_dict, log_conc_growth, wt_params)
    else:  # hill_mut
        gc_over, p_over = build_theta_gc_override_hill_mut(
            params_dict=params_dict,
            library_genotypes=library_df["genotype"].tolist(),
            sim_data=sim_data, sim_priors=sim_priors,
            log_conc=log_conc_growth, rng=rng)

    measured_binding_df = build_binding_theta_from_params(
        params_dict=params_dict, binding_concs=binding_concs,
        titrant_name=titrant_name, noise=0.0, rng=rng, wt_params=wt_params)
    return gc_over, p_over, measured_binding_df.to_dict("records")


def _spiked_keeps_own_theta(theta_component):
    """Whether spiked stratified/random binding genotypes keep their own theta.

    The thermodynamic components derive each genotype's theta from per-mutation
    equilibrium constants shared across the library, and their truth is those
    K's.  Pasting a pool curve over a spiked genotype would leave its K truth
    describing a curve that was never simulated, so for these components the
    spiked genotypes keep their own model-generated theta, and stratified or
    random only chooses which of them are measured.
    """
    return str(theta_component).startswith("thermo.")


def _select_own_theta_spiked(spiked_names, num, select_mode, binding_theta,
                             binding_concs, titrant_name, rng):
    """Choose spiked binding genotypes by their own simulated binding theta.

    ``binding_theta`` is the long-form (genotype, titrant_conc, theta) table
    from ``thermo_to_growth(..., binding_titrant_conc=...)``.  ``stratified``
    picks the ``num`` spiked genotypes whose binding curves span the most
    space (greedy maximin), ``random`` picks ``num`` at random; ``num`` None
    measures every spiked genotype.  Returns the binding rows
    (``theta_true`` straight from the table).
    """
    from tfscreen.simulate.sample_theta import _greedy_maximin

    concs = [float(c) for c in binding_concs]
    wide = (binding_theta.assign(genotype=binding_theta["genotype"].astype(str))
            .pivot(index="genotype", columns="titrant_conc", values="theta"))
    names = [g for g in standardize_genotypes(spiked_names) if g in wide.index]
    if num is None or int(num) >= len(names):
        chosen = names
    elif select_mode == "stratified":
        idx = _greedy_maximin(wide.loc[names, concs].to_numpy(dtype=float),
                              int(num))
        chosen = [names[int(i)] for i in idx]
    elif select_mode == "random":
        idx = rng.choice(len(names), size=int(num), replace=False)
        chosen = [names[int(i)] for i in idx]
    else:
        raise ValueError(f"Unknown spiked_binding choose_by: '{select_mode}'.")

    return [{"genotype": g, "titrant_name": titrant_name, "titrant_conc": c,
             "theta_true": float(wide.loc[g, c])}
            for g in chosen for c in concs]


def _spiked_stratified_binding(binding_genotypes, cf, sample_df, binding_sample_df,
                               theta_rng_key, conc_to_col, binding_concs,
                               titrant_name, select_mode):
    """Assign stratified/random synthetic theta curves to spiked binding genotypes.

    wt (if present) keeps its natural reference curve.  Returns
    ``(theta_gc_override, binding_rows)``.
    """
    binding_genotypes = list(standardize_genotypes(binding_genotypes))
    non_wt = [g for g in binding_genotypes if g != "wt"]
    override = {}
    rows = []

    wt_binding_gc = None
    if "wt" in binding_genotypes:
        single_wt_df = pd.DataFrame({"genotype": ["wt"]})
        binding_wt_sim = build_sim_data(single_wt_df, binding_sample_df,
                                        thermo_data=cf.get('thermo_data'),
                                        thermo_options=cf.get('thermo_options'),
                                        theta_component=cf['theta_component'],
                                        skip_pairs=True)
        wt_binding_gc, _ = sample_theta_prior(
            cf['theta_component'], binding_wt_sim, theta_rng_key,
            sim_priors_overrides=cf.get('theta_sim_priors'))

    selected_binding_gc = None
    if non_wt:
        selected_binding_gc, selected_growth_gc = sample_theta_stratified(
            component_name=cf['theta_component'],
            binding_sample_df=binding_sample_df,
            growth_sample_df=sample_df,
            rng_key=theta_rng_key,
            n_select=len(non_wt),
            thermo_data=cf.get('thermo_data'),
            thermo_options=cf.get('thermo_options'),
            pool_size=cf.get('binding_stratify_pool_size', 500),
            priors_overrides=cf.get('theta_priors'),
            sim_priors_overrides=cf.get('theta_sim_priors'),
            select_mode=select_mode)
        override.update({g: selected_growth_gc[i] for i, g in enumerate(non_wt)})

    if "wt" in binding_genotypes and wt_binding_gc is not None:
        for conc in binding_concs:
            rows.append({"genotype": "wt", "titrant_name": titrant_name,
                         "titrant_conc": float(conc),
                         "theta_true": float(wt_binding_gc[0, conc_to_col[float(conc)]])})
    for i, g in enumerate(non_wt):
        for conc in binding_concs:
            rows.append({"genotype": g, "titrant_name": titrant_name,
                         "titrant_conc": float(conc),
                         "theta_true": float(selected_binding_gc[i, conc_to_col[float(conc)]])})
    return override, rows


def library_prediction(cf: Union[Dict[str, Any], str, Path],
                       override_keys: dict=None,
                       return_mutation_truth: bool=False,
                       return_binding_theta: bool=False):
    """
    Predict the "ground-truth" phenotypes for a transcription factor screen
    experiment given a library, thermodynamic model, and map between that model
    and bacterial growth rate.

    Parameters
    ----------
    cf : dict or str or pathlib.Path
        The configuration for the simulation. Can be a dictionary or a path
        to a YAML file containing the configuration parameters.
    override_keys : dict, default=None
        after reading the configuration file, replace keys in the configuration
        with the key/value pairs in override keys. No error checking is done
        on these keys; the user is responsible for checking their sanity.
    return_mutation_truth : bool, default False
        Also return the per-mutation and per-pair ground truth from
        ``thermo_to_growth`` as a sixth value (``{"mutation": DataFrame or
        None, "pair": DataFrame or None}``).
    return_binding_theta : bool, default False
        Also return, last, the ground-truth theta of every library genotype
        at the ``binding_data`` concentrations (None without a
        ``binding_data`` block).

    Returns
    -------
    library_df : pandas.DataFrame
        dataframe holding genotypes one would get when using the degenerate
        codons and sequences specified in the configuration
    phenotype_df : pandas.DataFrame
        dataframe with predicted fractional occupancy and growth rates for each
        of the genotypes in each of the conditions specified in the configuration
    genotype_theta_df : pandas.DataFrame
        long-form dataframe with one row per (genotype, titrant_name,
        titrant_conc) combination; columns: genotype, titrant_name,
        titrant_conc, theta
    parameters_df : pandas.DataFrame
        one row per unique genotype; columns ``dk_geno``, ``activity``, and
        any scalar per-genotype fields from the theta component (e.g.
        ``theta_low``, ``theta_high``, ``log_hill_K``, ``hill_n``)
    binding_theta_df : pandas.DataFrame or None
        Present when ``binding_data`` is in the config.  One row per
        (genotype, titrant_conc) for the calibration genotypes listed in
        ``binding_data.genotypes``, with columns ``genotype``,
        ``titrant_name``, ``titrant_conc``, ``theta_true``.  Theta values
        here are stratified (greedy maximin) across the binding concentrations
        and are consistent with the growth phenotypes in ``phenotype_df``.
    mutation_truth : dict
        Only when ``return_mutation_truth`` is True; see
        ``thermo_to_growth``.
    library_binding_theta_df : pandas.DataFrame or None
        Only when ``return_binding_theta`` is True: columns ``genotype``,
        ``titrant_name``, ``titrant_conc``, ``theta``, evaluated from each
        genotype's drawn parameters at the binding concentrations (see
        ``thermo_to_growth._theta_at_concs``).  ``library_binding`` reads
        its theta from here.
    """

    # -------------------------------------------------------------------------
    # Read inputs and set up simulation

    cf = tfscreen.util.read_yaml(cf, override_keys=override_keys)

    # -------------------------------------------------------------------------
    # Do main calculation

    # Build library_df
    lm = library_manager.LibraryManager(cf)
    library_df = lm.build_library_df()

    # Build sample_df (holds all conditions for the experiment): from the
    # condition blocks, or from a design tube table (the union of its
    # replicates' conditions; each replicate later takes its own tubes).
    if cf.get('design') is not None:
        if cf.get('condition_blocks') is not None:
            raise ValueError("Give either 'design' or 'condition_blocks', not "
                             "both.")
        sample_df = design_conditions(read_design(cf['design']))
    else:
        sample_df = build_sample_dataframes(
            cf['condition_blocks'],
            replicate=1
        )

    # Build SimData: lightweight container for the theta model
    sim_data = build_sim_data(
        library_df=library_df,
        sample_df=sample_df,
        thermo_data=cf.get('thermo_data'),
        thermo_options=cf.get('thermo_options'),
        theta_component=cf.get('theta_component'),
    )

    # Derive both RNG objects from the single seed.  They are independent
    # objects with independent state; using the same integer for both introduces
    # no correlation between the two streams.
    seed = cf.get('seed', None)
    theta_rng_key = jax.random.PRNGKey(seed if seed is not None else 0)
    rng = np.random.default_rng(seed)

    dk_geno_zero = cf.get('dk_geno_zero', False)
    if dk_geno_zero:
        dk_geno_hyper_loc = cf.get('dk_geno_hyper_loc', -3.5)
        dk_geno_hyper_scale = cf.get('dk_geno_hyper_scale', 1.0)
        dk_geno_hyper_shift = cf.get('dk_geno_hyper_shift', 0.02)
    else:
        dk_geno_hyper_loc = cf['dk_geno_hyper_loc']
        dk_geno_hyper_scale = cf['dk_geno_hyper_scale']
        dk_geno_hyper_shift = cf['dk_geno_hyper_shift']

    # -------------------------------------------------------------------------
    # Binding-calibration genotypes: spiked (clean, congression-free) plus
    # in-library FILE injection.  Stratified/random in-library selection and all
    # in-library binding *measurements* happen post-growth-sim (see
    # simulate/library_binding_data.py); only the phenotype-setting parts live
    # here so the growth simulation reflects them.

    binding_cfg = cf.get('binding_data')
    phenotype_source = cf.get('phenotype_source', 'prior')

    # ddG_truth_sd: SD of the simulated per-structure ΔΔG around the table
    # (PddG/PnnC), set as the draw's sigma_s; checked before any sampling.
    ddG_truth_sd = cf.get('ddG_truth_sd')
    theta_model_params_override = None
    if ddG_truth_sd is not None:
        if phenotype_source == 'empirical':
            raise ValueError("ddG_truth_sd does not apply with "
                             "phenotype_source: empirical.")
        theta_model_params_override = ddG_truth_sigma_s(
            ddG_truth_sd, cf.get('theta_component'), cf.get('thermo_options'))

    theta_gc_override = {}
    theta_params_override = {}
    dk_geno_override = None
    binding_theta_df = None
    # spiked stratified/random binding whose genotypes keep their own theta
    # (thermodynamic components): chosen after the draw, below.
    deferred_spiked = None

    if phenotype_source == 'empirical':
        # Empirical phenotypes: resample every library genotype from the
        # fitted generating distribution and inject as overrides.  Binding
        # data (if configured) is regenerated from the same resampled params.
        model = _load_phenotype_model(cf)
        log_conc_growth = np.array(sim_data.log_titrant_conc)
        pheno_df, theta_gc_override, theta_params_override, dk_geno_override = \
            make_empirical_overrides(
                model, list(library_df["genotype"]), log_conc_growth,
                rng=rng, wt_ref=model.wt_ref)
        if binding_cfg is not None:
            spiked_names = list(pd.unique(library_df.loc[
                library_df["library_origin"] == "spiked", "genotype"]))
            binding_theta_df = build_empirical_binding_theta(
                pheno_df, binding_cfg, spiked_names, rng)

    elif binding_cfg is not None:
        binding_concs = binding_cfg['titrant_conc']
        titrant_name = binding_cfg['titrant_name']
        theta_component = cf['theta_component']

        binding_sample_df = pd.DataFrame({"titrant_conc": binding_concs})
        sorted_binding_concs = np.sort(np.unique(binding_concs))
        conc_to_col = {float(c): j for j, c in enumerate(sorted_binding_concs)}
        rows = []

        spiked_names = list(pd.unique(
            library_df.loc[library_df["library_origin"] == "spiked", "genotype"]))

        _validate_binding_config(binding_cfg, spiked_names, theta_component)

        # ---------- spiked_binding (clean, monoclonal) ----------
        sb = binding_cfg.get("spiked_binding")
        if sb is not None:
            choose_by = sb["choose_by"]
            if _is_file_choice(choose_by):
                gc_over, p_over, spiked_rows = _measured_params_override(
                    choose_by, cf, sim_data, library_df, rng,
                    binding_concs, titrant_name)
                theta_gc_override.update(gc_over)
                theta_params_override.update(p_over)
                rows.extend(spiked_rows)
            elif _spiked_keeps_own_theta(theta_component):
                deferred_spiked = (sb.get("num"), choose_by)
            else:
                num = sb.get("num")
                sel = spiked_names if num is None else spiked_names[:int(num)]
                s_over, s_rows = _spiked_stratified_binding(
                    sel, cf, sample_df, binding_sample_df, theta_rng_key,
                    conc_to_col, binding_concs, titrant_name,
                    select_mode=choose_by)
                theta_gc_override.update(s_over)
                rows.extend(s_rows)

        # ---------- library_binding: FILE injection only (pre-sim) ----------
        lb = binding_cfg.get("library_binding")
        if lb is not None and _is_file_choice(lb["choose_by"]):
            gc_over, p_over, _ = _measured_params_override(
                lb["choose_by"], cf, sim_data, library_df, rng,
                binding_concs, titrant_name)
            theta_gc_override.update(gc_over)
            theta_params_override.update(p_over)

        binding_theta_df = pd.DataFrame(rows) if rows else None

    # -------------------------------------------------------------------------
    # Calculate phenotype for each genotype across all conditions in sample_df

    # In empirical mode the resampled phenotypes fully replace the prior draw,
    # and are hill_geno-structured (marginal per-genotype).  So force hill_geno
    # for the (discarded) baseline draw — this both matches the parameters_df
    # schema the overrides patch and avoids running/overflowing an expensive
    # e.g. hill_mut prior draw that would be thrown away — drop the (ignored)
    # theta prior overrides, and force unit activity (A absorbed into theta).
    if phenotype_source == 'empirical':
        configured = cf.get('theta_component', 'hill_geno')
        if configured != 'hill_geno':
            warnings.warn(
                f"phenotype_source: empirical forces theta_component=hill_geno "
                f"(config has '{configured}', which is ignored): the resampled "
                f"phenotypes are per-genotype Hill curves.")
        theta_component = 'hill_geno'
        theta_priors_overrides = None
        theta_sim_priors_overrides = None
        activity_component = 'fixed'
        activity_wt = 1.0
        activity_mut_scale = 0.0
    else:
        theta_component = cf['theta_component']
        theta_priors_overrides = cf.get('theta_priors')
        theta_sim_priors_overrides = cf.get('theta_sim_priors')
        activity_component = cf.get('activity_component', 'fixed')
        activity_wt = cf.get('activity_wt', 1.0)
        activity_mut_scale = cf.get('activity_mut_scale', 0.0)

    need_binding_theta = binding_cfg is not None and (
        return_binding_theta or deferred_spiked is not None)

    result = thermo_to_growth(
        genotypes=library_df["genotype"],
        sim_data=sim_data,
        sample_df=sample_df,
        theta_component=theta_component,
        theta_rng_key=theta_rng_key,
        growth_params=cf['growth'],
        theta_priors_overrides=theta_priors_overrides,
        theta_sim_priors_overrides=theta_sim_priors_overrides,
        dk_geno_hyper_loc=dk_geno_hyper_loc,
        dk_geno_hyper_scale=dk_geno_hyper_scale,
        dk_geno_hyper_shift=dk_geno_hyper_shift,
        dk_geno_zero=dk_geno_zero,
        activity_wt=activity_wt,
        activity_mut_scale=activity_mut_scale,
        rng=rng,
        activity_component=activity_component,
        activity_priors_overrides=cf.get('activity_priors'),
        theta_rescale=cf.get('theta_rescale', 'passthrough'),
        theta_gc_override=theta_gc_override,
        theta_params_override=theta_params_override or None,
        dk_geno_override=dk_geno_override,
        return_mutation_truth=return_mutation_truth,
        theta_model_params_override=theta_model_params_override,
        **({"binding_titrant_conc": binding_cfg['titrant_conc']}
           if need_binding_theta else {}),
    )
    phenotype_df, genotype_theta_df, parameters_df = result[:3]

    library_binding_theta_df = None
    if need_binding_theta:
        library_binding_theta_df = result[-1].copy()
        library_binding_theta_df.insert(1, "titrant_name",
                                        binding_cfg['titrant_name'])

    # Spiked genotypes that keep their own theta: choose which are measured
    # from their simulated binding curves, so their binding theta, growth
    # theta and K truth all come from the same draw.
    if deferred_spiked is not None:
        num, choose_by = deferred_spiked
        spiked_rows = _select_own_theta_spiked(
            spiked_names, num, choose_by, library_binding_theta_df,
            binding_cfg['titrant_conc'], binding_cfg['titrant_name'], rng)
        prior = [] if binding_theta_df is None else \
            binding_theta_df.to_dict("records")
        binding_theta_df = pd.DataFrame(spiked_rows + prior)

    out = (library_df, phenotype_df, genotype_theta_df, parameters_df,
           binding_theta_df)
    if return_mutation_truth:
        out += (result[3],)
    if return_binding_theta:
        out += (library_binding_theta_df,)
    return out
