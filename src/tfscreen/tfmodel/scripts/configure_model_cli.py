from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
from tfscreen.tfmodel.configuration_io import write_configuration
from tfscreen.tfmodel.priors_edit import (
    GROWTH_PRIOR_FIELDS,
    GROWTH_PRIOR_PREFIX,
    apply_prior_overrides,
    apply_priors_updates,
    condition_rep_labels,
    growth_prior_updates,
    growth_priors_from_wt_rates,
    parse_prior_overrides,
)
from tfscreen.tfmodel.model_stats import (
    count_model_dimensions,
    format_model_stats,
    write_model_stats,
)
from tfscreen.util.cli.generalized_main import generalized_main
from tfscreen.util import read_dataframe, read_yaml
from tfscreen.genetics import (
    RESERVED_GENOTYPES,
    library_composition_table,
    standardize_genotypes,
    write_library_composition,
)

import numpy as np
import pandas as pd

# Maps (condition_growth, theta_rescale) pairs that are fundamentally incompatible
# to a human-readable explanation of why.
#
# power uses theta**n (non-integer exponent): negative theta from logit → NaN.
# saturation uses theta/(1+theta): logit can produce theta near -1 → pole/NaN.
INCOMPATIBLE_CG_TR = {
    ("power", "logit"): (
        "The 'power' growth model raises theta to a non-integer exponent (theta**n). "
        "The 'logit' rescale maps theta to all of R, producing negative values that "
        "cause NaN when raised to a non-integer power."
    ),
    ("saturation", "logit"): (
        "The 'saturation' growth model evaluates theta/(1+theta). "
        "The 'logit' rescale can map theta to values near -1, causing a "
        "division-by-zero singularity."
    ),
}


def check_component_compatibility(condition_growth_model, theta_rescale_model):
    """Raise ValueError if the (condition_growth, theta_rescale) pair is incompatible.

    Parameters
    ----------
    condition_growth_model : str
        Name of the condition_growth component.
    theta_rescale_model : str
        Name of the theta_rescale component.

    Raises
    ------
    ValueError
        If the pair is listed in INCOMPATIBLE_CG_TR.
    """
    key = (condition_growth_model, theta_rescale_model)
    if key in INCOMPATIBLE_CG_TR:
        raise ValueError(
            f"Incompatible model components: "
            f"condition_growth='{condition_growth_model}' and "
            f"theta_rescale='{theta_rescale_model}'. "
            f"{INCOMPATIBLE_CG_TR[key]}"
        )


def check_genotypes_in_library(library_genotypes, data_df, label,
                               max_report=20):
    """
    Fail unless every genotype in a data file is part of the library.

    The library YAML handed to ``tfs-configure-model`` must be the same one
    handed to ``tfs-process-fastq``.  Nothing cross-checks the two, but a
    mismatch in the *genetics* keys (residue numbering, wt sequence, tiles,
    spiked sequences) shows up here as genotypes the library does not contain.
    A one-off numbering shift, for instance, mismatches essentially every
    mutant.

    Parameters
    ----------
    library_genotypes : set of str
        Standardized genotype names in the library.
    data_df : str or pandas.DataFrame
        Data file (or frame) with a ``genotype`` column.
    label : str
        Name of the argument being checked, used in the error message.
    max_report : int, optional
        Maximum number of offending genotypes to name in the error.

    Raises
    ------
    ValueError
        If any genotype is absent from the library.  Reserved sentinels
        (``__unknown__``) are ignored.
    """

    if data_df is None:
        return

    df = read_dataframe(data_df)
    if "genotype" not in df.columns:
        raise ValueError(f"{label} has no 'genotype' column.")

    observed = standardize_genotypes(df["genotype"])
    observed = pd.unique(np.asarray(observed))
    observed = [g for g in observed if g not in RESERVED_GENOTYPES]

    missing = sorted(set(observed) - set(library_genotypes))
    if len(missing) == 0:
        return

    shown = missing[:max_report]
    suffix = "" if len(missing) <= max_report else f" (and {len(missing) - max_report} more)"
    source = data_df if isinstance(data_df, str) else "<dataframe>"
    raise ValueError(
        f"{len(missing)} of {len(observed)} genotypes in {label} "
        f"('{source}') are not in the library described by library_config: "
        f"{shown}{suffix}.\n"
        f"The library config must be the same one used to call the "
        f"sequencing reads (tfs-process-fastq).  A residue-numbering or "
        f"wt-sequence difference between the two will mismatch most or all "
        f"genotypes; a handful of mismatches usually means the data include "
        f"genotypes the library design does not contain."
    )


def check_spikes_in_data(composition, growth_df, allow_missing=False):
    """
    Fail unless every spiked genotype of the library has growth data.

    A spike named in the library config but absent from the counts usually
    means the reads were called with a different library config (a spike
    list or codon that disagrees with the real spikes).

    Parameters
    ----------
    composition : pandas.DataFrame
        ``library_composition_table`` output (``genotype``, ``in_spiked_origin``).
    growth_df : str or pandas.DataFrame
        Growth data with a ``genotype`` column.
    allow_missing : bool
        Print the missing spikes instead of raising.
    """
    spikes = set(composition.loc[composition["in_spiked_origin"].astype(bool),
                                 "genotype"])
    if not spikes or growth_df is None:
        return
    df = read_dataframe(growth_df)
    observed = set(pd.unique(np.asarray(standardize_genotypes(df["genotype"]))))
    missing = sorted(spikes - observed)
    if not missing:
        return
    msg = (f"{len(missing)} of {len(spikes)} spiked genotypes in "
           f"library_config have no growth data: {missing}. Check that the "
           f"counts were called with this library config "
           f"(tfs-process-fastq).")
    if not allow_missing:
        raise ValueError(msg + " Pass --allow_missing_spikes if these spikes "
                         "really dropped out.")
    print(f"warning: {msg}", flush=True)


def _edit_priors(orchestrator, priors_path, set_priors=None, growth_priors=None,
                 growth_priors_wt_rates=None, growth_priors_sd_floor=0.002,
                 condition_growth_model="linear", theta_model=None):
    """Apply --growth_priors, --growth_priors_wt_rates and --set_priors."""
    if growth_priors is not None or growth_priors_wt_rates is not None:
        if orchestrator.growth_tm is None:
            raise ValueError("growth_priors need growth data.")
        if condition_growth_model != "linear":
            raise ValueError("growth_priors and growth_priors_wt_rates support "
                             "condition_growth_model 'linear' only.")
        table = None
        if growth_priors is not None:
            table = read_dataframe(growth_priors)
        if growth_priors_wt_rates is not None:
            if theta_model != "hill_relative":
                raise ValueError("growth_priors_wt_rates sets k and m on the "
                                 "relative X gauge; it needs theta_model "
                                 "'hill_relative'.")
            gauge = orchestrator.settings["theta_gauge_conc"]
            wt = growth_priors_from_wt_rates(read_dataframe(growth_priors_wt_rates),
                                             gauge, sd_floor=growth_priors_sd_floor)
            if table is None:
                table = wt
            else:
                if "replicate" in table.columns:
                    raise ValueError("growth_priors_wt_rates cannot be combined "
                                     "with a per-replicate growth_priors table.")
                table = table.set_index("condition_rep")
                wt = wt.set_index("condition_rep")
                table = wt.combine_first(table).reset_index()

        priors_df = pd.read_csv(priors_path)
        scalar = priors_df[priors_df.get("flat_index", pd.Series(np.nan, index=priors_df.index)).isna()]
        defaults = {}
        for fields in GROWTH_PRIOR_FIELDS.values():
            for f in fields:
                row = scalar[scalar["parameter"] == GROWTH_PRIOR_PREFIX + f]
                if row.empty:
                    raise ValueError(f"Priors CSV has no '{GROWTH_PRIOR_PREFIX + f}' row.")
                defaults[f] = float(row["value"].iloc[0])

        labels = condition_rep_labels(orchestrator)
        updates = growth_prior_updates(table, labels, defaults)
        apply_priors_updates(priors_path, updates, cond_rep_labels=labels,
                             backup=False)

    if set_priors:
        resolved = apply_prior_overrides(priors_path,
                                         parse_prior_overrides(set_priors))
        for name, value in resolved.items():
            print(f"  prior {name} = {value:g}", flush=True)


def configure_model(binding_df=None,
                    growth_df=None,
                    presplit_df=None,
                    base_growth_df=None,
                    out_prefix="tfs_configure",
                    condition_growth_model="linear",
                    growth_transition_model="instant",
                    ln_cfu0_model="hierarchical",
                    dk_geno_model="hierarchical_geno",
                    activity_model="fixed",
                    theta_model="hill_geno",
                    transformation_model="single",
                    transformation_lambda=None,
                    congression_theta_rule="homodimer",
                    congression_dk_rule="dilution",
                    congression_dk_alpha=None,
                    theta_rescale_model="passthrough",
                    theta_growth_noise_model="zero",
                    theta_binding_noise_model="zero",
                    growth_noise_model="zero",
                    sample_offset_model="level",
                    growth_likelihood="counts",
                    theta_gauge_conc=None,
                    library_config=None,
                    growth_shares_replicates=False,
                    epistasis=False,
                    thermo_data=None,
                    batch_size=1024,
                    binding_weight=None,
                    set_priors=None,
                    growth_priors=None,
                    growth_priors_wt_rates=None,
                    growth_priors_sd_floor=0.002,
                    allow_missing_spikes=False,
                    skip_model_stats=False):
    """
    Build and write the YAML configuration files needed by tfs-fit-model.

    Constructs a ModelOrchestrator from the supplied data and model-component choices,
    then writes {out_prefix}_config.yaml (the main configuration),
    {out_prefix}_priors.csv (prior distributions for all parameters),
    {out_prefix}_guesses.csv (initial-value guesses for array parameters) and,
    for a growth model, {out_prefix}_library.csv (the per-genotype library
    composition resolved from library_config).

    At least one of growth_df and binding_df is required. With both, the
    joint growth + binding model is configured. With only binding_df, a
    binding-only model infers theta directly from the binding measurements.
    With only growth_df, a growth-only model infers theta from growth alone
    (no binding likelihood, so theta_binding_noise_model must stay 'zero'
    and binding_weight unset; tfs-prefit-calibration needs binding and
    refuses such a config).

    Parameters
    ----------
    binding_df : str, optional
        Path to the binding data CSV file (theta vs. titrant measurements per
        genotype). When omitted, a growth-only model is configured.
    growth_df : str, optional
        Path to the growth data CSV file (ln_cfu measurements per genotype,
        replicate, and timepoint). When omitted, a binding-only model is
        configured.
    presplit_df : str, optional
        Path to the pre-split (t = -t_pre) sequencing-observation CSV file.
        Provides a direct constraint on ln_cfu0 for genotypes it covers. See
        data_class.PreSplitData.
    base_growth_df : str, optional
        Path to a CSV of direct, reference-condition growth-rate
        measurements (columns: genotype, rate, rate_std) for a subset of
        genotypes (wt at minimum). Anchors the new k_ref latent scalar to
        dk_geno via ``rate_obs ~ Normal(k_ref + dk_geno, rate_std)``,
        resolving an identifiability confound between condition_growth's
        k/m and dk_geno's hierarchical hyperparameters. See
        model_orchestrator._read_base_growth_df and generative/model.py's
        base_growth_obs block.
    out_prefix : str, optional
        Prefix for the output files ({out_prefix}_config.yaml,
        {out_prefix}_priors.csv, {out_prefix}_guesses.csv and, for a growth
        model, {out_prefix}_library.csv). Default 'tfs_configure'.
    condition_growth_model: str, optional
        Model to use to describe growth under different conditions (e.g.,
        pheS+4CP). Allowed values are 'linear' (default), 'power', or
        'saturation'.
    growth_transition_model : str, optional
        Model to use to describe the transition between the pre-selection
        and selection phases. Allowed values are 'instant' (default), 'memory',
        'baranyi', 'baranyi_k', 'baranyi_tau', or 'two_pop'.
    ln_cfu0_model : str, optional
        Model to use to describe ln_cfu0, the initial populations of genotypes
        in each replicate. Allowed values are 'hierarchical' (default) or
        'hierarchical_factored'. The latter shares each genotype's baseline
        across pre-conditions and is refused when they come from different
        libraries (kanR and pheS are grown up separately).
    dk_geno_model : str, optional
        Model to use to describe dk_geno, the pleiotropic effect of a genotype
        on growth, independent of occupancy. Allowed values are
        'hierarchical_geno' (default), 'fixed', or 'pinned' (dk_geno fixed to
        externally supplied per-genotype values for a subset of genotypes).
    activity_model : str, optional
        Model to use to describe activity, a scalar multiplied against
        occupancy that defines how strongly a genotype alters transcription
        given its occupancy. Allowed values are 'fixed' (default; activity 1
        for every genotype), 'hierarchical_geno', 'horseshoe_geno',
        'hierarchical_mut', or 'horseshoe_mut'.
    theta_model : str, optional
        Model to use to describe theta, the fractional occupancy of a genotype
        on the transcription factor binding site. Allowed values are
        'hill_geno' (default), 'categorical_geno', 'hill_mut', and the
        thermodynamic partition-function models (pass the exact registry key,
        e.g. 'thermo.O2_C4_K3_U0_a.PK', 'thermo.O2_C4_K3_U0_a.PnnC',
        'thermo.O2_C4_K3_U0_a.PddG', 'thermo.O2_C12_K5_U0_a.PK',
        'thermo.O2_C12_K5_U0_a.PnnC', 'thermo.O2_C12_K5_U0_a.PddG', and
        their O2_C4_K3_U1_a / O2_C12_K5_U1_a unfolded equivalents).
        'hill_relative' fits growth alone on a wt-relative scale X instead
        of theta (wt's X is 1 at the low and 0 at the high gauge
        concentration, theta_gauge_conc). It takes no binding_df, and
        requires activity 'fixed', theta_rescale 'passthrough', linear
        condition growth, theta_growth_noise 'zero' and, with the
        'mixture' transformation, congression_theta_rule 'max'.
    transformation_model : str, optional
        Model for congression. Allowed values are 'single' (default; one
        plasmid per cell) or 'mixture' (clean and congressed cells mixed at
        the observable level). 'empirical' and 'logit_norm' were removed.
    transformation_lambda : list or tuple, optional
        ``(mean, std)`` -- the experimentally measured congression lambda,
        in linear space (e.g. ``(0.36, 0.05)``). Required when
        ``transformation_model`` is 'mixture'; forbidden
        when it is 'single'. Used to moment-match a LogNormal prior for the
        transformation's lambda parameter, replacing the manual step of
        hand-editing the priors/guesses CSVs with rescaled log-space values.
    congression_theta_rule : str, optional
        How the 'mixture' transformation builds a congressed cell's theta
        from its plasmids' thetas: 'homodimer' (default), 'heterodimer' or
        'max'. The homodimer and heterodimer rules require activity_model
        'fixed'. Ignored by 'single'.
    congression_dk_rule : str, optional
        How the 'mixture' transformation builds a congressed cell's dk_geno
        from its plasmids': 'dilution' (default; the share-weighted mean),
        'softmin' (with congression_dk_alpha) or 'min' (the worst variant
        sets the cost). Ignored by 'single'.
    congression_dk_alpha : float, optional
        alpha for congression_dk_rule 'softmin' (> 0, in units of 1/dk):
        ``dk_cell = -(1/alpha) log sum_g x_g exp(-alpha dk_g)``. Required by
        'softmin', refused by the other rules.
    theta_rescale_model : str, optional
        Rescaling applied to theta before it enters the growth model. Allowed
        values are 'passthrough' (default, identity) or 'logit' (maps theta to
        log(theta/(1-theta)), expanding the dynamic range at both extremes).
    theta_growth_noise_model : str, optional
        Model to use for stochastic experimental noise in theta measured by
        bacterial growth. Allowed values are 'zero' (default), 'beta', or
        'logit_normal'.
    theta_binding_noise_model : str, optional
        Model to use for stochastic experimental noise in theta measured by
        binding. Allowed values are 'zero' (default) or 'beta'.
    growth_noise_model : str, optional
        Model for additive growth-rate noise. 'zero' (default) adds no noise;
        'normal_kt' learns a global sigma_k that inflates the observation scale
        in quadrature with ln_cfu_std, capturing biological variability in
        growth rates not explained by theta or dk_geno.
    sample_offset_model : str, optional
        Per-tube offset shared by every genotype in a tube. 'level'
        (default) adds one ln_cfu offset per tube with a learned, constant
        SD (the tube's composition offset and any error in its supplied
        total, including the tube's own growth noise); 'zero' adds none;
        'normal' adds a per-tube growth-rate offset scaled by elapsed time.
        With growth_likelihood 'counts', an error in a tube's supplied total
        otherwise moves every genotype in it.
    growth_likelihood : str, optional
        How growth data are observed: 'counts' (default; negative binomial
        on the read counts, with a learned dispersion whose variance has a
        part proportional to the mean and a quadratic part; no pseudocount)
        or 'lncfu' (Student-t on ln_cfu). 'counts' needs a 'counts' column,
        each tube's total reads ('sample_reads', or 'adjusted_counts' and
        'frequency' to derive it) and total cells ('sample_ln_cfu', or
        'sample_cfu' with its uncertainty), and growth_noise_model 'zero'.
        It cut theta RMSE by 35-60% against 'lncfu' on simulations
        (planning/studies/count-likelihood/).
    theta_gauge_conc : list of float, optional
        ``c_lo c_hi``: the titrant concentrations at which theta_model
        'hill_relative' pins wt's X to 1 and 0. Default: the lowest and
        highest concentration in growth_df. Refused by other theta models.
    library_config : str, optional
        Path to the library YAML describing the screened library -- the same
        file handed to ``tfs-process-fastq``.  Required whenever ``growth_df``
        is given (a binding-only model has no congression correction and no
        ln_cfu0 latents, so it does not need one).  Read keys:
        ``reading_frame``, ``first_amplicon_residue``, ``wt_seq``,
        ``degen_sites``, ``tiles``, ``tile_combos``, ``spiked_seqs`` and
        ``library_mixture``; any other keys (a full simulate config, say) are
        ignored.  Genotypes encoded by a spiked sequence become the spiked
        set, replacing the old hand-supplied ``--spiked`` list.  The resolved
        per-genotype table is written to {out_prefix}_library.csv and is what
        ``tfs-fit-model`` reads.
    growth_shares_replicates : bool, optional
        Whether replicates should share the same parameters for the growth and
        growth transition models. Default is False.
    epistasis : bool, optional
        Whether to model pairwise epistatic interactions between mutations in
        any mutation-level model (``hill_mut``, ``hierarchical_mut``). When
        True, each pair of mutations present in the same genotype gets an
        independent epistasis term. When False (default), effects are purely
        additive at the mutation level.
    thermo_data : str, optional
        Path to the structural/thermodynamic data file.  Required when
        ``theta_model`` is a thermo-based model; ignored otherwise.  For
        ``PnnC`` models this must be the HDF5 file produced by
        ``scripts/generate_struct_ensemble.py``.  For ``PddG`` models this
        must be a CSV file with a ``mut`` column and one column per
        structure containing pre-computed ΔΔG prior means; the required
        structure columns depend on the topology -- ``O2_C4_*`` models use
        (``H``, ``HD``, ``L``, ``LE2``) while ``O2_C12_*`` models use
        (``H``, ``HO``, ``L``, ``LO``, ``HE2``, ``LE2``).
    batch_size : int, optional
        Mini-batch size for SVI. Defaults to 1024. Set to None to use the full
        dataset as a single batch.
    binding_weight : float, optional
        Multiplicative scale applied to the binding log-likelihood at every SVI
        step.  Because growth data typically outnumber binding observations by
        several orders of magnitude, the binding likelihood contributes a
        negligible gradient signal unless it is upweighted.  When None
        (default), the weight is auto-computed as
        ``N_growth_rows / N_binding_rows`` so that each binding observation
        contributes the same total weight as the average growth observation.
        Pass an explicit positive float to override this heuristic.  The
        resolved value (never None) is saved in the YAML so that
        ``tfs-fit-model`` applies the same weight without recomputing it.
    set_priors : list of str, optional
        ``name=value`` pairs that set scalar priors in
        {out_prefix}_priors.csv, so the file is never edited by hand. A name
        is a full row name (``theta.theta_log_hill_n_hyper_scale_fixed``) or
        a unique dotted suffix of one (``theta_log_hill_n_hyper_scale_fixed``,
        ``sample_offset.sigma_fixed``). An unknown or ambiguous name is an
        error. Common uses: ``sigma_fixed=0.17`` holds the level tube
        offsets' SD; ``theta_log_hill_n_hyper_scale_fixed=0.5`` holds the
        population SD of log(hill_n) (and, for hill_relative,
        ``theta_X_low_hyper_scale_fixed``, ``theta_X_delta_hyper_scale_fixed``,
        ``theta_log_hill_K_hyper_scale_fixed``).
    growth_priors : str, optional
        CSV of per-condition growth priors for linear condition growth: a
        ``condition_rep`` column (and ``replicate``, if conditions are not
        shared across replicates) plus any of ``k_loc``, ``k_scale``,
        ``m_loc``, ``m_scale``. Conditions or values it leaves out keep the
        defaults. Use it where tfs-prefit-calibration does not run (a
        growth-only model); the pre-fit overwrites k_loc and m_loc.
    growth_priors_wt_rates : str, optional
        CSV of wt monoculture growth rates (``condition_sel``,
        ``titrant_conc``, ``rate_mean``, ``rate_sd``, ``num_replicates``),
        for theta_model 'hill_relative' only. For each listed condition,
        k = the rate at the high gauge concentration and m = the rate at the
        low one minus k, each with the standard error of the mean as SD
        (floored at growth_priors_sd_floor). List only the conditions where
        the monoculture stands for the library's wt. Overrides
        growth_priors for the conditions it lists.
    growth_priors_sd_floor : float, optional
        Smallest k and m prior SD (per minute) derived from
        growth_priors_wt_rates. Default 0.002.
    allow_missing_spikes : bool, optional
        By default, a spiked genotype named in library_config with no growth
        data is an error: it usually means the counts were called with a
        different library config. Pass this to only report it (a spike can
        drop out of a real experiment).
    skip_model_stats : bool, optional
        Skip the pre-fit parameter/observation accounting. By default a
        summary is printed to stdout and written to
        {out_prefix}_model_stats.csv (one row per latent) and
        {out_prefix}_model_stats.json (headline counts, coverage, anchors,
        warnings). See tfmodel.model_stats. The accounting traces the model
        abstractly, so it costs no meaningful time or memory; skip it only if
        a component's trace misbehaves.

    Returns
    -------
    None
    """
    if binding_df is None and growth_df is None:
        raise ValueError("At least one of binding_df and growth_df must be "
                         "provided.")

    binding_only = growth_df is None
    if not binding_only:
        check_component_compatibility(condition_growth_model, theta_rescale_model)

    if transformation_model != "single" and transformation_lambda is None:
        raise ValueError(
            f"transformation_model='{transformation_model}' requires "
            f"transformation_lambda (mean, std) -- the experimentally measured "
            f"congression lambda in linear space, e.g. "
            f"transformation_lambda=(0.36, 0.05)."
        )

    # Resolve the library description.  The composition table is written
    # before the orchestrator is built, because the orchestrator reads the
    # snapshot (not the YAML) -- the same file tfs-fit-model will read.
    library_file = None
    library_meta = None
    if binding_only:
        if library_config is not None:
            raise ValueError(
                "library_config was supplied but no growth_df.  A "
                "binding-only model has no congression correction and no "
                "ln_cfu0 latents, so it has no use for the library."
            )
    else:
        if library_config is None:
            raise ValueError(
                "library_config is required when growth_df is given.  Pass "
                "the library YAML that describes the screened library -- the "
                "same file used for tfs-process-fastq."
            )

        composition = library_composition_table(library_config)

        for data_df, label in ((growth_df, "growth_df"),
                               (presplit_df, "presplit_df"),
                               (base_growth_df, "base_growth_df")):
            check_genotypes_in_library(set(composition["genotype"]),
                                       data_df, label)

        check_spikes_in_data(composition, growth_df,
                             allow_missing=allow_missing_spikes)

        library_file = f"{out_prefix}_library.csv"
        write_library_composition(composition, library_file)
        print(f"Wrote library composition to {library_file}", flush=True)

        library_meta = {
            "source": library_config if isinstance(library_config, str) else "<dict>",
            "library_mixture": read_yaml(library_config)["library_mixture"],
        }

    # Initialize model to build mappings and get guesses
    orchestrator = ModelOrchestrator(growth_df,
                     binding_df,
                     presplit_df=presplit_df,
                     base_growth_df=base_growth_df,
                     binding_only=binding_only,
                     condition_growth=condition_growth_model,
                     growth_transition=growth_transition_model,
                     ln_cfu0=ln_cfu0_model,
                     dk_geno=dk_geno_model,
                     activity=activity_model,
                     theta=theta_model,
                     transformation=transformation_model,
                     transformation_lambda=transformation_lambda,
                     congression_theta_rule=congression_theta_rule,
                     congression_dk_rule=congression_dk_rule,
                     congression_dk_alpha=congression_dk_alpha,
                     theta_rescale=theta_rescale_model,
                     theta_growth_noise=theta_growth_noise_model,
                     theta_binding_noise=theta_binding_noise_model,
                     growth_noise=growth_noise_model,
                     sample_offset=sample_offset_model,
                     growth_likelihood=growth_likelihood,
                     theta_gauge_conc=theta_gauge_conc,
                     library_file=library_file,
                     growth_shares_replicates=growth_shares_replicates,
                     epistasis=epistasis,
                     thermo_data=thermo_data,
                     batch_size=batch_size,
                     binding_weight=binding_weight)

    # Write the model configuration to a file. This includes the model component
    # names, the data file paths, and the parameter guesses/priors.
    growth_path = None if binding_only else (growth_df if isinstance(growth_df, str) else "growth.csv")
    presplit_path = presplit_df if isinstance(presplit_df, str) else None
    base_growth_path = base_growth_df if isinstance(base_growth_df, str) else None
    write_configuration(orchestrator=orchestrator,
                        out_prefix=out_prefix,
                        growth_df_path=growth_path,
                        binding_df_path=(None if binding_df is None else
                                         binding_df if isinstance(binding_df, str)
                                         else "binding.csv"),
                        presplit_df_path=presplit_path,
                        base_growth_df_path=base_growth_path,
                        library_meta=library_meta)

    _edit_priors(orchestrator, f"{out_prefix}_priors.csv",
                 set_priors=set_priors,
                 growth_priors=growth_priors,
                 growth_priors_wt_rates=growth_priors_wt_rates,
                 growth_priors_sd_floor=growth_priors_sd_floor,
                 condition_growth_model=condition_growth_model,
                 theta_model=theta_model)

    # Pre-fit accounting: how many parameters, how many observations, and
    # which genotypes are individually under-determined.
    if not skip_model_stats:
        stats = count_model_dimensions(orchestrator)
        print(format_model_stats(stats), flush=True)
        write_model_stats(stats, out_prefix)

def main():
    return generalized_main(configure_model,
                            manual_arg_types={"binding_df":str,
                                              "congression_dk_alpha":float,
                                              "growth_df":str,
                                              "presplit_df":str,
                                              "base_growth_df":str,
                                              "library_config":str,
                                              "thermo_data":str,
                                              "batch_size":int,
                                              "binding_weight":float,
                                              "set_priors":str,
                                              "transformation_lambda":float,
                                              "theta_gauge_conc":float},
                            manual_arg_nargs={"set_priors":"+",
                                              "transformation_lambda":2,
                                              "theta_gauge_conc":2})

if __name__ == "__main__":
    main()
