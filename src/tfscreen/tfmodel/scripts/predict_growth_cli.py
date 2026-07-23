import numpy as np
import pandas as pd
from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.inference.checkpoint_io import resolve_param_file
from tfscreen.tfmodel.inference.posteriors import load_posteriors
from tfscreen.tfmodel.analysis.prediction import predict
from tfscreen.tfmodel.analysis.batch_sizing import estimate_genotype_batch_size
from tfscreen.util.cli import generalized_main, read_lines


def _is_oom_error(exc):
    """True if an exception looks like a GPU/TPU out-of-memory error.

    JAX surfaces device OOM as a ``JaxRuntimeError`` whose message contains
    ``RESOURCE_EXHAUSTED``; match on the message so we don't need to import the
    specific error type (which varies across JAX versions).
    """
    msg = str(exc)
    return ("RESOURCE_EXHAUSTED" in msg
            or "Out of memory" in msg
            or "out of memory" in msg)


def _predict_subset_with_backoff(predict_kwargs, keep, extra):
    """Run predict() on ``keep + extra`` genotypes, halving ``extra`` on OOM.

    The mandatory ``keep`` genotypes (binding/spiked/file) are always retained;
    only the randomly-sampled ``extra`` genotypes are dropped when the device
    runs out of memory. This makes subset mode robust to an over-optimistic
    genotype_batch_size estimate: the memory-fit block is a sample already, so
    shrinking it on OOM still yields a valid input/output correlation check.
    Re-raises any non-OOM error, or an OOM that persists once ``extra`` is
    empty (the mandatory anchors alone don't fit).

    Parameters
    ----------
    predict_kwargs : dict
        Keyword arguments forwarded to predict() (without ``genotypes``).
    keep : list of str
        Mandatory genotypes, always predicted.
    extra : list of str
        Optional genotypes, shrunk by half on each OOM retry.

    Returns
    -------
    pandas.DataFrame
        The predict() result for the largest block that fit.
    """
    keep = list(keep)
    extra = list(extra)
    while True:
        genotypes = keep + extra
        try:
            return predict(**predict_kwargs, genotypes=genotypes)
        except Exception as exc:
            if not _is_oom_error(exc) or len(extra) == 0:
                raise
            new_n = len(extra) // 2
            print(f"  GPU out of memory at {len(genotypes)} genotypes; "
                  f"retrying with {len(keep) + new_n} "
                  f"({new_n} sampled + {len(keep)} mandatory)...", flush=True)
            extra = extra[:new_n]


def predict_growth(config_file,
                   param_file,
                   out_prefix="tfs_pred_growth",
                   genotypes_file=None,
                   titrant_names_file=None,
                   titrant_concs_file=None,
                   only_files=False,
                   num_samples=0,
                   num_marginal_samples=None,
                   genotype_batch_size=None,
                   subset_genotypes=False,
                   subset_seed=None):
    """
    Predict growth signal (ln_cfu) from a fitted hierarchical model.

    By default predicts at all (genotype, titrant_name, titrant_conc,
    replicate, t_pre, t_sel) combinations present in the training data, unioned
    with any genotypes or concentrations supplied via file arguments.  Pass
    --only_files to predict exclusively at the file-specified inputs and skip
    training-data combinations.

    titrant_names_file is a post-prediction row filter (restrict-only): it
    narrows which titrant names appear in the output but does not affect which
    concentrations are predicted.  It is not subject to union semantics.

    A boolean column 'in_training_data' is added to the output: 1 if the
    (genotype, titrant_name, titrant_conc) triple was in the training data,
    0 otherwise.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.
    param_file : str
        Path to a posterior .h5 file produced by tfs-sample-posterior, or a
        MAP checkpoint .pkl file produced by tfs-fit-model.

        When a .pkl file is supplied the output contains a single ``point_est``
        column with no uncertainty information.

        To obtain uncertainty estimates from a MAP fit, first run
        tfs-sample-posterior on the .pkl checkpoint; it will construct a
        Laplace (Hessian-based) posterior approximation and write a .h5 file.
        Passing that .h5 here produces the full quantile columns (median,
        lower_95, upper_95, etc.).

        NUTS and SVI checkpoints are not supported directly; run
        tfs-sample-posterior first.
    out_prefix : str, optional
        Prefix for the output CSV file. Written to {out_prefix}.csv.
        Default 'tfs_pred_growth'.
    genotypes_file : str or None, optional
        Plain-text file with one genotype per line (slash-separated mutations,
        e.g. 'M42I/K84L', or 'wt'). These genotypes are unioned with all
        training genotypes unless --only_files is set. Default None.
    titrant_names_file : str or None, optional
        Plain-text file with one titrant name per line. Filters output rows to
        the named titrants (restrict-only; does not affect which concentrations
        are predicted). If None, all titrant names are included.
    titrant_concs_file : str or None, optional
        Plain-text file with one concentration per line. These concentrations
        are unioned with all training concentrations unless --only_files is set.
        Default None.
    only_files : bool, optional
        If True, predict only at the genotypes and concentrations supplied via
        file arguments, ignoring training-data combinations. Default False.
    num_samples : int or None, optional
        Number of joint posterior samples to include as sample_0 … sample_N-1
        columns alongside the quantile columns. Set to None for quantiles only.
        Default 0.
    num_marginal_samples : int or None, optional
        Number of posterior samples to run through the model when computing
        quantiles. If None, all available samples are used.
    genotype_batch_size : int or None, optional
        Maximum number of genotypes per predict() call. The genotype list is
        split into chunks of this size, each chunk is predicted separately,
        and the results are concatenated. Reduces peak memory at the cost of
        one JAX re-compilation per batch. If None (default), a batch size is
        estimated automatically from the available device memory and the
        per-genotype tensor cost; pass an explicit value to override.
    subset_genotypes : bool, optional
        If True, predict only a single memory-fit block of genotypes instead
        of every genotype. The block size is the auto-sized (or explicit)
        genotype_batch_size; the block always includes the binding genotypes,
        the spiked genotypes, and any genotypes supplied via genotypes_file,
        with the remainder of the block filled by a random sample of the other
        genotypes. Intended for quickly assessing the input/output ln_cfu
        correlation without paying for a full prediction sweep. Default False.
    subset_seed : int or None, optional
        Seed for the random draw used by subset_genotypes, making the sampled
        block reproducible. Ignored unless subset_genotypes is True. Default
        None (non-deterministic draw).
    """
    file_genotypes = read_lines(genotypes_file) if genotypes_file else []
    titrant_names = read_lines(titrant_names_file) if titrant_names_file else None
    file_concs = [float(x) for x in read_lines(titrant_concs_file)] if titrant_concs_file else []

    print(f"Loading configuration from {config_file}...", flush=True)
    orchestrator, _ = read_configuration(config_file)
    is_map = param_file.endswith(".pkl")
    param_file = resolve_param_file(param_file, orchestrator, out_prefix)

    # Build training-data membership set for in_training_data column.
    training_tuples = set(
        zip(orchestrator.growth_df["genotype"],
            orchestrator.growth_df["titrant_name"],
            orchestrator.growth_df["titrant_conc"])
    )

    if only_files:
        genotypes = file_genotypes if file_genotypes else None
        titrant_concs = file_concs if file_concs else None
    else:
        training_genotypes = list(orchestrator.growth_df["genotype"].unique())
        genotypes = list(dict.fromkeys(training_genotypes + file_genotypes)) if file_genotypes else None
        training_concs = list(orchestrator.growth_df["titrant_conc"].unique())
        titrant_concs = sorted(set(training_concs) | set(file_concs)) if file_concs else None

    # Batching (manual or auto-sized) requires an explicit genotype list to
    # split into chunks; resolve it when no file restriction narrowed it.
    if genotypes is None:
        genotypes = list(orchestrator.growth_df["genotype"].unique())

    if genotype_batch_size is None:
        _, resolved_posteriors = load_posteriors(param_file, q_to_get=None)
        first_key = next(iter(resolved_posteriors.keys()))
        total_available = resolved_posteriors[first_key].shape[0]
        n_for_quantiles = (total_available if num_marginal_samples is None
                          else min(num_marginal_samples, total_available))
        genotype_batch_size = estimate_genotype_batch_size(
            orchestrator,
            predict_sites=["growth_pred"],
            num_marginal_samples=n_for_quantiles,
        )
        print(f"Auto-sized genotype_batch_size to {genotype_batch_size} "
              f"genotypes based on available memory.", flush=True)

    # Binding genotypes (those with direct theta_obs measurements) must appear
    # in every batch so the binding TensorManager is never empty.
    try:
        binding_genos = [str(g) for g in orchestrator.binding_df["genotype"].unique()]
    except Exception:
        binding_genos = []
    binding_set = set(binding_genos)

    # Subset mode: predict a single memory-fit block of genotypes rather than
    # every genotype, for a fast input/output ln_cfu correlation check. The
    # block always keeps the binding, spiked, and file-specified genotypes;
    # the rest of the block is a random sample of the remaining genotypes.
    if subset_genotypes:
        try:
            spiked_list = orchestrator.settings.get("spiked_genotypes") or []
            spiked_set = set(str(g) for g in spiked_list)
        except Exception:
            spiked_set = set()

        universe = list(genotypes)
        universe_set = set(universe)
        # Mandatory-keep genotypes, restricted to those actually in the
        # universe (file genotypes may be novel/absent under some paths).
        keep_set = (binding_set | spiked_set | set(file_genotypes)) & universe_set
        keep = [g for g in universe if g in keep_set]          # preserve order
        remaining = [g for g in universe if g not in keep_set]

        block = genotype_batch_size
        n_random = max(0, block - len(keep))
        rng = np.random.default_rng(subset_seed)
        if n_random < len(remaining):
            idx = rng.choice(len(remaining), size=n_random, replace=False)
            sampled = [remaining[i] for i in sorted(idx)]
        else:
            sampled = remaining

        genotypes = keep + sampled

        if len(keep) > block:
            print(f"WARNING: {len(keep)} mandatory (binding/spiked/file) "
                  f"genotypes exceed the auto-sized block of {block}; "
                  f"predicting all of them in one call anyway.", flush=True)
        print(f"Subset mode: predicting {len(genotypes)} genotypes "
              f"({len(keep)} mandatory + {len(sampled)} random) in a single "
              f"block.", flush=True)

    q_to_get = [0.5] if is_map else None
    print("Running growth predictions...", flush=True)

    predict_kwargs = dict(
        orchestrator=orchestrator,
        param_posteriors=param_file,
        predict_sites=["growth_pred"],
        num_samples=num_samples,
        num_marginal_samples=num_marginal_samples,
        titrant_conc=titrant_concs,
        q_to_get=q_to_get,
    )

    if subset_genotypes:
        # Subset mode is always a single block; retry with fewer sampled
        # genotypes if the device runs out of memory (the estimate is only a
        # guess and can overshoot on GPU).
        result_df = _predict_subset_with_backoff(predict_kwargs, keep, sampled)
    elif genotype_batch_size is not None and genotypes is not None and len(genotypes) > genotype_batch_size:
        batches = [genotypes[i:i + genotype_batch_size]
                   for i in range(0, len(genotypes), genotype_batch_size)]
        n_batches = len(batches)
        batch_dfs = []
        for batch_idx, batch in enumerate(batches, 1):
            print(f"  Batch {batch_idx}/{n_batches} ({len(batch)} genotypes)...", flush=True)
            # Binding genotypes (those with direct theta_obs measurements) must
            # appear in every predict() call so the binding TensorManager is
            # never empty.  Prepend any that are missing from this chunk, then
            # strip their rows from the result so each binding genotype appears
            # only once — in the batch where it falls naturally.
            batch_set = set(batch)
            extra_binding = [g for g in binding_genos if g not in batch_set]
            run_genotypes = extra_binding + list(batch) if extra_binding else list(batch)
            batch_df = predict(**predict_kwargs, genotypes=run_genotypes)
            if extra_binding:
                extra_set = set(extra_binding)
                batch_df = batch_df[~batch_df["genotype"].isin(extra_set)].reset_index(drop=True)
            batch_dfs.append(batch_df)
        result_df = pd.concat(batch_dfs, ignore_index=True)
    else:
        result_df = predict(**predict_kwargs, genotypes=genotypes)

    # Apply titrant_name filter post-prediction.
    if titrant_names is not None:
        result_df = result_df[result_df["titrant_name"].isin(titrant_names)].reset_index(drop=True)

    result_df["in_training_data"] = result_df.apply(
        lambda row: int((row["genotype"], row["titrant_name"], row["titrant_conc"])
                        in training_tuples),
        axis=1,
    )

    out_file = f"{out_prefix}.csv"
    result_df.to_csv(out_file, index=False)
    print(f"Wrote {len(result_df)} rows to {out_file}", flush=True)


def main():
    generalized_main(predict_growth,
                     manual_arg_types={"genotypes_file": str,
                                       "titrant_names_file": str,
                                       "titrant_concs_file": str,
                                       "num_marginal_samples": int,
                                       "genotype_batch_size": int,
                                       "subset_seed": int,
                                       "only_files": bool,
                                       "subset_genotypes": bool})


if __name__ == "__main__":
    main()
