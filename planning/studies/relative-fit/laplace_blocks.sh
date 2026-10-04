#!/bin/bash
# Block and arrowhead Laplace on run 4's MAP runs (README, "Block Laplace on
# run 4's MAPs" and "Arrowhead Laplace on run 4's MAPs").
#
#   bash laplace_blocks.sh block      # --laplace_blocks (shared params held)
#   bash laplace_blocks.sh arrowhead  # --laplace_blocks --laplace_shared
#
# Run from this directory after relative_fit_v4/ is complete. Each
# relative_fit_v4/run_*_map is copied (config, priors, guesses, library,
# checkpoint, simulated truth) into relative_fit_v4_<mode>/, so the map arm's
# own posterior is left alone. Then 500 draws, extract, predict, summarize,
# and tfs-summarize-calibration pools the six runs into
# relative_fit_v4_<mode>/calib_<mode>_*. About 5 minutes per run on a laptop
# CPU; RUN_JOBS (default 3) runs that many at once.
#
# Tables in the README: block at cd208d15, arrowhead at a874ac38.

set -euo pipefail

mode="${1:?usage: laplace_blocks.sh block|arrowhead}"
case "${mode}" in
    block) laplace_args=(--laplace_blocks) ;;
    arrowhead) laplace_args=(--laplace_blocks --laplace_shared) ;;
    *) echo "unknown mode '${mode}'" >&2; exit 1 ;;
esac

src=relative_fit_v4
out="relative_fit_v4_${mode}"
mkdir -p "${out}"
cp "${src}/grid_summary.json" "${out}/"

one_run() {
    local run="$1" dest
    dest="${out}/$(basename "${run}")"
    mkdir -p "${dest}"
    cp "${run}"/combo.json "${run}"/tfs_configure_* "${run}"/tfs_sim_* \
       "${run}"/tfs_fit_model_checkpoint.pkl "${dest}/"
    (
        cd "${dest}"
        tfs-sample-posterior tfs_configure_config.yaml tfs_fit_model_checkpoint.pkl \
            "${laplace_args[@]}" --num_posterior_samples 500 \
            --sampling_batch_size 100 > sample.log 2>&1
        tfs-extract-params tfs_configure_config.yaml tfs_posterior.h5 > extract.log 2>&1
        tfs-predict-theta tfs_configure_config.yaml tfs_posterior.h5 > theta.log 2>&1
        tfs-predict-growth tfs_configure_config.yaml tfs_posterior.h5 \
            --num_marginal_samples=500 > growth.log 2>&1
        tfs-summarize-fit . > summary.log 2>&1
    )
    echo "done ${dest}"
}
export -f one_run
export out mode
export laplace_args_str="${laplace_args[*]}"

ls -d "${src}"/run_*_map | xargs -P "${RUN_JOBS:-3}" -I{} bash -c \
    'set -euo pipefail; laplace_args=(${laplace_args_str}); one_run "$1"' _ {}

tfs-summarize-calibration "${out}" --out_prefix "${out}/calib_${mode}"
