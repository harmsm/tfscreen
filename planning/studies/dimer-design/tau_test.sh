#!/usr/bin/env bash
# Is the pair horseshoe's runaway tau an optimizer trap or the MAP
# objective's optimum? (2026-11-09)
#
#   bash tau_test.sh <run> <arm> <mode>      e.g. bash tau_test.sh N_c1 PK flip
#
# Works in runs/<run>/tautest_<arm>_<mode>/, a copy of fit_<arm>'s
# configuration; fit_<arm> itself is never written.
#
#   cont   resume fit_<arm>'s final MAP checkpoint for EXTRA (default
#          100000) more epochs (a resume's max_num_epochs counts epochs
#          after the checkpoint): the control, run as long as flip. A
#          rerun after a time limit runs EXTRA more again.
#   flip   a joint MAP of EXTRA epochs from fit_<arm>'s final MAP with the
#          pair horseshoe moved to the other regime and every pair offset
#          set to 0 (so every pair shift starts at 0): a runaway fit
#          (tau > 1) starts at tau 0.01, c2 8/3 (the slab prior's mode);
#          a small-tau fit starts at tau 800, c2 600 (the runaway fits'
#          values). Same step size (1e-4) and convergence rules as cont.
#
# Both minimize the same objective on the same data, so their final
# exact losses (loss_exact in tfs_fit_model_convergence.csv) compare
# directly: tau_test_report.py prints them.
set -euo pipefail

run=$1; arm=$2; mode=$3
EXTRA=${EXTRA:-100000}
here=$(cd "$(dirname "$0")" && pwd)
if [[ -x ~/miniconda3/bin/tfs-fit-model ]]; then
    bin=~/miniconda3/bin
else
    bin=$(dirname "$(command -v tfs-fit-model)")
fi
src="${here}/runs/${run}/fit_${arm}"
dst="${here}/runs/${run}/tautest_${arm}_${mode}"
[[ -f "${src}/.done/map" ]] || { echo "${src}: MAP not finished"; exit 1; }
mkdir -p "${dst}"
cd "${dst}"
exec > >(tee -a job.log) 2>&1
echo ">>> tau test ${run} ${arm} ${mode} $(date) on $(hostname)"
for f in tfs_configure_config.yaml tfs_configure_priors.csv \
         tfs_configure_guesses.csv tfs_configure_library.csv; do
    [[ -f "${f}" ]] || cp "${src}/${f}" .
done

case "${mode}" in
    cont)
        if [[ ! -f tfs_fit_model_checkpoint.pkl ]]; then
            cp "${src}/tfs_fit_model_checkpoint.pkl" \
               "${src}/tfs_fit_model_convergence.csv" \
               "${src}/tfs_fit_model_losses.txt" .
            [[ -f "${src}/tfs_fit_model_losses.bin" ]] && \
                cp "${src}/tfs_fit_model_losses.bin" .
        fi
        ${bin}/tfs-fit-model tfs_configure_config.yaml \
            --analysis_method map --seed 1 --epoch_checkpoint_interval 5000 \
            --checkpoint_file tfs_fit_model_checkpoint.pkl \
            --max_num_epochs "${EXTRA}" ;;
    flip)
        # a rerun after a time limit resumes the checkpoint instead
        if [[ -f tfs_fit_model_checkpoint.pkl ]]; then
            start=(--checkpoint_file tfs_fit_model_checkpoint.pkl)
        else
            "${bin}/python" "${here}/tau_test_start.py" \
                "${src}/tfs_fit_model_params.npz" tau_start.npz
            start=(--init_from tau_start.npz)
        fi
        ${bin}/tfs-fit-model tfs_configure_config.yaml \
            --analysis_method map --seed 1 --epoch_checkpoint_interval 5000 \
            --stage_offsets off --adam_step_size 1e-4 \
            --max_num_epochs "${EXTRA}" "${start[@]}" ;;
    *) echo "mode is cont or flip"; exit 1 ;;
esac
echo ">>> tau test ${run} ${arm} ${mode} done $(date)"
