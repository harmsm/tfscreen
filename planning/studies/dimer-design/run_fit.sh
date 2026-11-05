#!/usr/bin/env bash
# One dimer-design fit, end to end: configure, staged MAP, arrowhead
# Laplace, the MAP point, extract, predict theta, summarize against the
# simulation truth.
#
#   bash run_fit.sh <run> <arm>      e.g. bash run_fit.sh N_c1 PK
#
# Arms (plan step 4e), every one with the truth's variant (monomers and
# unfolded monomer on, unfolded dimer off), the pair term, the contexts
# file, and the Oii scales held at 1 kT:
#   PK          thermo.dimer.PK, every per-mutation scale held at 1 kT
#   PddG_s1     thermo.dimer.PddG, sigma_s held at 1 kT in every state
#   PddG_dial   thermo.dimer.PddG, sigma_s held at the truth's ddG_truth_sd
#               (0.5 kcal/mol = 0.811 kT): the noise is known, a table
#               slope (ddG_truth_scale) is not
# Joint MAP does not estimate these scales (dimer-utility, A2: sigma_s
# collapsed to 0), so every arm holds them.
#
# Each step is skipped if its marker in fit_<arm>/.done exists. MAP_CAP
# (default 50000) caps every MAP stage; CKPT (default 5000) is the MAP's
# checkpoint interval in epochs, so a MAP killed at a job's time limit
# resumes when resubmitted; LAPLACE (default arrowhead) sets the
# posterior. STOP_AFTER=<step> ends the script after that step, so a
# cluster job can stop after the MAP (dimer_design.srun). The tfs-*
# commands come from ~/miniconda3/bin (the laptop) or else from PATH.
set -euo pipefail

run=$1; arm=$2
here=$(cd "$(dirname "$0")" && pwd)
if [[ -x ~/miniconda3/bin/tfs-fit-model ]]; then
    bin=~/miniconda3/bin
else
    bin=$(dirname "$(command -v tfs-fit-model)")
fi
MAP_CAP=${MAP_CAP:-50000}
CKPT=${CKPT:-5000}
NUM_DRAWS=${NUM_DRAWS:-500}
STOP_AFTER=${STOP_AFTER:-}
LAPLACE=${LAPLACE:-arrowhead}

cd "${here}/runs/${run}"
[[ -f sim/tfs_sim_growth.csv ]] || { echo "runs/${run} is not simulated"; exit 1; }
mkdir -p "fit_${arm}/.done"
cd "fit_${arm}"
exec > >(tee -a job.log) 2>&1

done_step() {
    [[ -f ".done/$1" ]] || return 1
    if [[ "${STOP_AFTER}" == "$1" ]]; then
        echo ">>> $1 already done, stopping $(date)"; exit 0
    fi
}
mark() {
    touch ".done/$1"
    if [[ "${STOP_AFTER}" == "$1" ]]; then
        echo ">>> stopping after $1 $(date)"; exit 0
    fi
}
if [[ $(uname) == Darwin ]]; then
    timed() { /usr/bin/time -l "$@"; }
elif [[ -x /usr/bin/time ]]; then
    timed() { /usr/bin/time -v "$@"; }
else
    timed() { "$@"; }
fi

thermo=../../../inputs/2026-11-03_laci-state-ddg.csv
variant=(theta_monomers=1 theta_unfolded_monomer=1 theta_unfolded_dimer=0
         theta_sigma_d_ln_K_h_oii_fixed=1 theta_sigma_d_ln_K_l_oii_fixed=1)
pddg=(--theta_model thermo.dimer.PddG --thermo_data "${thermo}"
      --thermo_ddG_units kcal/mol --thermo_temperature_K 310.15)
case "${arm}" in
    PK)
        theta=(--theta_model thermo.dimer.PK)
        held=()
        for t in h_l h_o l_o h_e l_e d_h m_h_l m_h_o m_l_o m_u; do
            held+=("theta_sigma_d_ln_K_${t}_fixed=1")
        done ;;
    PddG_s1)   theta=("${pddg[@]}"); held=(theta_ddG_sigma_fixed=1) ;;
    PddG_dial) theta=("${pddg[@]}"); held=(theta_ddG_sigma_fixed=0.811) ;;
    *) echo "unknown arm ${arm}"; exit 1 ;;
esac

if ! done_step configure; then
    echo ">>> configure ${run} ${arm} $(date)"
    timed ${bin}/tfs-configure-model \
        --growth_df ../sim/tfs_sim_growth.csv \
        --binding_df ../sim/tfs_sim_binding.csv \
        --library_config ../tfs_sim_config.yaml \
        --growth_shares_replicates --epistasis \
        --contexts ../../../inputs/contexts.yaml \
        "${theta[@]}" --set_priors "${variant[@]}" "${held[@]}"
    mark configure
fi
if ! done_step map; then
    echo ">>> MAP (cap ${MAP_CAP} per stage) $(date)"
    timed ${bin}/tfs-fit-model tfs_configure_config.yaml \
        --analysis_method map --seed 1 --epoch_checkpoint_interval "${CKPT}" \
        --max_num_epochs "${MAP_CAP}"
    mark map
fi
if ! done_step laplace; then
    echo ">>> ${LAPLACE} Laplace $(date)"
    timed ${bin}/tfs-sample-posterior tfs_configure_config.yaml \
        tfs_fit_model_checkpoint.pkl --laplace "${LAPLACE}" \
        --num_posterior_samples "${NUM_DRAWS}" --skip_growth_observations
    mark laplace
fi
if ! done_step map_point; then
    echo ">>> MAP point $(date)"
    ${bin}/tfs-sample-posterior tfs_configure_config.yaml \
        tfs_fit_model_checkpoint.pkl --laplace point \
        --num_posterior_samples 1 --skip_growth_observations \
        --out_prefix tfs_map
    ${bin}/tfs-extract-params tfs_configure_config.yaml tfs_map.h5 \
        --out_prefix tfs_map_params
    ${bin}/tfs-predict-theta tfs_configure_config.yaml tfs_map.h5 \
        --out_prefix tfs_map_theta
    mark map_point
fi
if ! done_step extract; then
    ${bin}/tfs-extract-params tfs_configure_config.yaml tfs_posterior.h5
    ${bin}/tfs-predict-theta tfs_configure_config.yaml tfs_posterior.h5
    mark extract
fi
if ! done_step summarize; then
    for f in parameters mutation_parameters pair_parameters genotype_theta \
             growth_parameters transformation_lam ddG_sigma operator_shift \
             operator_dev ln_tf_total; do
        [[ -f "../sim/tfs_sim_${f}.csv" ]] \
            && ln -sf "../sim/tfs_sim_${f}.csv" "tfs_sim_${f}.csv"
    done
    ${bin}/tfs-summarize-fit .
    mark summarize
fi
echo ">>> ${run} ${arm} done $(date)"
