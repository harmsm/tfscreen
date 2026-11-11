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
# Mis-specification arms (2026-11-07): PK_m0 and PddG_s1_m0 are PK and
# PddG_s1 fit without the monomer states (theta_monomers off, so no
# unfolded monomer), with the unfolded dimer on instead: the standard
# MWC-dimer view, folding as dimer unfolding. The truth keeps monomers.
# Joint MAP does not estimate these scales (dimer-utility, A2: sigma_s
# collapsed to 0), so every arm holds them.
#
# Any arm with the suffix _he (PK_he, PddG_s1_he, PK_m0_he, ...) also holds
# the pair horseshoe's global scale tau and slab variance c2 at the values
# that generated the truth (2026-11-10): the tau test found that joint MAP
# runs both away (tau about 800, c2 about 600) even on additive truths, and
# that this is the objective's optimum. EPI_TAU defaults to the truth's
# HalfCauchy scale, 0.3 for the *_ep runs and 1e-9 otherwise (effectively
# an additive fit); EPI_C2 defaults to 4, the slab scale squared
# (theta_epi_slab_scale 2 in truth and fit). Each is overridable.
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
epi_held=()
model_arm=${arm}
if [[ "${arm}" == *_he ]]; then
    model_arm=${arm%_he}
    if [[ "${run}" == *_ep || "${run}" == *_ep_* ]]; then
        EPI_TAU=${EPI_TAU:-0.3}
    else
        EPI_TAU=${EPI_TAU:-1e-9}
    fi
    EPI_C2=${EPI_C2:-4}
    epi_held=(theta_epi_tau_fixed="${EPI_TAU}" theta_epi_c2_fixed="${EPI_C2}")
fi
base_arm=${model_arm%_m0}
if [[ "${model_arm}" == *_m0 ]]; then
    variant=(theta_monomers=0 theta_unfolded_monomer=0 theta_unfolded_dimer=1)
    pk_terms="h_l h_o l_o h_e l_e u"
else
    variant=(theta_monomers=1 theta_unfolded_monomer=1 theta_unfolded_dimer=0)
    pk_terms="h_l h_o l_o h_e l_e d_h m_h_l m_h_o m_l_o m_u"
fi
variant+=(theta_sigma_d_ln_K_h_oii_fixed=1 theta_sigma_d_ln_K_l_oii_fixed=1)
pddg=(--theta_model thermo.dimer.PddG --thermo_data "${thermo}"
      --thermo_ddG_units kcal/mol --thermo_temperature_K 310.15)
case "${base_arm}" in
    PK)
        theta=(--theta_model thermo.dimer.PK)
        held=()
        for t in ${pk_terms}; do
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
        --contexts "../../../inputs/contexts_${run%%_*}.yaml" \
        "${theta[@]}" --set_priors "${variant[@]}" "${held[@]}" ${epi_held[@]+"${epi_held[@]}"}
    mark configure
fi
if ! done_step map; then
    echo ">>> MAP (cap ${MAP_CAP} per stage) $(date)"
    # Stages 1 and 2 of the staged MAP resume from their own files; a job
    # that timed out in stage 3 left the final checkpoint, which
    # tfs-fit-model resumes only when it is named.
    resume=()
    [[ -f tfs_fit_model_checkpoint.pkl ]] && \
        resume=(--checkpoint_file tfs_fit_model_checkpoint.pkl)
    timed ${bin}/tfs-fit-model tfs_configure_config.yaml \
        --analysis_method map --seed 1 --epoch_checkpoint_interval "${CKPT}" \
        --max_num_epochs "${MAP_CAP}" "${resume[@]}"
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
