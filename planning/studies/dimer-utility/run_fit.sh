#!/usr/bin/env bash
# One stage A fit, end to end: configure, staged MAP, full Laplace, the MAP
# point, extract, predict theta, summarize against the simulation truth.
#
#   bash run_fit.sh <run> <arm>      e.g. bash run_fit.sh A1 dimerPK
#
# Arms: O2PK (thermo.O2_C12_K5_U1_a.PK), dimerPK_nested (thermo.dimer.PK,
# second engagement off), dimerPK and dimerPddG (default variant). A1 fits
# without the pair term, A2 and A3 with it. A3 (plan step 3g) adds the
# contexts file and holds every hierarchical scale at 1 kT, as the EM
# iteration 1 fits of A2 did (dimer_em.py). Each step is skipped if its marker in
# fit_<arm>/.done exists. MAP_CAP (default 50000) caps every MAP stage.
# LAPLACE sets the posterior: full for A1 and A2, arrowhead for A3, whose
# 32,001 MAP parameters made the dense full Laplace need about 67 GB.
set -euo pipefail

run=$1; arm=$2
here=$(cd "$(dirname "$0")" && pwd)
bin=~/miniconda3/bin
MAP_CAP=${MAP_CAP:-50000}
NUM_DRAWS=${NUM_DRAWS:-500}
[[ ${run} == A3 ]] && default_laplace=arrowhead || default_laplace=full
LAPLACE=${LAPLACE:-${default_laplace}}

cd "${here}/runs/${run}"
[[ -f sim/tfs_sim_growth.csv ]] || { echo "runs/${run} is not simulated"; exit 1; }
mkdir -p "fit_${arm}/.done"
cd "fit_${arm}"
exec > >(tee -a job.log) 2>&1

done_step() { [[ -f ".done/$1" ]]; }
mark() { touch ".done/$1"; }
timed() { /usr/bin/time -l "$@"; }

thermo=../../../inputs/2026-10-11_laci-state-ddg.csv
case "${arm}" in
    O2PK)           theta=(--theta_model thermo.O2_C12_K5_U1_a.PK) ;;
    dimerPK_nested) theta=(--theta_model thermo.dimer.PK
                           --set_priors theta_second_engagement=0) ;;
    dimerPK)        theta=(--theta_model thermo.dimer.PK) ;;
    dimerPddG)      theta=(--theta_model thermo.dimer.PddG
                           --thermo_data "${thermo}"
                           --thermo_ddG_units kcal/mol
                           --thermo_temperature_K 310.15) ;;
    *) echo "unknown arm ${arm}"; exit 1 ;;
esac
epi=()
[[ "${run}" == A2 || "${run}" == A3 ]] && epi=(--epistasis)
if [[ "${run}" == A3 ]]; then
    held=(theta_sigma_d_ln_K_h_oii_fixed=1 theta_sigma_d_ln_K_l_oii_fixed=1)
    if [[ "${arm}" == dimerPK ]]; then
        held+=(theta_sigma_d_ln_K_h_l_fixed=1 theta_sigma_d_ln_K_h_o_fixed=1
               theta_sigma_d_ln_K_l_o_fixed=1 theta_sigma_d_ln_K_u_fixed=1
               theta_sigma_d_ln_K_h_e_fixed=1 theta_sigma_d_ln_K_l_e_fixed=1)
    else
        held+=(theta_ddG_sigma_H_fixed=1 theta_ddG_sigma_HO_fixed=1
               theta_ddG_sigma_L_fixed=1 theta_ddG_sigma_LO_fixed=1
               theta_ddG_sigma_HE2_fixed=1 theta_ddG_sigma_LE2_fixed=1)
    fi
    epi+=(--contexts ../../../inputs/contexts_A3.yaml --set_priors "${held[@]}")
fi

if ! done_step configure; then
    echo ">>> configure ${run} ${arm} $(date)"
    timed ${bin}/tfs-configure-model \
        --growth_df ../sim/tfs_sim_growth.csv \
        --binding_df ../sim/tfs_sim_binding.csv \
        --library_config ../tfs_sim_config.yaml \
        --growth_shares_replicates \
        "${theta[@]}" ${epi[@]+"${epi[@]}"}
    mark configure
fi
if ! done_step map; then
    echo ">>> MAP (cap ${MAP_CAP} per stage) $(date)"
    timed ${bin}/tfs-fit-model tfs_configure_config.yaml \
        --analysis_method map --seed 1 --epoch_checkpoint_interval 0 \
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
