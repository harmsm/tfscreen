#!/usr/bin/env bash
# One stage A fit, end to end: configure, staged MAP, full Laplace, the MAP
# point, extract, predict theta, summarize against the simulation truth.
#
#   bash run_fit.sh <run> <arm>      e.g. bash run_fit.sh A1 dimerPK
#
# Arms: O2PK (thermo.O2_C12_K5_U1_a.PK), dimerPK_nested (thermo.dimer.PK,
# second engagement off), dimerPK and dimerPddG (default variant). A1 fits
# without the pair term, A2 with it. Each step is skipped if its marker in
# fit_<arm>/.done exists. MAP_CAP (default 50000) caps every MAP stage.
set -euo pipefail

run=$1; arm=$2
here=$(cd "$(dirname "$0")" && pwd)
bin=~/miniconda3/bin
MAP_CAP=${MAP_CAP:-50000}
NUM_DRAWS=${NUM_DRAWS:-500}

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
[[ "${run}" == A2 ]] && epi=(--epistasis)

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
    echo ">>> full Laplace $(date)"
    timed ${bin}/tfs-sample-posterior tfs_configure_config.yaml \
        tfs_fit_model_checkpoint.pkl --laplace full \
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
             growth_parameters transformation_lam ddG_sigma; do
        [[ -f "../sim/tfs_sim_${f}.csv" ]] \
            && ln -sf "../sim/tfs_sim_${f}.csv" "tfs_sim_${f}.csv"
    done
    ${bin}/tfs-summarize-fit .
    mark summarize
fi
echo ">>> ${run} ${arm} done $(date)"
