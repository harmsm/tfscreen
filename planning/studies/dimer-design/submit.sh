#!/bin/bash
# Submit dimer-design runs as chained SLURM jobs, from this directory:
#
#   bash submit.sh                 every run in RUNS
#   bash submit.sh N_c1 W_c1       just these
#
# Per run: simulate (skipped when runs/<run>/sim exists); then, for each
# arm, the configure and MAP in one job and the Laplace, extract, predict
# and summarize in a second, after the first. Logs go to runs/<run>/logs/.
# Each step skips finished work, so resubmitting resumes a run. The
# configs (runs/, not tracked) are written here, once, by make_configs.py
# before anything is submitted; it needs the tfscreen environment's python.
set -e
RUNS="${*:-W_c1 N_c1 R_c1 W_c03 N_c03 R_c03 N_c1_d12 N_c1_d18}"
ARMS="${ARMS:-PK PddG_s1 PddG_dial}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRUN="${HERE}/dimer_design.srun"
(cd "${HERE}" && python make_configs.py)
for RUN in ${RUNS}; do
    LOGS="${HERE}/runs/${RUN}/logs"
    mkdir -p "${LOGS}"
    sub() {  # name dependency exports
        local name=$1 dep=$2 exports=$3
        sbatch --parsable --job-name="dd_${RUN}_${name}" \
            --output="${LOGS}/${name}_%j.out" \
            --error="${LOGS}/${name}_%j.err" \
            ${dep:+--dependency=afterok:${dep}} \
            --export=ALL,HERE="${HERE}",RUN="${RUN}",${exports} "${SRUN}"
    }
    sim=""
    if [[ ! -f "${HERE}/runs/${RUN}/sim/tfs_sim_growth.csv" ]]; then
        sim=$(sub simulate "" STEP=simulate)
        echo "${RUN} simulate ${sim}"
    fi
    for arm in ${ARMS}; do
        a=$(sub "${arm}_map" "${sim}" "STEP=fit,ARM=${arm},STOP_AFTER=map")
        b=$(sub "${arm}_rest" "${a}" "STEP=fit,ARM=${arm}")
        echo "${RUN} ${arm}: map ${a}; laplace and the rest ${b}"
    done
done
