#!/bin/bash
# Submit one dimer-utility run as chained SLURM jobs, from this directory:
#
#   bash submit.sh A4
#
# simulate (skipped when runs/<run>/sim exists); then, for each arm, the
# configure and MAP in one job and the Laplace, extract, predict and
# summarize in a second, after the first. Logs go to runs/<run>/logs/.
# Each step skips finished work, so resubmitting resumes a run.
set -e
RUN="${1:?usage: bash submit.sh <run>}"
ARMS="${ARMS:-dimerPK dimerPddG}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRUN="${HERE}/dimer_utility.srun"
LOGS="${HERE}/runs/${RUN}/logs"
mkdir -p "${LOGS}"
sub() {  # name dependency exports
    local name=$1 dep=$2 exports=$3
    sbatch --parsable --job-name="du_${RUN}_${name}" \
        --output="${LOGS}/${name}_%j.out" --error="${LOGS}/${name}_%j.err" \
        ${dep:+--dependency=afterok:${dep}} \
        --export=ALL,HERE="${HERE}",RUN="${RUN}",${exports} "${SRUN}"
}
sim=""
if [[ ! -f "${HERE}/runs/${RUN}/sim/tfs_sim_growth.csv" ]]; then
    sim=$(sub simulate "" STEP=simulate)
    echo "simulate ${sim}"
fi
for arm in ${ARMS}; do
    a=$(sub "${arm}_map" "${sim}" "STEP=fit,ARM=${arm},STOP_AFTER=map")
    b=$(sub "${arm}_rest" "${a}" "STEP=fit,ARM=${arm}")
    echo "${arm}: map ${a}; laplace and the rest ${b}"
done
