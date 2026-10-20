#!/bin/bash
# Submit tfs-fit-two-stage as chained SLURM jobs:
#
#   bash two_stage_submit.sh two_stage.yaml out_dir
#
# stage1; then the jackknife array and stage2 in parallel; then kshift +
# predict (after both) and consistency (after stage2); then the summary
# table (after both). Logs go to out_dir/logs/. Every path is made
# absolute, so it runs from any directory. Each step skips finished work,
# so submitting the chain again resumes it. Set JACKKNIFE_PARALLEL to run
# more or fewer jackknife refits at once (default 10).
set -e
CFG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUT="$(mkdir -p "$2" && cd "$2" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRUN="${HERE}/two_stage_step.srun"
N="$(python -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); print((c.get('two_stage') or {}).get('jackknife_subsamples', 30))" "${CFG}")"
mkdir -p "${OUT}/logs"
sub() {  # name dependency extra-args...
    local name=$1 dep=$2; shift 2
    sbatch --parsable --job-name="ts_${name}" \
        --output="${OUT}/logs/${name}_%A_%a.out" --error="${OUT}/logs/${name}_%A_%a.err" \
        ${dep:+--dependency=afterok:${dep}} \
        --export=ALL,CFG="${CFG}",OUT="${OUT}",STEP="${name}" "$@" "${SRUN}"
}
a=$(sub stage1 "")
b=$(sub jackknife "${a}" --array=1-${N}%${JACKKNIFE_PARALLEL:-10})
c=$(sub stage2 "${a}")
d=$(sub kshift_predict "${b}:${c}")
e=$(sub consistency "${c}")
f=$(sub summary "${d}:${e}")
echo "stage1 ${a}; jackknife ${b} (1-${N}); stage2 ${c}; kshift_predict ${d}; consistency ${e}; summary ${f}"
