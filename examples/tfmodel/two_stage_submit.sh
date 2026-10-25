#!/bin/bash
# Submit tfs-fit-two-stage as chained SLURM jobs:
#
#   bash two_stage_submit.sh two_stage.yaml out_dir
#
# stage1; then stage2; then predict and consistency (after stage2); then
# the summary table (after both). With two_stage.kshift true (experimental)
# the jackknife array runs beside stage2 and kshift runs before predict. Logs go to out_dir/logs/. Every path is made
# absolute, so it runs from any directory. Each step skips finished work,
# so submitting the chain again resumes it. Set JACKKNIFE_PARALLEL to run
# more or fewer jackknife refits at once (default 10).
set -e
CFG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUT="$(mkdir -p "$2" && cd "$2" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRUN="${HERE}/two_stage_step.srun"
N="$(python -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); print((c.get('two_stage') or {}).get('jackknife_subsamples', 30))" "${CFG}")"
KSHIFT="$(python -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); print(bool((c.get('two_stage') or {}).get('kshift', False)))" "${CFG}")"
mkdir -p "${OUT}/logs"
sub() {  # name dependency extra-args...
    local name=$1 dep=$2; shift 2
    sbatch --parsable --job-name="ts_${name}" \
        --output="${OUT}/logs/${name}_%A_%a.out" --error="${OUT}/logs/${name}_%A_%a.err" \
        ${dep:+--dependency=afterok:${dep}} \
        --export=ALL,CFG="${CFG}",OUT="${OUT}",STEP="${name}" "$@" "${SRUN}"
}
a=$(sub stage1 "")
c=$(sub stage2 "${a}")
if [ "${KSHIFT}" = "True" ]; then
    b=$(sub jackknife "${a}" --array=1-${N}%${JACKKNIFE_PARALLEL:-10})
    d=$(sub kshift_predict "${b}:${c}")
else
    d=$(sub predict "${c}")
fi
e=$(sub consistency "${c}")
f=$(sub summary "${d}:${e}")
echo "stage1 ${a}; stage2 ${c}${b:+; jackknife ${b} (1-${N})}; predict ${d}; consistency ${e}; summary ${f}"
