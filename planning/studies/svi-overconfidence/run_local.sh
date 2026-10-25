#!/bin/bash
# Run a rendered grid locally instead of through SLURM: each run's run.srun
# with the cluster's `module load` line dropped, JOBS runs at a time.
#
#   bash run_local.sh svi_overconfidence [JOBS] [PYTHON_BIN_DIR] [TFSCREEN_SRC]
#
# PYTHON_BIN_DIR holds python and the tfs-* entry points (default
# ~/miniconda3/bin); TFSCREEN_SRC, if given, goes first on PYTHONPATH so the
# runs use that checkout. Each run logs to its run.out/run.err; a run that
# already finished (run.out ends with ">>> Done") is skipped, so the script
# can be restarted.

set -u
GRID="$1"
JOBS="${2:-3}"
BIN="${3:-$HOME/miniconda3/bin}"
SRC="${4:-}"

export PATH="${BIN}:${PATH}"
if [[ -n "${SRC}" ]]; then
    export PYTHONPATH="${SRC}${PYTHONPATH:+:${PYTHONPATH}}"
fi
run_one() {
    d="$1"
    if [[ -f "$d/run.out" ]] && tail -1 "$d/run.out" | grep -q ">>> Done"; then
        echo "skip $d (done)"
        return 0
    fi
    rm -rf "$d/checkpoints" "$d"/tfs_fit_model_* "$d"/tfs_prefit_* 2>/dev/null
    echo "start $d $(date +%H:%M:%S)"
    ( cd "$d" && sed '/^module load/d' run.srun | bash > run.out 2> run.err )
    status=$?
    echo "end   $d $(date +%H:%M:%S) status ${status}"
}
export -f run_one

ls -d "${GRID}"/run_*/ | sed 's#/$##' | xargs -P "${JOBS}" -I{} bash -c 'run_one "$@"' _ {}
