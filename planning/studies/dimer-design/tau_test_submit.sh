#!/bin/bash
# Submit the tau test (tau_test.sh), from this directory:  bash tau_test_submit.sh
#
# Same truth (c1, N design), both directions, each with its control:
#   runaway fits  N_c1 PK (tau 841), N_c1_s2 PddG_s1 (527)  -> start small
#   small fits    N_c1_s3 PK (6e-6), N_c1 PddG_s1 (7e-6)    -> start large
#   with epistasis in the truth: N_ep PddG_s1 (739)          -> start small
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
for fit in N_c1:PK N_c1_s2:PddG_s1 N_c1_s3:PK N_c1:PddG_s1 N_ep:PddG_s1; do
    run=${fit%%:*}; arm=${fit#*:}
    for mode in cont flip; do
        id=$(sbatch --parsable --job-name="tt_${run}_${arm}_${mode}" \
            --output="${HERE}/runs/${run}/logs/tt_${arm}_${mode}_%j.out" \
            --error="${HERE}/runs/${run}/logs/tt_${arm}_${mode}_%j.err" \
            --export=ALL,HERE="${HERE}",RUN="${run}",ARM="${arm}",MODE="${mode}" \
            "${HERE}/tau_test.srun")
        echo "${run} ${arm} ${mode} ${id}"
    done
done
