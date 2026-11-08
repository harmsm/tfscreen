#!/bin/bash
# Submit the 2026-11-07 error panel (make_configs.py, PANEL) on CPU nodes,
# from this directory:
#
#   bash submit_panel.sh           trimmed: G (ten greedy backgrounds) in the
#                                  c1 condition only, 46 fits
#   FULL=1 bash submit_panel.sh    G in every condition, 64 fits
#
# Conditions: c1 (random table error), c03 (plus systematic compression),
# ep (plus random pair epistasis), and the mis-specified fits PK_m0 and
# PddG_s1_m0 (no monomer states) on the c1 simulations. Arms PK and
# PddG_s1. Seed 1 of W and N in c1 and c03 are the pilots' fits, so only
# their new arms are submitted. Each run's arms go in one submit.sh call,
# so a run is simulated once.
set -e
cd "$(dirname "$0")"
FIT="PK PddG_s1"
MIS="PK_m0 PddG_s1_m0"
for d in W N; do
    ARMS="${MIS}" bash submit.sh ${d}_c1
    ARMS="${FIT} ${MIS}" bash submit.sh ${d}_c1_s2 ${d}_c1_s3
    ARMS="${FIT}" bash submit.sh ${d}_c03_s2 ${d}_c03_s3 ${d}_ep ${d}_ep_s2 ${d}_ep_s3
done
G_ARMS="${FIT}"
[[ -n "${FULL:-}" ]] && G_ARMS="${FIT} ${MIS}"
MEM=48G ARMS="${G_ARMS}" bash submit.sh G_c1 G_c1_s2 G_c1_s3
if [[ -n "${FULL:-}" ]]; then
    MEM=48G ARMS="${FIT}" bash submit.sh G_c03 G_c03_s2 G_c03_s3 G_ep G_ep_s2 G_ep_s3
fi
