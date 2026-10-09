#!/bin/bash
# One factorial arm (planning/studies/m-bias-factorial), run inside the arm's
# directory: simulate, process, configure, MAP, extract. The chain and model
# are the full-size simulation's (run_full_size.srun), without the Laplace.
set -e
SEED="$(grep '^seed:' simulate_config.yaml | awk '{print $2}')"

[[ -f tfs_sim_tubes.csv ]] || tfs-simulate simulate_config.yaml \
    --out_prefix tfs_sim --seed "${SEED}" --no_write_growth > sim.log 2>&1

[[ -f tfs_growth.csv ]] || tfs-process-counts tfs_sim_tubes.csv tfs_sim_counts \
    --od600_file tfs_sim_tube_od600.csv \
    --od600_calibration_file tfs_sim_od600_calibration.yaml \
    --tube_volume_mL 5 --out_prefix tfs_growth --no_verbose > process.log 2>&1

[[ -f tfs_configure_config.yaml ]] || tfs-configure-model \
    --allow_missing_spikes \
    --growth_df tfs_growth.csv \
    --library_config library_config.yaml \
    --theta_model hill_relative \
    --growth_shares_replicates \
    --batch_size 4096 \
    --growth_priors growth_priors_loose.csv \
    --set_priors sigma_fixed=0.17 theta_log_hill_n_hyper_scale_fixed=0.5 \
    --skip_model_stats > configure.log 2>&1

[[ -f tfs_fit_model_params.npz ]] || tfs-fit-model tfs_configure_config.yaml \
    --seed "${SEED}" --analysis_method map > fit.log 2>&1

tfs-extract-params tfs_configure_config.yaml tfs_fit_model_checkpoint.pkl \
    > extract.log 2>&1
echo "done"
