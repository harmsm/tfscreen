#!/bin/bash -l
#SBATCH --account=harmslab      ### change this to your actual account for charging
#SBATCH --job-name=tfs-guide    ### job name
#SBATCH --output=hostname.out   ### file in which to store job stdout
#SBATCH --error=hostname.err    ### file in which to store job stderr
#SBATCH --partition=gpu
#SBATCH --time=00-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0

# ---------------------------------------------------------------------------
# Guide-calibration run: simulate -> fit with one variational guide ->
# posterior -> predictions -> calibration summary against the simulated truth.
#
# Rendered by tfs-setup-sim-grid (grid.yaml) into one subdirectory per run.
# tfs-setup-sim-grid has already written tfs_sim_config.yaml here (base config
# plus this run's simulate: overrides, e.g. the simulation seed).  The fit
# choices below come from grid.yaml's template: blocks.  Run from this
# directory:  sbatch run.sh   (or: bash run.sh)
# ---------------------------------------------------------------------------

# On a cluster keep "module load"; on a local CPU comment it out and uncomment
# XLA_FLAGS (8 virtual CPU devices for JAX).
module load cuda/12.4.1
#export XLA_FLAGS="--xla_force_host_platform_device_count=8"

set -e

GUIDE_TYPE="{{ guide_type }}"
GUIDE_FLAGS="{% if guide_rank %}--guide_rank {{ guide_rank }}{% endif %}"
BATCH_SIZE={{ batch_size }}
THETA_GROWTH_NOISE="{{ theta_growth_noise_model }}"
FIT_SEED={{ fit_seed }}

echo ">>> guide=${GUIDE_TYPE} ${GUIDE_FLAGS} batch_size=${BATCH_SIZE} theta_growth_noise=${THETA_GROWTH_NOISE} fit_seed=${FIT_SEED}"

# ---------------------------------------------------------------------------
# 1. Simulate (seed comes from tfs_sim_config.yaml)
# ---------------------------------------------------------------------------
echo ">>> Simulate, `date`"
tfs-simulate tfs_sim_config.yaml .

# ---------------------------------------------------------------------------
# 2. Configure -- matches the production fit (dev/run.sh) except for the
#    batch size and theta_growth_noise model, which are grid factors.
# ---------------------------------------------------------------------------
echo ">>> Configure, `date`"
tfs-configure-model \
    tfs_sim_binding.csv \
    --growth_df tfs_sim_growth.csv \
    --presplit_df tfs_sim_presplit.csv \
    --base_growth_df tfs_sim_base_growth.csv \
    --transformation_lambda 0.3572 0.1296 \
    --condition_growth_model linear \
    --growth_transition_model instant \
    --ln_cfu0_model hierarchical_factored \
    --dk_geno_model hierarchical_geno \
    --activity_model fixed \
    --theta_model hill_mut \
    --transformation_model empirical \
    --theta_rescale_model passthrough \
    --theta_growth_noise_model ${THETA_GROWTH_NOISE} \
    --theta_binding_noise_model zero \
    --growth_noise_model normal_kt \
    --spiked wt M42I H74A K84L M42I/H74A M42I/K84L H74A/K84L M42I/H74A/K84L D88A \
    --growth_shares_replicates \
    --epistasis \
    --batch_size ${BATCH_SIZE}

# ---------------------------------------------------------------------------
# 3. Pre-fit calibration (no --pin_m: the binding anchors identify m)
# ---------------------------------------------------------------------------
echo ">>> Prefit calibration, `date`"
tfs-prefit-calibration tfs_configure_config.yaml --seed ${FIT_SEED}

# ---------------------------------------------------------------------------
# 4. Fit with the guide under test
# ---------------------------------------------------------------------------
echo ">>> Fit model, `date`"
tfs-fit-model \
    tfs_configure_config.yaml \
    --seed ${FIT_SEED} \
    --analysis_method svi \
    --guide_type ${GUIDE_TYPE} ${GUIDE_FLAGS} \
    --convergence_tolerance 0.0000005

# ---------------------------------------------------------------------------
# 5. Posterior and predictions
# ---------------------------------------------------------------------------
echo ">>> Sample posterior, `date`"
tfs-sample-posterior tfs_configure_config.yaml tfs_fit_model_checkpoint.pkl \
    --num_posterior_samples 500 --sampling_batch_size=5

echo ">>> Predict epistasis, `date`"
tfs-predict-epistasis tfs_configure_config.yaml tfs_posterior.h5 --scale_constant -0.6159

echo ">>> Extract parameter estimates, `date`"
tfs-extract-params tfs_configure_config.yaml tfs_posterior.h5

echo ">>> Predict theta, `date`"
tfs-predict-theta tfs_configure_config.yaml tfs_posterior.h5

echo ">>> Predict growth, `date`"
tfs-predict-growth tfs_configure_config.yaml tfs_posterior.h5 --num_marginal_samples=500 \
    --subset_genotypes --subset_seed 42

# ---------------------------------------------------------------------------
# 6. Summarize (writes the calibration outputs against tfs_sim_* ground truth)
# ---------------------------------------------------------------------------
echo ">>> Summarize fit, `date`"
tfs-summarize-fit .

echo ">>> Complete, `date`"
