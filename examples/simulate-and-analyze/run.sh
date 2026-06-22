#!/bin/bash -l
#SBATCH --account=harmslab      ### change this to your actual account for charging
#SBATCH --job-name=tfscreen     ### job name
#SBATCH --output=hostname.out   ### file in which to store job stdout
#SBATCH --error=hostname.err    ### file in which to store job stderr
#SBATCH --partition=gpu
#SBATCH --time=01-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0

# On a cluster: comment out the XLA_FLAGS line and uncomment "module load cuda/..."
# On a local CPU: keep the XLA_FLAGS line (sets JAX to use 8 virtual CPU devices).
#module load cuda/12.4.1
export XLA_FLAGS="--xla_force_host_platform_device_count=8"

# Stop immediately on any error.
set -e

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
config_file="${1}"
run_dir="${2}"
if [[ ! "${run_dir}" || ! "${config_file}" ]]; then
    echo "Usage: run.sh config_file out_dir [seed]"
    exit 1
fi

seed="${3}"
if [[ ! "${seed}" ]]; then
    seed=1
fi

# ---------------------------------------------------------------------------
# 1. Simulate library
# ---------------------------------------------------------------------------
# tfs-simulate reads the config YAML and writes simulated growth data,
# binding curves, and ground-truth parameter CSVs to run_dir.
echo ">>> Simulate library"
tfs-simulate "${config_file}" "${run_dir}" --seed "${seed}"

cd "${run_dir}"

# ---------------------------------------------------------------------------
# 2. Configure model
# ---------------------------------------------------------------------------
# tfs-configure-model validates the data, maps labels to indices, selects
# model components, and writes tfs_configure_config.yaml + priors/guesses CSVs.
# Edit the flags here to change which model components are used.
echo ">>> Configure model"
tfs-configure-model \
    tfs_sim_binding.csv \
    --growth_df tfs_sim_growth.csv \
    --presplit_df tfs_sim_presplit.csv \
    --condition_growth_model linear \
    --growth_transition_model instant \
    --ln_cfu0_model hierarchical_factored \
    --dk_geno_model hierarchical_geno \
    --activity_model fixed \
    --theta_model hill_mut \
    --transformation_model single \
    --theta_rescale_model passthrough \
    --theta_growth_noise_model logit_normal \
    --theta_binding_noise_model zero \
    --growth_noise_model normal_kt \
    --spiked wt M42I H74A K84L M42I/H74A M42I/K84L H74A/K84L D88A \
    --growth_shares_replicates \
    --epistasis

# ---------------------------------------------------------------------------
# 3. Pre-fit calibration
# ---------------------------------------------------------------------------
# A fast MAP fit on a simplified model calibrates the growth-linking-function
# priors (m and b) before the full inference run.  Updates priors.csv and
# guesses.csv in place.
echo ">>> Pre-fit calibration"
tfs-prefit-calibration tfs_configure_config.yaml --seed "${seed}"

# ---------------------------------------------------------------------------
# 4. Fit model (SVI)
# ---------------------------------------------------------------------------
# Main hierarchical Bayesian inference.  SVI produces a full approximate
# posterior; for a point estimate only, use --analysis_method map instead.
echo ">>> Fit model"
tfs-fit-model \
    tfs_configure_config.yaml \
    --seed "${seed}" \
    --analysis_method svi \
    --convergence_tolerance 0.0001

# ---------------------------------------------------------------------------
# 5. Sample posterior
# ---------------------------------------------------------------------------
# Draw posterior samples from the SVI variational distribution and write
# them to an HDF5 file used by the prediction steps below.
echo ">>> Sample posterior"
tfs-sample-posterior tfs_configure_config.yaml tfs_fit_model_checkpoint.pkl

# ---------------------------------------------------------------------------
# 6. Extract parameter estimates
# ---------------------------------------------------------------------------
# Summarise the posterior into per-parameter CSV files (quantiles, means).
echo ">>> Extract parameter estimates"
tfs-extract-params tfs_configure_config.yaml tfs_posterior.h5

# ---------------------------------------------------------------------------
# 7. Predict theta
# ---------------------------------------------------------------------------
# Predict operator occupancy θ as a function of titrant concentration for
# every genotype in the training data.
echo ">>> Predict theta"
tfs-predict-theta tfs_configure_config.yaml tfs_posterior.h5

# ---------------------------------------------------------------------------
# 8. Predict growth
# ---------------------------------------------------------------------------
# Predict ln(CFU) with posterior uncertainty for every training observation.
echo ">>> Predict growth"
tfs-predict-growth tfs_configure_config.yaml tfs_posterior.h5 --num_marginal_samples=500

# ---------------------------------------------------------------------------
# 9. Summarise fit
# ---------------------------------------------------------------------------
# Collects all outputs, computes statistics, and writes diagnostic PDFs and
# CSVs to the summary/ subdirectory.
echo ">>> Summarise fit"
tfs-summarize-fit .

cd ..
