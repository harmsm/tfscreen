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
# binding curves, and ground-truth parameter CSVs as run_dir/tfs_sim_*. It
# also writes the experiment in the raw formats a lab's data come in: one
# counts file per tube (tfs_sim_counts/) and the tube table
# (tfs_sim_tubes.csv).
echo ">>> Simulate library"
tfs-simulate "${config_file}" --out_prefix "${run_dir}/tfs_sim" --seed "${seed}"

# The simulate config also describes the library (genetics keys +
# library_mixture), so tfs-configure-model reads the same file.  Resolve it to
# an absolute path before changing directory.
library_config="$(cd "$(dirname "${config_file}")" && pwd)/$(basename "${config_file}")"

cd "${run_dir}"

# ---------------------------------------------------------------------------
# 1b. Process the raw counts, exactly as for real data
# ---------------------------------------------------------------------------
# The growth table the model reads comes from the per-tube count files and the
# tube table, through the same command a lab runs. (This config has no od600
# block, so the tube table carries each tube's total; with one, add the
# --od600_file, --od600_calibration_file and --tube_volume_mL that
# tfs-simulate prints.) The presplit table is still written directly by
# tfs-simulate.
echo ">>> Process counts"
tfs-process-counts tfs_sim_tubes.csv tfs_sim_counts \
    --out_prefix tfs_sim_processed_growth --no_verbose

# ---------------------------------------------------------------------------
# 2. Configure model
# ---------------------------------------------------------------------------
# tfs-configure-model validates the data, selects model components, and
# writes tfs_configure_config.yaml, the priors/guesses/library CSVs and a
# parameter census (tfs_configure_model_stats.*). Edit the flags here to
# change which model components are used. The data were simulated with
# hill_mut theta and epistasis, one plasmid per cell
# (transformation_poisson_lambda: 0) and an instant growth transition, so the
# model below matches the simulation; growth_likelihood counts with a level
# tube offset is the recommended default.
echo ">>> Configure model"
tfs-configure-model \
    --binding_df tfs_sim_binding.csv \
    --growth_df tfs_sim_processed_growth.csv \
    --presplit_df tfs_sim_presplit.csv \
    --condition_growth_model linear \
    --growth_transition_model instant \
    --ln_cfu0_model hierarchical \
    --dk_geno_model hierarchical_geno \
    --activity_model fixed \
    --theta_model hill_mut \
    --transformation_model single \
    --theta_rescale_model passthrough \
    --theta_growth_noise_model logit_normal \
    --theta_binding_noise_model zero \
    --growth_likelihood counts \
    --sample_offset_model level \
    --growth_noise_model zero \
    --library_config "${library_config}" \
    --growth_shares_replicates \
    --epistasis

# ---------------------------------------------------------------------------
# 3. Pre-fit calibration
# ---------------------------------------------------------------------------
# A MAP fit on a simplified model, using the genotypes that have binding
# data, calibrates each condition's growth baseline k and slope m. It writes
# them into tfs_configure_priors.csv and tfs_configure_guesses.csv in place
# (keeping .bak copies) and leaves its own diagnostics as tfs_prefit_*.
echo ">>> Pre-fit calibration"
tfs-prefit-calibration tfs_configure_config.yaml --seed "${seed}"

# ---------------------------------------------------------------------------
# 4. Fit model (SVI)
# ---------------------------------------------------------------------------
# Main hierarchical Bayesian inference. SVI starts from a MAP warm-up
# (tfs_fit_model_premap_*) and fits the component guide, an approximate
# posterior. Writes tfs_fit_model_checkpoint.pkl, _params.npz, _losses.txt
# and _convergence.csv. --analysis_method map gives the MAP point instead;
# tfs-sample-posterior then builds a Laplace posterior from it.
echo ">>> Fit model"
tfs-fit-model \
    tfs_configure_config.yaml \
    --seed "${seed}" \
    --analysis_method svi

# ---------------------------------------------------------------------------
# 5. Sample posterior
# ---------------------------------------------------------------------------
# Draw posterior samples from the fitted guide and write them to
# tfs_posterior.h5 for the steps below. --skip_growth_observations leaves out
# the stored per-observation growth sites (growth_pred, growth_obs). They are
# most of the file (several GB here at the default 10000 samples), and
# nothing below reads them: tfs-predict-growth and tfs-summarize-fit
# recompute growth from the parameter samples.
echo ">>> Sample posterior"
tfs-sample-posterior tfs_configure_config.yaml tfs_fit_model_checkpoint.pkl \
    --skip_growth_observations

# ---------------------------------------------------------------------------
# 6. Extract parameter estimates
# ---------------------------------------------------------------------------
# Summarize the posterior into per-parameter CSV files (tfs_params_*.csv,
# one column per quantile).
echo ">>> Extract parameter estimates"
tfs-extract-params tfs_configure_config.yaml tfs_posterior.h5

# ---------------------------------------------------------------------------
# 7. Predict theta
# ---------------------------------------------------------------------------
# Predict operator occupancy θ at every (genotype, titrant concentration) in
# the training data. Writes tfs_pred_theta.csv.
echo ">>> Predict theta"
tfs-predict-theta tfs_configure_config.yaml tfs_posterior.h5

# ---------------------------------------------------------------------------
# 8. Predict growth
# ---------------------------------------------------------------------------
# Predict ln(CFU) with posterior uncertainty for every training observation,
# from 500 of the posterior samples. Writes tfs_pred_growth.csv.
echo ">>> Predict growth"
tfs-predict-growth tfs_configure_config.yaml tfs_posterior.h5 --num_marginal_samples 500

# ---------------------------------------------------------------------------
# 9. Summarize fit
# ---------------------------------------------------------------------------
# Compares the predictions with the simulated truth (tfs_sim_genotype_theta.csv)
# and writes diagnostic PDFs and CSVs to the summary/ subdirectory.
echo ">>> Summarize fit"
tfs-summarize-fit .
