========
Analysis
========

The `tfscreen.analysis` module provides tools for extracting quantitative biochemical parameters from high-throughput screening data. The primary tool is a Bayesian hierarchical model that jointly fits growth and direct binding data.

Hierarchical Growth Model
=========================

The hierarchical model allows for robust parameter estimation by sharing information across variants and conditions. It can account for experimental noise, transformation effects, and pleiotropic growth defects.

Input Requirements
------------------

To run the analysis, you need two CSV or Excel files: one for growth data and one for optional direct binding data (e.g., from a separate assay).

Growth Data (``growth_df``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The growth spreadsheet must contain the following columns:

*   **genotype**: The name of the variant (e.g., "wt" or "A12V"). "wt" is used as the reference for activity and growth.
*   **ln_cfu** (or **cfu**): The log-transformed colony forming units (or raw CFU counts).
*   **ln_cfu_std** (or **cfu_std**): The standard deviation of the ``ln_cfu`` measurement.
*   **t_pre**: Time (in hours) of the pre-selection outgrowth phase.
*   **t_sel**: Time (in hours) of the selective growth phase.
*   **condition_pre**: Name of the pre-selection condition.
*   **condition_sel**: Name of the selection condition.
*   **titrant_name**: Name of the chemical effector (e.g., "IPTG").
*   **titrant_conc**: Concentration of the titrant.
*   **replicate**: Replicate number (integer).

Binding Data (``binding_df``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Direct binding data (optional) helps constrain the occupancy parameters (theta). It must contain:

*   **genotype**: The variant name (must match those in the growth data).
*   **titrant_name**: Name of the titrant.
*   **titrant_conc**: Concentration of the titrant.
*   **theta_obs**: Observed fractional occupancy (0.0 to 1.0).
*   **theta_std**: Standard deviation of the ``theta_obs`` measurement.

Step 1: Configuration (``tfs-configure-growth-analysis``)
---------------------------------------------------------

The first step is to prepare the model configuration. This script validates your input files, maps categorical labels to numerical indices, and sets up the model components.

**What it does:**

*   Generates a ``tfs_config.yaml`` file containing all model settings.
*   Creates ``tfs_priors.csv`` and ``tfs_guesses.csv`` for initial parameter values.

**Usage:**

.. code-block:: bash

    tfs-configure-growth-analysis --growth_df library_growth.csv --binding_df binding_data.csv --out_root my_analysis

Step 2: Inference (``tfs-growth-analysis``)
-------------------------------------------

This script performs the actual parameter estimation using the JAX-based inference engine. It supports both Stochastic Variational Inference (SVI) for full posterior estimation and Maximum A Posteriori (MAP) for faster point estimates.

**What it does:**

*   Loads the configuration and data into GPU/CPU memory via JAX.
*   Optimizes the model parameters.
*   Samples from the posterior distribution (if using SVI).
*   Saves the results to an HDF5 file (e.g., ``my_analysis_posterior.h5``) and creates checkpoints.

**Usage:**

.. code-block:: bash

    tfs-growth-analysis --config_file my_analysis_config.yaml --seed 42 --analysis_method svi

Step 3: Summarization (``tfs-summarize-posteriors``)
----------------------------------------------------

Finally, you can extract human-readable summaries from the posterior samples.

**What it does:**

*   Extracts physical parameters (activity, Hill coefficients, etc.) into spreadsheets.
*   Generates growth predictions to compare against original data.
*   Computes fractional occupancy curves.

**Usage:**

.. code-block:: bash

    tfs-summarize-posteriors --config_file my_analysis_config.yaml --posterior_file my_analysis_posterior.h5

**Key Outputs:**

*   ``my_analysis_parameters.csv``: Summary of all estimated biochemical parameters.
*   ``my_analysis_growth_pred.csv``: Model predictions vs. experimental observations.
*   ``my_analysis_theta_curves.csv``: Estimated occupancy curves for each genotype.

Model Components
================

The hierarchical model is modular, allowing you to select different sub-models for various physical processes. These are specified during the configuration step (``tfs-configure-growth-analysis``).

Condition Growth (``condition_growth_model``)
---------------------------------------------

Defines how the growth rate responds to changes in transcription factor occupancy across different conditions.

*   **linear** (default): Shared hierarchical prior for growth rates across conditions.
*   **linear_independent**: Each condition has an independent prior.
*   **linear_fixed**: Growth parameters are fixed to specified values.
*   **power**: Model growth using a power law relationship.
*   **saturation**: Model growth using a saturating (Michaelis-Menten-like) relationship.

Growth Transition (``growth_transition_model``)
-----------------------------------------------

Describes how genotypes transition between the pre-selection and selection growth phases.

*   **instant** (default): Genotypes immediately assume the new growth rate upon switching conditions.
*   **memory**: Accounts for a "lag" or memory effect during the transition.
*   **baranyi**: Uses the Baranyi-Roberts model to describe the transition into exponential growth.

Genotype Death/Pleiotropy (``dk_geno_model``)
---------------------------------------------

Models the baseline effect of a genotype on growth, independent of transcription factor occupancy.

*   **hierarchical** (default): Genotype-specific effects are sampled from a global prior.
*   **fixed**: No genotype-specific growth effects are modeled.
*   **mut_decomp**: Decomposes genotypes into individual mutation effects and pairwise interactions.

Activity (``activity_model``)
-----------------------------

Defines the "strength" of a genotype's effect on transcription given its occupancy of the binding site.

*   **horseshoe** (default): Uses a sparse horseshoe prior to identify variants with significant activity differences.
*   **hierarchical**: Uses a standard hierarchical prior for variant activity.
*   **fixed**: All variants are assumed to have the same activity as wildtype.
*   **mut_decomp**: Decomposes activity into individual mutation and pairwise interaction effects.

Occupancy (``theta_model``)
---------------------------

Describes the fractional occupancy (theta) of the binding site by the transcription factor.

*   **hill** (default): Models occupancy using the Hill equation (requires ``titrant_conc``).
*   **categorical**: Treats occupancy at each titrant concentration as an independent parameter.
*   **hill_mut**: Decomposes Hill parameters (Kd and n) into mutation-specific effects.

Transformation (``transformation_model``)
-----------------------------------------

Corrects for biases introduced during the bacterial transformation process.

*   **empirical** (default): Uses an empirical relationship to correct for "congression" where multiple plasmids enter a single cell.
*   **logit_norm**: Uses a logit-normal distribution for transformation correction.
*   **single**: Assumes each cell received exactly one plasmid.

Experimental Noise (``theta_growth_noise_model`` / ``theta_binding_noise_model``)
--------------------------------------------------------------------------------

Models stochastic noise in the measurement of fractional occupancy.

*   **zero** (default): No additional stochastic noise is modeled.
*   **beta**: Uses a Beta distribution to model measurements of fractional occupancy.
