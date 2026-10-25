# Computational Methods (Rough Draft)

## Overview

We measured ln colony-forming units (ln CFU) as a function of IPTG concentration under two selective conditions. We aimed to use these data to infer the fractional occupancy (θ) of LacI bound to the operator upstream of both reporter genes. The full model comprised a three-level stack: a biophysical Hill model mapped binding parameters to fractional occupancy; a multiple-transformation correction adjusted that occupancy for plasmid copy number heterogeneity; and a linking function mapped corrected occupancy to observed ln CFU over time.

## Selective Conditions

We used two selection schemes to jointly constrain the occupancy inference. The first used *pheS* (4-chlorophenylalanine sensitivity) to measure **repression**: cells grew best when the LacI repressor was bound to the operator, blocking expression of the *pheS* susceptibility gene. The second used kanamycin resistance (*kanR*) to measure **induction**: cells grew best when the repressor was displaced from the operator, permitting KanR expression. Because these two conditions responded in opposite directions to occupancy, fitting them simultaneously allowed robust inference of θ across the dynamic range of repressor function.

## Data

We jointly fit two types of data. The first were high-throughput growth measurements: 209,000 mutant cycles covering approximately 210,000 distinct genotypes, with ln CFU measured across a structured set of conditions. Each genotype yielded up to 60 observations per replicate. We observed each genotype at 8 IPTG concentrations between 0 and 1 mM using each of the two reporters (kanR and pheS) under appropriate selection (kanamycin and 4CP). We also included 4 unselected controls (0 mM and 1 mM IPTG paired with each reporter in the absence of selection agent, allowing use to measure pre-selection growth rates). Together, this yielded 20 total conditions, each measured at 3 time points. We performed the full experiment in biological duplicate, yielding 120 observations per genotype.

The second data type was low-throughput *in vitro* binding measurements: fractional occupancy of selected LacI variants binding to the same operator sequence as we used *in vivo*, measured by fluorescence anisotropy in at least biological triplicate. We used the standard deviation across replicates the uncertainty on each θ measurement. We used these data in the model to anchor the Hill model parameters to physical binding affinities.

To assess inference quality, we spiked four additional genotypes with known binding properties into the library: the double mutants M42I/H74A, M42I/K84L, and H74A/K84L, and the triple mutant M42I/H74A/K84L. We did not provide their independently measured θ values to the model, allowing us to compare the inferred Hill parameters against ground-truth measurements as a post-hoc validation.

## Sequencing and CFU Computation

We derived the high-throughput growth data from deep sequencing of pooled library samples. We performed sequencing at the University of Oregon GC3F facility using 150 bp paired-end reads on the NovaSeq platform, yielding approximately 2.5 billion read pairs per biological replicate. We multiplexed samples using barcoding, and the sequencing facility demultiplexed them prior to analysis.

We performed read assignment using tfscreen, mapping reads to the *lacI* reference sequence via a direct sequencing strategy. We assigned a base position an "n" if its PHRED quality score was below 15. For positions in the center of the gene with overlapping forward and reverse reads, we called bases hierarchically: if reads agreed, we used their consensus; if both reads disagreed and neither was "n", we assigned "n" to both; if one read had a non-n call and the other was "n", we assigned the non-n call. We matched reads against the known library sequences, allowing up to one base mismatch with an expected sequence, and discarded reads equidistant from two or more library members as ambiguous.

We converted reads to per-genotype frequencies within each sample (defined by IPTG concentration, selection condition, biological replicate, and time point). We retained a genotype for analysis only if we observed it more than 15 times across all 60 conditions within a biological replicate, ensuring sufficient coverage for reliable frequency estimation. We added a pseudocount of one to all read counts for numerical stability. To convert frequencies to absolute CFU, we estimated the total CFU for each sample from OD$_{600}$ measurements taken in biological triplicate and averaged across replicates. We converted OD$_{600}$ to CFU using a second-order polynomial calibration curve derived from 25 carefully measured samples spanning the relevant density range, obtaining colony counts by serial dilution plating. We computed absolute per-genotype CFU as the product of the genotype frequency and the total CFU for that sample, and log-transformed the result to give ln CFU.

We propagated uncertainty on ln CFU from two independent sources: counting noise in the sequencing frequencies and variability in the total CFU estimate across triplicates. Because ln CFU = ln(frequency) + ln(sample CFU), the variance on ln CFU is the sum of the relative variances of its two components (delta method):

$$\sigma^2_{\ln \mathrm{CFU}} = \frac{\sigma^2_f}{f^2} + \frac{\sigma^2_C}{C^2}$$

The frequency variance $\sigma^2_f = f(1-f)/N$ followed from the binomial distribution, where $f$ was the estimated frequency and $N$ was the total read count for that sample. The CFU variance $\sigma^2_C$ was the squared standard deviation of the total CFU across the three biological replicates. We propagated this combined $\sigma^2_{\ln \mathrm{CFU}}$ into the likelihood as the scale parameter on the growth model distribution.

## Biophysical Model

We modeled fractional occupancy using a scaled Hill equation:

$$\theta(\mathrm{[IPTG]}) = \theta_\mathrm{min} + \Delta\theta \cdot \frac{[\mathrm{IPTG}]^n}{[\mathrm{IPTG}]^n + K^n}$$

where $\Delta\theta = \theta_\mathrm{max} - \theta_\mathrm{min}$ was the change in occupancy from zero to saturating IPTG, $K$ was the binding constant, and $n$ was the Hill coefficient. We inferred the four parameters ($K$, $n$, $\theta_\mathrm{min}$, $\theta_\mathrm{max}$) in the hierarchical Bayesian framework described below.

## Multiple-Transformation Correction

Before passing the Hill-model occupancy θ to the linking function, we applied a correction for plasmid copy number heterogeneity. In a high-throughput transformation, most cells take up a single plasmid copy, but a minority acquire two or more. We empirically characterized this distribution and found it followed a zero-truncated Poisson with parameter λ (predominantly one copy, rarely two, vanishingly rarely three).

Because LacI is a repressor, alleles with higher operator affinity will exhibit a dominant phenotype when present alongside a lower-affinity allele in the same cell. Multiple transformation thus confounded our inference: if a given genotype's frequency changed over the course of selection, it could reflect its own fitness or the fact that it was co-transformed with a dominant plasmid. This effect systematically biased our estimates of low occupancies ($\theta \rightarrow 0$), because the lower the value of $\theta$, the higher the probability that it was masked by a co-transformed variant with higher affinity.

We applied the correction as follows. At each iteration of the model, we built an empirical distribution of θ values across all genotypes in the current batch. For each genotype's θ, we computed the probability — given λ and this empirical distribution — that an observed cell carried a second, distinct plasmid whose θ masked the genotype's own fitness. Genotypes with high θ had a low probability of being masked; those with low θ were more susceptible. We computed the corrected θ by integrating over this sampling probability. [*Equation to be added.*]

We parameterized the correction with a single λ, using a prior informed by an independent empirical measurement of co-transformation rates. To measure λ, we transformed bacteria with an equimolar mixture of 8 plasmids that differed slightly in size but shared a single restriction site. We extracted plasmids from 107 individual colonies, digested them with the shared restriction enzyme, and resolved the products by gel electrophoresis; the number of bands per lane directly reported the plasmid copy number for that colony. We fit a zero-truncated Poisson to these counts by maximum likelihood, yielding λ = 0.4 (range 0.1–0.6). Zero-truncation was necessary because cells that failed to take up any plasmid were lethal under selection and therefore unobservable. We used the resulting estimate and its uncertainty as the prior on λ in the full model.

## Linking Function

We defined a *linking function* that predicted ln CFU as a function of time given the corrected θ. The core assumption--borne out experimentally--was that growth rate scaled linearly with θ. Cells experienced two sequential growth phases: a pre-selection phase (duration $t_\mathrm{pre}$) followed by a selective phase measured at multiple durations $t_\mathrm{sel}$. The full model was:

$$\ln \mathrm{CFU} = \ln \mathrm{CFU}_0 + \bigl(\theta \cdot m_\mathrm{pre} + b_\mathrm{pre} + \Delta k_\mathrm{geno}\bigr)\, (t_\mathrm{pre} + \tau) + \bigl(\theta \cdot m_\mathrm{sel} + b_\mathrm{sel} + \Delta k_\mathrm{geno}\bigr)\,(t_\mathrm{sel} - \tau)$$

The parameters were:

- $\ln \mathrm{CFU}_0$: initial colony count for a given genotype
- $\theta \in [0, 1]$: corrected fractional occupancy of the operator (0 = fully unoccupied, 1 = fully occupied)
- $m_\mathrm{pre}$, $b_\mathrm{pre}$: slope and intercept mapping θ to growth rate during pre-selection
- $m_\mathrm{sel}$, $b_\mathrm{sel}$: analogous slope and intercept during selection
- $\Delta k_\mathrm{geno}$: pleiotropic fitness effect of a mutation, independent of its effect on operator occupancy (for example, a destabilizing mutation may impose a metabolic cost from misfolded protein regardless of its DNA-binding phenotype)
- $\tau$: lag parameter describing the delay in growth rate adjustment when transitioning from pre-selection to selective conditions (see below)

## Lag Model

We evaluated three functional forms for the lag: an instantaneous transition model (no lag), a model with a sigmoidal transition integrated over time, and a memory model in which the lag depended on current occupancy. We selected the model based on posterior predictive checks, with visual inspection confirming a clear winner. The memory model performed substantially better than the alternatives [*quantitative comparison to be added*]:

$$\tau = \tau_0 + \frac{k_1}{k_2 + \theta}$$

When θ was large (repressor tightly bound, downstream gene expression suppressed), the denominator was large and the lag was shorter — the cell required little time to adjust because the gene product was already scarce. When θ was small (operator largely unoccupied, gene product abundant), the lag was longer because the existing product had to be diluted or turned over before growth under the new condition stabilized.

## Parameter Inference

### Likelihood

We evaluated two likelihood terms simultaneously. We fit the growth data with a Student-t likelihood, which provided robustness to occasional outliers; an equivalent Normal likelihood yielded similar results. We fit the in vitro fluorescence anisotropy binding data with a Normal likelihood. In both cases, we set the observation variance to the experimental uncertainty in the $\ln \mathrm{CFU}$ and $\theta$ values. The tight experimental uncertainties on the in vitro measurements produced a sharp likelihood that effectively pinned the Hill model parameters to physically measured binding affinities without requiring those measurements to function as priors.

### Prior structure

Model parameters fell into two categories. Parameters that appeared once across the entire dataset — the linking function growth terms ($m_\mathrm{pre}$, $b_\mathrm{pre}$, $m_\mathrm{sel}$, $b_\mathrm{sel}$), the lag terms ($\tau_0$, $k_1$, $k_2$), and the co-transformation rate (λ) — we assigned simple, weakly informative priors. We additionally informed the prior on λ using the externally measured co-transformation rates described above.

Parameters that varied across genotypes — $\Delta k_\mathrm{geno}$ and the Hill model parameters ($K$, $n$, $\theta_\mathrm{min}$, $\theta_\mathrm{max}$) — we assigned hierarchical priors, with separate hyperparameter distributions learned for single mutants and double mutants. This distinction reflected the experimental design: we spiked single mutants into the library at higher initial frequencies than double mutants, so the two classes were expected to have different $\ln \mathrm{CFU}_0$ distributions. For all hierarchical parameters, we drew per-genotype values from a Normal distribution whose location and scale were themselves treated as random variables: we placed a Normal hyperprior on the location and a HalfNormal hyperprior on the scale, and learned these hyperparameters jointly from all genotypes within each class.

We assigned wild type and the four reference genotypes (WT, M42I, H74A, K84L) — spiked in at substantially higher concentrations than the bulk library — individual simple priors on $\ln \mathrm{CFU}_0$ rather than drawing them from either hierarchical class.

### Fitting procedure

We performed MAP optimization using NumPyro's AutoDelta guide in two steps: first, a constrained fit in which we optimized only growth parameters (τ and the pre-selection slope) while holding all other parameters at simple initial values; then an unconstrained MAP over all parameters simultaneously. This initialization strategy provided stable starting points for the stochastic phase.

We then ran SVI using NumPyro's clipped Adam optimizer with an exponentially decaying step size schedule, starting at $10^{-3}$ and decaying to $10^{-6}$. Each SVI step used 2 ELBO particles and operated on a minibatch of 1,024 genotypes: 1,020 drawn from the library and the 4 reference genotypes included in every batch. Because we oversampled the reference genotypes relative to their true library frequency, we used JAX's built-in importance weighting to correct their contribution to the ELBO.

We declared convergence within a run when the mean ELBO changed by less than $10^{-6}$ over a sliding window of 100 epochs, sustained for at least 10 consecutive windows (patience counter). [*Total epoch counts to be added.*] We assessed convergence across runs by comparing posterior parameter estimates across multiple random seeds. [*Cross-seed convergence thresholds to be added.*]

### Outputs

After SVI, we extracted posterior distributions for each model parameter and derived point estimates and 95% credibility intervals. The primary outputs reported per genotype were the Hill model parameters: binding constant $K$, Hill coefficient $n$, unoccupied-state baseline $\theta_\mathrm{min}$, and bound-state baseline $\theta_\mathrm{max}$.

## Implementation

We implemented the model in Python using the standard scientific computing stack. We performed probabilistic modeling and inference with NumPyro and ran all calculations on NVIDIA A100 GPUs via the University of Oregon TALAPAS high-performance computing cluster. The implementation is available as the open-source package **tfscreen** at https://github.com/harmslab/tfscreen.
