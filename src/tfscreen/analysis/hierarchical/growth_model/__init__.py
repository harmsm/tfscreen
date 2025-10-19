"""
Models the growth of individual bacterial genotypes in a culture using a 
hierarchical model. 

The core model is:

ln_cfu = ln_cfu0 + (k_pre + dk_geno + m_pre*A*theta)*t_pre + (k_sel + dk_geno + m_sel*A*theta)*t_sel

This models our experimental set up. A given genotype begins with ln_cfu0. We do
an outgrowth for t_pre, then a selective growth for t_sel. The growth rate under
these both sets of conditions depends on the fractional occupancy of a
transcription factor operator (theta). We modulate transcription factor operator
occupancy by adding titrant (an effector like IPTG) to different samples. We
carefully ensure that all growth happens in log phase. We then measure ln_cfu
for each genotype over time.

The wildtype growth behavior under specific conditions is given by
k_x + m_x*(theta). k_x is the growth rate under condition x with 
theta = 0. m_x models the change in growth rate with increasing occupancy. Note
that condition 'x' is defined by things like antibiotic concentration, the
marker under control of the transcription factor, and media composition. It does
NOT depend on effector concentration, which is modeled as a separate effect 
captured by theta. 

In this model, global priors are treated heirarchically. There functional forms
are fixed, but their shape parameters are treated as hyperpriors. 

+ Each genotype in a ln_cfu0_block (replicate + library background) has its own
  ln_cfu0 sampled from a ln_cfu0_block-level prior.
+ Each genotype has its own A (activity), mapping operator occupancy to 
  overall transcriptional output. There is one value of A per genotype
  across all conditions. wildtype A is defined as 1.0. There is a single
  global prior for all A. 
+ Each genotype has its own dk_geno (pleiotropic effect of mutations on growth
  independent of occupancy). There is one value of dk_geno per genotype across
  all conditions/replicate/titrant. wildtype dk_geno is defined as 0.9. There is
  a single global prior all dk_geno. 
+ Each genotype at a given titrant concentration has its own theta (occupancy of
  the transcription factor operator). We use a single global prior for all
  theta. 
+ Each condition x (such as pheS-4CP or kanR+kan) has its own (k_x and m_x). 
  Each condition gets its own global prior that is shared across all replicates.
  Each replicate is assigned its own growth parameters. 
"""

from .model import growth_model