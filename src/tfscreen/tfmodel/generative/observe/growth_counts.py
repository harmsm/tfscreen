"""
Growth observer on read counts (``growth_likelihood: counts``).

Roadmap step 7 (planning/analysis-roadmap.md). Instead of observing
``ln_cfu`` (a pseudocount-adjusted log frequency plus the tube total), each
genotype's reads in each tube are observed directly:

    reads[g, s] ~ NegBin(mean = mu[g, s], var = mu (1 + phi) + mu^2 / r)
    log mu[g, s] = ln(depth_s) + ln_cfu_pred[g, s] - ln(N_s)

``depth_s`` is the tube's total reads (including ``__unknown__``), ``N_s``
the tube's total cells as supplied (``sample_ln_cfu``), and ``ln_cfu_pred``
the model's cells of genotype g in tube s (with any per-tube
``sample_offset``). So ``mu / depth`` is the predicted frequency. There is
no pseudocount: a zero is an observation.

The variance has two learned parts (study 0b, planning/studies/noise-anatomy/):

- ``phi`` (``{name}_phi``): variance proportional to the mean, as from a
  bottleneck upstream of the reads (template molecules into PCR, founder
  cells per tube). Real data: about 3-10.
- ``1/r`` (``{name}_inv_r``): a constant coefficient of variation, as from
  jackpotting that scales with abundance.

Both reduce to Poisson as they go to zero. The distribution is a
negative binomial with ``mean = mu`` and concentration
``1 / (phi / mu + 1 / r)``, evaluated in log space. Its log-pmf uses Loader's
algorithm (``CountNegativeBinomial``): numpyro's own negative binomial loses
several nats per observation to float32 cancellation at 10^4-10^6 reads.

The per-row observation weights, masking and genotype mini-batching are
identical to the ``ln_cfu`` observer (``observe/growth.py``).
"""

import jax
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from flax.struct import dataclass

from tfscreen.tfmodel.data_class import GrowthData


@dataclass(frozen=True)
class GrowthCountsObsPriors:
    """
    LogNormal priors (in log space: loc, scale) on the count observer's
    dispersion: ``phi`` (variance proportional to the mean) and ``inv_r``
    (1/r, the squared coefficient of variation that does not shrink with
    depth).
    """
    phi_loc: float
    phi_scale: float
    inv_r_loc: float
    inv_r_scale: float


def log_mean(data: GrowthData, ln_cfu_pred: jnp.ndarray) -> jnp.ndarray:
    """``log mu = ln(depth) + ln_cfu_pred - ln(N)`` per growth-tensor cell."""
    return data.ln_sample_reads + ln_cfu_pred - data.sample_ln_cfu


# --- A negative binomial log-pmf that stays accurate in float32 -----------
#
# numpyro's NegativeBinomial computes lgamma(k + c) - lgamma(c) - lgamma(k + 1)
# + c log p + k log q directly. At the read depths here (10^4-10^6 reads for
# spiked genotypes) those terms are ~10^7 and cancel to a few nats, so float32
# rounding alone costs several nats per observation. Loader's algorithm (C.
# Loader, "Fast and accurate computation of binomial probabilities", 2000;
# R's dbinom_raw / dnbinom_mu) writes the same probability through a deviance
# term (bd0) and Stirling-series errors (stirlerr), none of which cancel.

_HALF_LOG_2PI = 0.9189385332046727
_BD0_TERMS = 10


def _stirlerr(n):
    """lgamma(n + 1) - (n + 1/2) log n + n - log(2 pi)/2, for n > 0."""
    n = jnp.asarray(n)
    # Each branch sees only inputs where it is valid, so the unused one
    # cannot overflow and leak NaN into the gradient through jnp.where.
    big_n = jnp.maximum(n, 15.0)
    nn = big_n * big_n
    large = (1/12 - (1/360 - (1/1260 - (1/1680 - (1/1188)/nn)/nn)/nn)/nn)/big_n
    small_n = jnp.minimum(n, 15.0)
    small = (jax.scipy.special.gammaln(small_n + 1.0)
             - (small_n + 0.5)*jnp.log(small_n) + small_n - _HALF_LOG_2PI)
    return jnp.where(n > 15.0, large, small)


def _bd0(x, log_m):
    """
    Deviance ``x log(x / m) + m - x`` for x > 0, m = exp(log_m), without
    cancellation when x is close to m (series in v = (x - m) / (x + m)).
    """
    m = jnp.exp(log_m)
    direct = x*(jnp.log(x) - log_m) + m - x
    v = (x - m)/(x + m)
    near = jnp.abs(x - m) < 0.1*(x + m)
    v = jnp.where(near, v, 0.0)
    s = (x - m)*v
    ej = 2.0*x*v
    v2 = v*v
    for j in range(1, _BD0_TERMS + 1):
        ej = ej*v2
        s = s + ej/(2*j + 1)
    return jnp.where(near, s, direct)


def _nb_log_prob(k, log_mu, log_c):
    """
    log NB(k; mean mu, concentration c), with the variance mu + mu^2 / c.
    R's dnbinom_mu: log(c / (c + k)) + dbinom_raw(c, k + c, p, q) with
    p = c / (c + mu) and q = mu / (c + mu); k = 0 is c log p.
    """
    c = jnp.exp(log_c)
    log_p = -jax.nn.softplus(log_mu - log_c)
    log_q = -jax.nn.softplus(log_c - log_mu)

    zero = c*log_p

    k_safe = jnp.maximum(k, 1.0)
    n = k_safe + c
    log_n = jnp.log(n)
    lc = (_stirlerr(n) - _stirlerr(c) - _stirlerr(k_safe)
          - _bd0(c, log_n + log_p) - _bd0(k_safe, log_n + log_q))
    # log(2 pi c k / n), the binomial's normalizing term
    lf = 2.0*_HALF_LOG_2PI + log_c + jnp.log(k_safe) - log_n
    positive = (log_c - log_n) + lc - 0.5*lf

    return jnp.where(k > 0, positive, zero)


class CountNegativeBinomial(dist.Distribution):
    """
    Negative binomial on counts, parameterized by ``log_mu`` (log mean) and
    ``log_c`` (log concentration; variance ``mu + mu^2 / c``), with a
    float32-accurate log-pmf (see ``_nb_log_prob``). Sampling is Gamma-Poisson.
    """
    arg_constraints = {"log_mu": dist.constraints.real,
                       "log_c": dist.constraints.real}
    support = dist.constraints.nonnegative_integer

    def __init__(self, log_mu, log_c, *, validate_args=None):
        self.log_mu, self.log_c = jnp.broadcast_arrays(log_mu, log_c)
        super().__init__(batch_shape=jnp.shape(self.log_mu),
                         validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        c = jnp.exp(self.log_c)
        return dist.GammaPoisson(c, c*jnp.exp(-self.log_mu)).sample(key, sample_shape)

    def log_prob(self, value):
        return _nb_log_prob(value, self.log_mu, self.log_c)

    @property
    def mean(self):
        return jnp.exp(self.log_mu)

    @property
    def variance(self):
        mu = jnp.exp(self.log_mu)
        return mu + mu*mu*jnp.exp(-self.log_c)


# Floor on the concentration, c + exp(_LOG_C_FLOOR). Without it c ~ mu / phi
# underflows float32 for genotypes predicted near extinction (log mu below
# about -40, which the MAP reaches for genotypes with zero reads everywhere:
# P(0) keeps rising as mu falls), and the gradient and Hessian of the log-pmf
# go to NaN (count-likelihood grid, 2026-09-27). It changes only rows whose
# expected reads are below ~phi * 1e-13, where P(0) is 1 either way and a
# nonzero count still pushes mu up.
_LOG_C_FLOOR = -30.0


def count_distribution(log_mu, phi, inv_r):
    """
    Negative binomial with mean ``exp(log_mu)`` and variance
    ``mu (1 + phi) + mu^2 inv_r``: concentration ``1 / (phi / mu + inv_r)``
    (plus a floor of ``exp(_LOG_C_FLOOR)``), in log space so that very small
    means do not underflow.
    """
    log_c = -jnp.logaddexp(jnp.log(phi) - log_mu, jnp.log(inv_r))
    log_c = jnp.logaddexp(log_c, _LOG_C_FLOOR)
    return CountNegativeBinomial(log_mu, log_c)


def observe(name: str,
            data: GrowthData,
            ln_cfu_pred: jnp.ndarray,
            sigma_k: jnp.ndarray = 0.0,
            *,
            priors: GrowthCountsObsPriors):
    """
    Observation site for the growth read counts.

    Parameters
    ----------
    name : str
        Prefix for the sample sites (``{name}_phi``, ``{name}_inv_r``,
        ``{name}_obs``).
    data : GrowthData
        Needs ``counts``, ``ln_sample_reads`` and ``sample_ln_cfu`` (built by
        ``ModelOrchestrator`` when ``growth_likelihood='counts'``) besides
        the usual mask, sizes and mini-batch scale vector.
    ln_cfu_pred : jnp.ndarray
        The model's ln cells of each genotype in each tube.
    sigma_k : jnp.ndarray, optional
        Accepted for signature compatibility with the ``ln_cfu`` observer and
        ignored: ``growth_noise`` must be ``zero`` with this observer (the
        orchestrator refuses anything else), because extra per-row noise is
        what ``phi`` and ``inv_r`` describe.
    priors : GrowthCountsObsPriors
    """
    phi = pyro.sample(f"{name}_phi",
                      dist.LogNormal(priors.phi_loc, priors.phi_scale))
    inv_r = pyro.sample(f"{name}_inv_r",
                        dist.LogNormal(priors.inv_r_loc, priors.inv_r_scale))

    log_mu = log_mean(data, ln_cfu_pred)

    with pyro.plate(f"{name}_replicate", size=data.num_replicate, dim=-7):
        with pyro.plate(f"{name}_time", size=data.num_time, dim=-6):
            with pyro.plate(f"{name}_condition_pre", size=data.num_condition_pre, dim=-5):
                with pyro.plate(f"{name}_condition_sel", size=data.num_condition_sel, dim=-4):
                    with pyro.plate(f"{name}_titrant_name", size=data.num_titrant_name, dim=-3):
                        with pyro.plate(f"{name}_titrant_conc", size=data.num_titrant_conc, dim=-2):
                            with pyro.plate("shared_genotype_plate", size=data.batch_size, dim=-1):
                                with pyro.handlers.scale(scale=data.scale_vector):
                                    with pyro.handlers.mask(mask=data.good_mask):
                                        pyro.sample(f"{name}_obs",
                                                    count_distribution(log_mu, phi, inv_r),
                                                    obs=data.counts)


def guide(name: str,
          data: GrowthData,
          ln_cfu_pred: jnp.ndarray,
          sigma_k: jnp.ndarray = 0.0,
          *,
          priors: GrowthCountsObsPriors):
    """LogNormal guides for ``phi`` and ``inv_r``, started at their prior
    medians."""
    for site, loc in (("phi", priors.phi_loc), ("inv_r", priors.inv_r_loc)):
        site_loc = pyro.param(f"{name}_{site}_loc", jnp.asarray(loc))
        site_scale = pyro.param(f"{name}_{site}_scale", jnp.array(0.1),
                                constraint=dist.constraints.positive)
        pyro.sample(f"{name}_{site}", dist.LogNormal(site_loc, site_scale))


def get_hyperparameters():
    """
    Default priors. ``phi``: median 5, 95% interval about 0.7-35 (study 0b
    found 3-10 on real data; a simulation without a bottleneck has about
    0). ``inv_r``: median 0.01 (a 10% coefficient-of-variation floor), 95%
    interval about 5e-4 to 0.2.
    """
    return {"phi_loc": float(jnp.log(5.0)), "phi_scale": 1.0,
            "inv_r_loc": float(jnp.log(0.01)), "inv_r_scale": 1.5}


def get_priors():
    """Build GrowthCountsObsPriors from get_hyperparameters()."""
    return GrowthCountsObsPriors(**get_hyperparameters())
