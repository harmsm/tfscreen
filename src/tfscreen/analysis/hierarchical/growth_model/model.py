
import jax.numpy as jnp
import numpyro as pyro
from numpyro.handlers import mask
import numpyro.distributions as dist


def growth_model(data,priors,fix_A=False,fix_dk_geno=False):
    """
    Model growth rates of bacteria in culture.
    """

    # -------------------------------------------------------------------------
    # ln[cfu0]
    # -------------------------------------------------------------------------
    
    with pyro.plate("ln_cfu0_blocks", data.num_ln_cfu0_block):
        ln_cfu0_hyper_locs = pyro.sample(
            "ln_cfu0_hyper_locs",
            dist.Normal(priors.ln_cfu0_hyper_loc_loc, priors.ln_cfu0_hyper_loc_scale)
        )
        ln_cfu0_hyper_scales = pyro.sample(
            "ln_cfu0_hyper_scales",
            dist.HalfCauchy(priors.ln_cfu0_hyper_scale_loc)
        )
        with pyro.plate("genotypes", data.num_geno):
            ln_cfu0_offsets = pyro.sample("ln_cfu0_offsets", dist.Normal(0, 1))

    # CORRECTED #1: Transpose offsets to align dimensions for broadcasting.
    ln_cfu0 = pyro.deterministic("ln_cfu0", ln_cfu0_hyper_locs[:, None] + ln_cfu0_offsets.T * ln_cfu0_hyper_scales[:, None])

    # -------------------------------------------------------------------------
    # A
    # -------------------------------------------------------------------------

    if not fix_A:
        A_hyper_loc = pyro.sample(
            "A_hyper_loc",
            dist.Normal(priors.A_hyper_loc_loc, priors.A_hyper_loc_scale)
        )
        A_hyper_scale = pyro.sample(
            "A_hyper_scale",
            dist.HalfCauchy(priors.A_hyper_scale_loc)
        )

        with pyro.plate("mutants_A", data.num_geno - 1):
            A_offset = pyro.sample("A_offset", dist.Normal(0, 1))
            A_mutants = A_hyper_loc + A_offset * A_hyper_scale

        A = jnp.empty(data.num_geno)
        A = A.at[data.wt_index].set(1.0)
        A = A.at[data.not_wt_mask].set(A_mutants)
        pyro.deterministic("A", A)
    else:
        A = jnp.ones(data.num_geno)

    # -------------------------------------------------------------------------
    # dk_geno
    # -------------------------------------------------------------------------

    if not fix_dk_geno:
        dk_geno_hyper_shift = pyro.sample(
            "dk_geno_shift",
            dist.Normal(priors.dk_geno_hyper_shift_loc, priors.dk_geno_hyper_shift_scale)
        )
        dk_geno_hyper_loc = pyro.sample(
            "dk_geno_hyper_loc",
            dist.Normal(priors.dk_geno_hyper_loc_loc, priors.dk_geno_hyper_loc_scale)
        )
        dk_geno_hyper_scale = pyro.sample(
            "dk_geno_hyper_scale",
            dist.HalfCauchy(priors.dk_geno_hyper_scale_loc)
        )

        with pyro.plate("mutants_dkgeno", data.num_geno - 1):
            dk_geno_offset = pyro.sample("dk_geno_offset", dist.Normal(0, 1))
            dk_geno_lognormal = jnp.exp(dk_geno_hyper_loc + dk_geno_offset * dk_geno_hyper_scale)
            dk_geno_mutants = dk_geno_hyper_shift - dk_geno_lognormal

        dk_geno = jnp.empty(data.num_geno)
        dk_geno = dk_geno.at[data.wt_index].set(0.0)
        dk_geno = dk_geno.at[data.not_wt_mask].set(dk_geno_mutants)
        pyro.deterministic("dk_geno", dk_geno)
    else:
        dk_geno = jnp.zeros(data.num_geno)

    # -------------------------------------------------------------------------
    # theta
    # -------------------------------------------------------------------------

    wt_theta = pyro.sample(
        "theta_wt",
        dist.TruncatedNormal(
            loc=priors.wt_theta_loc,
            scale=priors.wt_theta_scale,
            low=0.0,
            high=1.0
        ).to_event(1)
    )

    log_alpha_hyper_loc = pyro.sample("log_alpha_hyper_loc", dist.Normal(priors.log_alpha_hyper_loc_loc, priors.log_alpha_hyper_loc_scale))
    log_alpha_hyper_offset = pyro.sample("log_alpha_hyper_offset", dist.Normal(0, 1))
    theta_alpha = pyro.deterministic("theta_alpha", jnp.exp(log_alpha_hyper_loc + log_alpha_hyper_offset))
    
    log_beta_hyper_loc = pyro.sample("log_beta_hyper_loc", dist.Normal(priors.log_beta_hyper_loc_loc, priors.log_beta_hyper_loc_scale))
    log_beta_hyper_offset = pyro.sample("log_beta_hyper_offset", dist.Normal(0, 1))
    theta_beta = pyro.deterministic("theta_beta", jnp.exp(log_beta_hyper_loc + log_beta_hyper_offset))
    
    mutant_theta_prior = dist.Beta(theta_alpha, theta_beta)

    with pyro.plate("mutants_geno", data.num_geno - 1):
        with pyro.plate("titrant_concs", data.num_theta):
            mutant_thetas = pyro.sample("theta_mutants", mutant_theta_prior)
            
    thetas = jnp.empty((data.num_geno, data.num_theta))
    thetas = thetas.at[data.wt_index].set(wt_theta)
    thetas = thetas.at[data.not_wt_mask].set(mutant_thetas.T)
    pyro.deterministic("thetas", thetas)

    # -------------------------------------------------------------------------
    # Growth parameters
    # -------------------------------------------------------------------------

    with pyro.plate("replicates", data.num_rep):
        with pyro.plate("raw_conditions", data.num_raw_cond):
            growth_k = pyro.sample("growth_k_prior",
                                   dist.Normal(priors.growth_k_loc, priors.growth_k_scale))
            growth_m = pyro.sample("growth_m_prior",
                                   dist.Normal(priors.growth_m_loc, priors.growth_m_scale))
    
    pyro.deterministic("growth_k", growth_k)
    pyro.deterministic("growth_m", growth_m)

    # -------------------------------------------------------------------------
    # Calculate predicted outputs from model parameters
    # -------------------------------------------------------------------------

    geno_idx_array = jnp.arange(data.num_geno)
    rep_idx_array = jnp.arange(data.num_rep)[:, None, None, None]

    ln_cfu0_selected = ln_cfu0[data.ln_cfu0_block_map, geno_idx_array]

    k_pre = growth_k[rep_idx_array, data.cond_pre_map[None, :, None, None]]
    m_pre = growth_m[rep_idx_array, data.cond_pre_map[None, :, None, None]]
    k_sel = growth_k[rep_idx_array, data.cond_sel_map[None, :, None, None]]
    m_sel = growth_m[rep_idx_array, data.cond_sel_map[None, :, None, None]]
    
    g_idx = jnp.arange(data.num_geno)[None, None, :, None]
    th_idx = data.theta_map[None, :, None, None]
    theta_selected_bcast = thetas[g_idx, th_idx]

    growth_pre = k_pre + dk_geno[None, None, :, None] + m_pre * A[None, None, :, None] * theta_selected_bcast
    growth_sel = k_sel + dk_geno[None, None, :, None] + m_sel * A[None, None, :, None] * theta_selected_bcast

    predicted_ln_cfu = ln_cfu0_selected[..., None] + growth_pre * data.t_pre + growth_sel * data.t_sel

    # -------------------------------------------------------------------------
    # Observe data
    # -------------------------------------------------------------------------
    
    with mask(mask=data.good_mask):
        pyro.sample(
            "obs",
            dist.Normal(predicted_ln_cfu, data.ln_cfu_std).to_event(4),
            obs=data.ln_cfu_obs
        )