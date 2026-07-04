
from .components.growth import linear as growth_hierarchical
from .components.growth import power as growth_power
from .components.growth import saturation as growth_saturation
from .components.growth_transition import instant as growth_transition_instant
from .components.growth_transition import memory as growth_transition_memory
from .components.growth_transition import baranyi as growth_transition_baranyi
from .components.growth_transition import baranyi_k as growth_transition_baranyi_k
from .components.growth_transition import baranyi_tau as growth_transition_baranyi_tau
from .components.growth_transition import two_pop as growth_transition_two_pop

from .components.ln_cfu0 import hierarchical as ln_cfu0
from .components.ln_cfu0 import hierarchical_factored as ln_cfu0_factored

from .components.dk_geno import fixed as dk_geno_fixed
from .components.dk_geno import hierarchical_geno as dk_geno_hierarchical
from .components.dk_geno import pinned as dk_geno_pinned

from .components.activity import fixed as activity_fixed
from .components.activity import hierarchical_geno as activity_hierarchical
from .components.activity import horseshoe_geno as activity_horseshoe
from .components.activity import hierarchical_mut as activity_mut_decomp
from .components.activity import horseshoe_mut as activity_horseshoe_mut

from .components.theta import _simple as theta_simple
from .components.theta import categorical_geno as theta_cat
from .components.theta import hill_geno as theta_hill
from .components.theta import hill_mut as theta_hill_mut
from .components.theta.thermo.O2_C4_K3_U0_a import PK as theta_lac_dimer_lnK_mut
from .components.theta.thermo.O2_C4_K3_U0_a import PnnC as theta_lac_dimer_lnK_nn_prior
from .components.theta.thermo.O2_C4_K3_U0_a import PddG as theta_lac_dimer_lnK_ddG_prior
from .components.theta.thermo.O2_C4_K3_U1_a import PK as theta_lac_dimer_unfolded_lnK_mut
from .components.theta.thermo.O2_C4_K3_U1_a import PnnC as theta_lac_dimer_unfolded_lnK_nn_prior
from .components.theta.thermo.O2_C4_K3_U1_a import PddG as theta_lac_dimer_unfolded_lnK_ddG_prior
from .components.theta.thermo.O2_C12_K5_U0_a import PK as theta_mwc_dimer_lnK_mut
from .components.theta.thermo.O2_C12_K5_U0_a import PnnC as theta_mwc_dimer_lnK_nn_prior
from .components.theta.thermo.O2_C12_K5_U0_a import PddG as theta_mwc_dimer_lnK_ddG_prior
from .components.theta.thermo.O2_C12_K5_U1_a import PK as theta_mwc_dimer_unfolded_lnK_mut
from .components.theta.thermo.O2_C12_K5_U1_a import PnnC as theta_mwc_dimer_unfolded_lnK_nn_prior
from .components.theta.thermo.O2_C12_K5_U1_a import PddG as theta_mwc_dimer_unfolded_lnK_ddG_prior
from .components.transformation import empirical as transformation_empirical
from .components.transformation import logit_norm as transformation_logit_norm
from .components.transformation import single as transformation_single

from .components.theta_rescale import passthrough as theta_rescale_passthrough
from .components.theta_rescale import logit as theta_rescale_logit

from .components.noise import zero as no_noise
from .components.noise import beta as beta_noise
from .components.noise import logit_normal as logit_normal_noise

from .components.growth_noise import zero as growth_noise_zero
from .components.growth_noise import normal_kt as growth_noise_normal_kt

from .components.sample_offset import zero as sample_offset_zero
from .components.sample_offset import normal as sample_offset_normal

from .observe import binding 
from .observe import growth 

model_registry = {
    "condition_growth":{
        "linear":growth_hierarchical,
        "power":growth_power,
        "saturation":growth_saturation,
    },
    "ln_cfu0":{
        "hierarchical":ln_cfu0,
        "hierarchical_factored":ln_cfu0_factored,
    },
    "dk_geno":{
        "fixed":dk_geno_fixed,
        "hierarchical_geno":dk_geno_hierarchical,
        "pinned":dk_geno_pinned,
    },
    "activity":{
        "fixed":activity_fixed,
        "hierarchical_geno":activity_hierarchical,
        "horseshoe_geno":activity_horseshoe,
        "hierarchical_mut":activity_mut_decomp,
        "horseshoe_mut":activity_horseshoe_mut,
    },
    "transformation":{
        "empirical": transformation_empirical,
        "logit_norm": transformation_logit_norm,
        "single": transformation_single,
    },
    "theta_rescale":{
        "passthrough": theta_rescale_passthrough,
        "logit": theta_rescale_logit,
    },
    "theta":{
        "_simple":theta_simple,
        "categorical_geno":theta_cat,
        "hill_geno":theta_hill,
        "hill_mut":theta_hill_mut,
        "thermo.O2_C4_K3_U0_a.PK":theta_lac_dimer_lnK_mut,
        "thermo.O2_C4_K3_U0_a.PnnC":theta_lac_dimer_lnK_nn_prior,
        "thermo.O2_C4_K3_U0_a.PddG":theta_lac_dimer_lnK_ddG_prior,
        "thermo.O2_C4_K3_U1_a.PK":theta_lac_dimer_unfolded_lnK_mut,
        "thermo.O2_C4_K3_U1_a.PnnC":theta_lac_dimer_unfolded_lnK_nn_prior,
        "thermo.O2_C4_K3_U1_a.PddG":theta_lac_dimer_unfolded_lnK_ddG_prior,
        "thermo.O2_C12_K5_U0_a.PK":theta_mwc_dimer_lnK_mut,
        "thermo.O2_C12_K5_U0_a.PnnC":theta_mwc_dimer_lnK_nn_prior,
        "thermo.O2_C12_K5_U0_a.PddG":theta_mwc_dimer_lnK_ddG_prior,
        "thermo.O2_C12_K5_U1_a.PK":theta_mwc_dimer_unfolded_lnK_mut,
        "thermo.O2_C12_K5_U1_a.PnnC":theta_mwc_dimer_unfolded_lnK_nn_prior,
        "thermo.O2_C12_K5_U1_a.PddG":theta_mwc_dimer_unfolded_lnK_ddG_prior,
    },
    "theta_growth_noise":{
        "zero":no_noise,
        "beta":beta_noise,
        "logit_normal":logit_normal_noise,
    },
    "theta_binding_noise":{
        "zero":no_noise,
        "beta":beta_noise,
    },
    "growth_noise":{
        "zero":growth_noise_zero,
        "normal_kt":growth_noise_normal_kt,
    },
    "sample_offset":{
        "zero":sample_offset_zero,
        "normal":sample_offset_normal,
    },
    "growth_transition":{
        "instant":growth_transition_instant,
        "memory":growth_transition_memory,
        "baranyi":growth_transition_baranyi,
        "baranyi_k":growth_transition_baranyi_k,
        "baranyi_tau":growth_transition_baranyi_tau,
        "two_pop":growth_transition_two_pop,
    },
    "observe_binding":binding,
    "observe_growth":growth,
}
