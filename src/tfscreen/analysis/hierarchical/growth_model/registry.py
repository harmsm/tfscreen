
from .components.growth import linear as growth_hierarchical
from .components.growth import linear_fixed as growth_fixed
from .components.growth import power as growth_power
from .components.growth import saturation as growth_saturation
from .components.growth_transition import instant as growth_transition_instant
from .components.growth_transition import memory as growth_transition_memory
from .components.growth_transition import baranyi as growth_transition_baranyi
from .components.growth_transition import baranyi_k as growth_transition_baranyi_k
from .components.growth_transition import baranyi_tau as growth_transition_baranyi_tau
from .components.growth_transition import two_pop as growth_transition_two_pop

from .components.ln_cfu0 import hierarchical as ln_cfu0

from .components.dk_geno import fixed as dk_geno_fixed
from .components.dk_geno import hierarchical as dk_geno_hierarchical
from .components.dk_geno import hierarchical_mut as dk_geno_mut_decomp

from .components.activity import fixed as activity_fixed
from .components.activity import hierarchical as activity_hierarchical
from .components.activity import horseshoe as activity_horseshoe
from .components.activity import hierarchical_mut as activity_mut_decomp

from .components.theta import categorical as theta_cat
from .components.theta import hill as theta_hill
from .components.theta import hill_mut as theta_hill_mut
from .components.theta import lac_dimer_mut as theta_lac_dimer_mut
from .components.theta.lac_dimer import lnK_mut as theta_lac_dimer_lnK_mut
from .components.theta import simple as theta_simple
from .components.transformation import empirical as transformation_empirical
from .components.transformation import logit_norm as transformation_logit_norm
from .components.transformation import single as transformation_single 

from .components.noise import zero as no_noise 
from .components.noise import beta as beta_noise 

from .observe import binding 
from .observe import growth 

model_registry = {
    "condition_growth":{
        "linear":growth_hierarchical,
        "linear_fixed":growth_fixed,
        "power":growth_power,
        "saturation":growth_saturation,
    },
    "ln_cfu0":{
        "hierarchical":ln_cfu0,
    },
    "dk_geno":{
        "fixed":dk_geno_fixed,
        "hierarchical":dk_geno_hierarchical,
        "hierarchical_mut":dk_geno_mut_decomp,
    },
    "activity":{
        "fixed":activity_fixed,
        "hierarchical":activity_hierarchical,
        "horseshoe":activity_horseshoe,
        "hierarchical_mut":activity_mut_decomp,
    },
    "transformation":{
        "empirical": transformation_empirical,
        "logit_norm": transformation_logit_norm,
        "single": transformation_single,
    },
    "theta":{
        "categorical":theta_cat,
        "hill":theta_hill,
        "hill_mut":theta_hill_mut,
        "lac_dimer_mut":theta_lac_dimer_mut,
        "lac_dimer_lnK_mut":theta_lac_dimer_lnK_mut,
        "simple":theta_simple,
    },
    "theta_growth_noise":{
        "zero":no_noise,
        "beta":beta_noise,
    },
    "theta_binding_noise":{
        "zero":no_noise,
        "beta":beta_noise,
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
