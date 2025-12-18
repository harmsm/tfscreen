
from .components import growth_independent 
from .components import growth_hierarchical
from .components import growth_fixed

from .components import ln_cfu0

from .components import dk_geno_fixed 
from .components import dk_geno_hierarchical

from .components import activity_fixed
from .components import activity_hierarchical 
from .components import activity_horseshoe 

from .components import theta_cat 
from .components import theta_hill 
from .components import transformation_congression
from .components import transformation_single 

from .components import no_noise 
from .components import beta_noise 

from .observe import binding 
from .observe import growth 

model_registry = {
    "condition_growth":{
        "independent":growth_independent,
        "hierarchical":growth_hierarchical,
        "fixed":growth_fixed,
    },
    "ln_cfu0":{
        "hierarchical":ln_cfu0,
    },
    "dk_geno":{
        "fixed":dk_geno_fixed,
        "hierarchical":dk_geno_hierarchical,
    },
    "activity":{
        "fixed":activity_fixed,
        "hierarchical":activity_hierarchical,
        "horseshoe":activity_horseshoe,
    },
    "transformation":{
        "congression": transformation_congression,
        "single": transformation_single,
    },
    "theta":{
        "categorical":theta_cat,
        "hill":theta_hill,
    },
    "theta_growth_noise":{
        "none":no_noise,
        "beta":beta_noise,
    },
    "theta_binding_noise":{
        "none":no_noise,
        "beta":beta_noise,
    },
    "observe_binding":binding,
    "observe_growth":growth,
}
