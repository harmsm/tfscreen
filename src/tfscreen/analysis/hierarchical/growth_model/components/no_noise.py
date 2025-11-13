
from flax.struct import (
    dataclass
)

@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding data needed to specify model priors.
    """

    pass


def define_model(name,fx_calc,priors):

    return fx_calc


def get_hyperparameters():
    """
    Get default values for the model hyperparameters.
    """

    parameters = {}

    return parameters


def get_guesses(name,data):
    """
    Get guesses for the model parameters. 
    """
    
    guesses = {}

    return guesses

def get_priors():
    return ModelPriors(**get_hyperparameters())