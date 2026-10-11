"""
Learned parameters that the model itself registers with ``pyro.param``.

A few components register point-estimated parameters inside the *model*, not
only in a guide: the thermodynamic PnnC structure MLP weights
(``theta/thermo/nn.py``) and the per-structure ddG trust scale sigma_s of
PddG and PnnC (``theta/thermo/prior.py``). They are empirical-Bayes
estimates: the optimizer fits them alongside the guide, and they live in the
same param store (``svi.get_params`` constrained, the optimizer state
unconstrained), keyed by their bare names.

Any code that runs the model after the fit (posterior forward passes, the
Laplace Hessian, predictions, init hand-offs) must give the model these
learned values. Without them every ``pyro.param`` falls back to its init
value (NN = 0, sigma_s = 1), and the output no longer matches the fit.

``model_param_constraints`` finds the model's param sites by tracing the
model; ``model_params`` picks their values out of any param dict;
``hold_model_params`` wraps a model so they are fixed constants. The Laplace
treats them as fixed (they are not part of the Hessian), and posterior files
carry them in the ``model_params`` group (``write_model_params`` /
``read_model_params``). A model without model-side params gets an empty dict
everywhere and runs unchanged.
"""

import numpy as np
import jax.numpy as jnp
from numpyro import handlers
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to

# HDF5 group of a posterior file that holds the model-side params.
MODEL_PARAMS_GROUP = "model_params"


def model_param_constraints(model, priors, data, seed=0):
    """
    ``{name: constraint}`` for every ``pyro.param`` site the model registers.

    Parameters
    ----------
    model : callable
        Numpyro model taking ``priors=`` and ``data=``.
    priors, data
        Passed to the model.
    seed : int, optional
        PRNG seed for the trace.
    """
    from tfscreen.tfmodel.inference.initialization import structure_trace

    # a structure trace: a prior draw the observation refuses must not stop it
    tr = structure_trace(model, seed=seed, priors=priors, data=data)
    return {name: site["kwargs"].get("constraint", constraints.real)
            for name, site in tr.items() if site["type"] == "param"}


def model_params(params, param_constraints, unconstrained=False):
    """
    The model-side params found in ``params``, in constrained space.

    Parameters
    ----------
    params : mapping
        Any param dict keyed by param name: ``svi.get_params`` (constrained),
        the optimizer state ``svi.optim.get_params`` (unconstrained), a MAP
        ``_params.npz``, or an h5py group.
    param_constraints : dict
        ``model_param_constraints`` of the model.
    unconstrained : bool, optional
        True if ``params`` holds unconstrained values (an optimizer state);
        they are mapped through each param's constraint.

    Returns
    -------
    dict
        ``{name: jnp array}``, constrained. Empty if the model has no
        model-side params or ``params`` holds none of them.
    """
    out = {}
    for name, constraint in param_constraints.items():
        if name not in params:
            continue
        value = jnp.asarray(np.asarray(params[name]))
        out[name] = biject_to(constraint)(value) if unconstrained else value
    return out


def hold_model_params(model, params):
    """
    ``model`` with its model-side params fixed at ``params``.

    The values are substituted and the param sites are hidden from outer
    handlers, so neither an outer ``substitute`` (``Predictive``,
    ``potential_energy``) nor an SVI param store can replace them. Returns
    ``model`` itself when ``params`` is empty.
    """
    if not params:
        return model
    params = {k: jnp.asarray(v) for k, v in params.items()}
    return handlers.block(handlers.substitute(model, data=params),
                          hide=list(params))


def write_model_params(hf, params):
    """Write ``params`` to the ``model_params`` group of an open h5 file."""
    if not params:
        return
    group = hf.create_group(MODEL_PARAMS_GROUP)
    for name, value in params.items():
        group.create_dataset(name, data=np.asarray(value))


def read_model_params(posteriors, names=()):
    """
    The model-side params stored with a posterior.

    A posterior file (``tfs-sample-posterior``) keeps them in its
    ``model_params`` group. A raw MAP ``_params.npz`` keeps them as top-level
    arrays under their own names; pass the model's param names (``names``) to
    pick those up.

    Returns
    -------
    dict
        ``{name: numpy array}``; empty if none are stored.
    """
    if MODEL_PARAMS_GROUP in posteriors:
        group = posteriors[MODEL_PARAMS_GROUP]
        return {k: np.asarray(group[k]) for k in group.keys()}
    return {k: np.asarray(posteriors[k]) for k in names if k in posteriors}


def sample_keys(posteriors):
    """The keys of a posterior that hold samples (not the params group)."""
    return [k for k in posteriors.keys() if k != MODEL_PARAMS_GROUP]
