"""
Toy four-state thermodynamic TF model for teaching / interactive exploration.

A self-contained (NumPy/SciPy/pandas only) simulator of a monomer TF whose two
conformations (H, L) bind DNA and effector respectively. Activity is fixed at 1
(pure repressor) and effector is the sole titrant. Mutations perturb per-state
stabilities with optional in-state pairwise epistasis, and the builder emits a
long-form table ready for ``tfs-cat-response`` / ``tfs-extract-epistasis``.

See the module docstrings in ``core`` and ``genotypes`` for the model math.
"""

from .core import fraction_bound, solve_species  # noqa: F401
from .genotypes import (  # noqa: F401
    ThermoModel,
    MutationEffects,
    enumerate_genotypes,
    build_titration_df,
    parse_genotype,
    STATES,
)
from .sampling import sample_effects  # noqa: F401
from .basis import (  # noqa: F401
    free_ensemble,
    basis_curves,
    epistasis_coeffs,
    predict_epistasis,
    exact_epistasis,
    classify_shape,
    resolvable_logit,
    logit_ci,
    measurement_window,
    plot_basis,
    plot_epistasis_decomposition,
    plot_measurement_window,
    FREE_STATES,
)
