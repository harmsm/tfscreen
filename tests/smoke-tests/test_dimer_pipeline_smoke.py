"""
Pipeline smoke test for the thermo.dimer theta components.

configure (with variant flags set through --set_priors) -> MAP fit ->
Laplace posterior -> extract params -> predict theta, for training
genotypes and for an unmeasured combination, for PK and PddG.
"""

import os

import jax
import numpy as np
import pandas as pd
import pytest

from tfscreen.tfmodel.scripts.configure_model_cli import configure_model
from tfscreen.tfmodel.scripts.extract_params_cli import extract_params
from tfscreen.tfmodel.scripts.fit_model_cli import fit_model
from tfscreen.tfmodel.scripts.predict_theta_cli import predict_theta
from tfscreen.tfmodel.scripts.sample_posterior_cli import sample_posterior


@pytest.mark.slow
@pytest.mark.parametrize("theta_model, set_priors", [
    ("thermo.dimer.PK", None),
    ("thermo.dimer.PK", ["theta_monomers=1", "theta_effector_coupling=1"]),
    ("thermo.dimer.PddG", None),
])
def test_dimer_pipeline(tmpdir, growth_smoke_csv, binding_smoke_csv,
                        library_smoke_yaml, ddg_prior_csv_mwc, theta_model,
                        set_priors):
    # GrowthData keeps numpy arrays (the mutation COO indices) as static
    # jit metadata, so a second fit in one process with the same shapes
    # fails JAX's cache comparison. Each case starts from empty caches.
    jax.clear_caches()
    cfg = os.path.join(tmpdir, "cfg")
    configure_model(binding_df=binding_smoke_csv,
                    growth_df=growth_smoke_csv,
                    library_config=library_smoke_yaml,
                    theta_model=theta_model,
                    thermo_data=(ddg_prior_csv_mwc
                                 if theta_model.endswith("PddG") else None),
                    epistasis=True,
                    set_priors=set_priors,
                    allow_missing_spikes=True,
                    out_prefix=cfg)
    config_file = f"{cfg}_config.yaml"
    priors = pd.read_csv(f"{cfg}_priors.csv").set_index("parameter")["value"]
    if set_priors:
        assert float(priors["theta.theta_monomers"]) == 1.0

    out = os.path.join(tmpdir, "fit")
    fit_model(config_file=config_file, seed=1, analysis_method="map",
              max_num_epochs=3, out_prefix=out)
    sample_posterior(config_file=config_file,
                     checkpoint_file=f"{out}_checkpoint.pkl",
                     out_prefix=f"{out}_posterior",
                     num_posterior_samples=20, sampling_batch_size=20)
    posterior = f"{out}_posterior.h5"
    assert os.path.exists(posterior)

    extract_params(config_file=config_file, param_file=posterior,
                   out_prefix=out)
    d = pd.read_csv(f"{out}_d_ln_K_h_l.csv")
    assert len(d) > 0 and np.isfinite(d["median"] if "median" in d
                                      else d.filter(like="q").values).all()
    if set_priors:
        assert os.path.exists(f"{out}_d_ln_K_d_h.csv")
        assert os.path.exists(f"{out}_d_ln_K_h_ee.csv")
    if theta_model.endswith("PddG"):
        assert os.path.exists(f"{out}_ddG_H.csv")

    predict_theta(config_file, posterior, out_prefix=f"{out}_theta")
    pred = pd.read_csv(f"{out}_theta.csv")
    assert len(pred) > 0
    assert np.isfinite(pred["q0.5"]).all()
    assert pred["q0.5"].between(0, 1).all()

    # An unmeasured combination of two library mutations.
    lib = pd.read_csv(growth_smoke_csv)
    muts = sorted({m for g in lib["genotype"].unique() if g != "wt"
                   for m in g.split("/")})
    combo = "/".join(muts[:3])
    gfile = os.path.join(tmpdir, "genotypes.txt")
    with open(gfile, "w") as f:
        f.write(combo + "\n")
    predict_theta(config_file, posterior, out_prefix=f"{out}_unmeasured",
                  genotypes_file=gfile)
    un = pd.read_csv(f"{out}_unmeasured.csv")
    row = un[un["genotype"] == combo]
    assert len(row) > 0 and np.isfinite(row["q0.5"]).all()
