"""The staged level-offset MAP on a real (small) model, end to end."""
import os

import numpy as np
import pytest

from tfscreen.tfmodel.scripts.configure_model_cli import configure_model
from tfscreen.tfmodel.scripts.fit_model_cli import fit_model
from tfscreen.tfmodel.scripts.sample_posterior_cli import sample_posterior

OFF = "sample_offset_offset_auto_loc"


def _keys(path):
    with np.load(path) as z:
        return set(z.files)


@pytest.mark.slow
def test_staged_map_smoke(tmp_path, growth_smoke_csv, library_smoke_yaml):
    cfg = str(tmp_path / "cfg")
    configure_model(growth_df=growth_smoke_csv, library_config=library_smoke_yaml,
                    theta_model="hill_relative", growth_shares_replicates=True,
                    out_prefix=cfg, skip_model_stats=True)
    out = str(tmp_path / "fit")
    fit_model(f"{cfg}_config.yaml", seed=1, analysis_method="map",
              max_num_epochs=2, convergence_window_steps=20, out_prefix=out)

    stage1, stage2, final = (_keys(f"{out}_stage1_params.npz"),
                             _keys(f"{out}_stage2_params.npz"),
                             _keys(f"{out}_params.npz"))
    # stage 1 holds the offsets (and their learned SD); stage 2 fits only them
    assert OFF not in stage1 and "sample_offset_sigma_auto_loc" not in stage1
    assert stage2 == {OFF}
    assert OFF in final and stage1 <= final
    for stage in ("stage1", "stage2"):
        assert os.path.exists(f"{out}_{stage}_convergence.csv")

    # the final checkpoint is the full model's
    sample_posterior(f"{cfg}_config.yaml", f"{out}_checkpoint.pkl",
                     out_prefix=str(tmp_path / "post"),
                     num_posterior_samples=10, skip_growth_observations=True)
    assert os.path.exists(tmp_path / "post.h5")

    # rerun with the final outputs gone: stages 1 and 2 are reused
    for f in ("checkpoint.pkl", "params.npz"):
        os.remove(f"{out}_{f}")
    mtime = os.path.getmtime(f"{out}_stage1_params.npz")
    fit_model(f"{cfg}_config.yaml", seed=1, analysis_method="map",
              max_num_epochs=2, convergence_window_steps=20, out_prefix=out)
    assert os.path.getmtime(f"{out}_stage1_params.npz") == mtime
    assert os.path.exists(f"{out}_params.npz")
