"""
Convergence on a small real fit: configure -> prefit -> fit (pre-MAP + SVI,
and MAP) on the smoke data, run until the monitor stops them.
"""

import os

import dill
import pandas as pd
import pytest

from tfscreen.tfmodel.scripts.configure_model_cli import configure_model
from tfscreen.tfmodel.scripts.fit_model_cli import fit_model
from tfscreen.tfmodel.scripts.prefit_calibration_cli import (
    run_prefit_calibration,
)

# Small windows keep the smoke fits to about a minute.
_FAST = dict(convergence_window_steps=1000, patience=2,
             epoch_checkpoint_interval=None)


@pytest.fixture
def configured(tmpdir, growth_smoke_csv, binding_smoke_csv,
               library_smoke_yaml):
    out_prefix = os.path.join(tmpdir, "conv")
    configure_model(binding_smoke_csv,
                    growth_df=growth_smoke_csv,
                    library_config=library_smoke_yaml,
                    out_prefix=out_prefix,
                    skip_model_stats=True)
    return f"{out_prefix}_config.yaml", str(tmpdir)


def _windows(prefix):
    return pd.read_csv(f"{prefix}_convergence.csv")


@pytest.mark.slow
def test_prefit_then_svi_converges(configured):
    config_file, tmpdir = configured

    prefit = os.path.join(tmpdir, "conv_prefit")
    _, _, prefit_converged = run_prefit_calibration(
        config_file=config_file, seed=1, out_prefix=prefit,
        max_num_epochs=60000, hessian_chunk_size=8, **_FAST)
    prefit_windows = _windows(prefit)
    assert len(prefit_windows) >= 2
    if prefit_converged:
        assert prefit_windows["decision"].iloc[-1] == "converged"

    fit = os.path.join(tmpdir, "conv_fit")
    *_, converged = fit_model(config_file=config_file, seed=42,
                              out_prefix=fit, max_num_epochs=100000,
                                pre_map_num_epoch=5000, **_FAST)
    assert converged

    windows = _windows(fit)
    assert windows["decision"].iloc[-1] == "converged"
    assert (windows["decision"] == "cut").sum() == 3  # 1e-3 -> 1e-6
    last = windows.iloc[-1]
    assert last["step_size"] == pytest.approx(1e-6)
    assert not last["loss_improving"]
    assert not last["params_moving"]

    # the pre-MAP ran under the same monitor, in its own files
    assert os.path.exists(f"{fit}_premap_convergence.csv")

    with open(f"{fit}_checkpoint.pkl", "rb") as f:
        ckpt = dill.load(f)
    assert ckpt["convergence"]["converged"]
    assert ckpt["step_size"] == pytest.approx(1e-6)


@pytest.mark.slow
def test_svi_resume_after_convergence_stays_converged(configured):
    """Resuming a converged fit continues at the final step size and
    converges again within patience windows, rather than restarting the
    schedule or never stopping."""
    config_file, tmpdir = configured
    fit = os.path.join(tmpdir, "conv_fit")
    fit_model(config_file=config_file, seed=42, out_prefix=fit,
              max_num_epochs=100000, pre_map_num_epoch=0, **_FAST)
    n_before = len(_windows(fit))

    *_, converged = fit_model(config_file=config_file,
                                checkpoint_file=f"{fit}_checkpoint.pkl",
                                out_prefix=fit, max_num_epochs=20000,
                                **_FAST)
    windows = _windows(fit)
    assert converged
    resumed = windows.iloc[n_before:]
    assert list(resumed["step_size"]) == pytest.approx([1e-6] * len(resumed))
    assert len(resumed) <= 6


@pytest.mark.slow
def test_map_reports_honestly(configured):
    """MAP of the default hierarchical model has unbounded directions (the
    hyper-scale funnel); whatever it does, a stop must be backed by the last
    window and a cap must be reported as not converged."""
    config_file, tmpdir = configured
    fit = os.path.join(tmpdir, "conv_map")
    *_, converged = fit_model(config_file=config_file, seed=42,
                                analysis_method="map", out_prefix=fit,
                                max_num_epochs=20000, **_FAST)
    last = _windows(fit).iloc[-1]
    if converged:
        assert last["decision"] == "converged"
        assert not last["params_moving"]
    else:
        assert last["decision"] != "converged"
