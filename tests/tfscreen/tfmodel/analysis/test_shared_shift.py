import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
from tfscreen.tfmodel.analysis.shared_shift import (
    apply_shared_shift,
    read_shift_table,
    row_shifts,
)
from tfscreen.tfmodel.analysis.extraction import (
    extract_theta_curves,
    extract_theta_epistasis,
)

_GENOTYPES = ["wt", "A10G", "C20D", "A10G/C20D"]


def _rows(genotypes=("wt", "mut"), concs=(0.0, 1.0)):
    return pd.DataFrame([{"genotype": g, "titrant_name": "iptg",
                          "titrant_conc": c}
                         for g in genotypes for c in concs])


def _shift(rows, value):
    out = rows.copy()
    out["shift"] = value
    return out


# --- the table ---

def test_read_shift_table_checks_columns_and_duplicates(tmp_path):
    rows = _shift(_rows(), 0.03)
    path = tmp_path / "shift.csv"
    rows.to_csv(path, index=False)
    assert len(read_shift_table(str(path))) == len(rows)
    with pytest.raises(ValueError, match="missing column"):
        read_shift_table(rows.drop(columns="shift"))
    with pytest.raises(ValueError, match="more than once"):
        read_shift_table(pd.concat([rows, rows.iloc[:1]]))


def test_row_shifts_align_and_zero_wt():
    rows = _rows()
    table = _shift(rows, 0.03).iloc[::-1]          # order must not matter
    table.loc[table["genotype"] == "wt", "shift"] = 99.0   # wt ignored
    out = row_shifts(rows, table)
    assert np.allclose(out[rows["genotype"].to_numpy() == "wt"], 0.0)
    assert np.allclose(out[rows["genotype"].to_numpy() == "mut"], 0.03)


def test_row_shifts_needs_every_non_wt_row():
    rows = _rows()
    table = _shift(rows[rows["genotype"] == "wt"], 0.03)   # wt only
    with pytest.raises(ValueError, match="no shift"):
        row_shifts(rows, table)
    # wt rows need no entry at all
    table = _shift(rows[rows["genotype"] != "wt"], 0.03)
    assert np.allclose(row_shifts(rows, table)[:2], 0.0)


# --- applying it ---

def test_one_shared_multiple_per_draw():
    rows = _rows(("wt", "a", "b"))
    table = _shift(rows, 0.0)
    table.loc[table["genotype"] == "a", "shift"] = 0.02
    table.loc[table["genotype"] == "b", "shift"] = -0.05
    samples = np.zeros((500, len(rows)))
    out = apply_shared_shift(samples, rows, table, seed=3)
    is_a = rows["genotype"].to_numpy() == "a"
    is_b = rows["genotype"].to_numpy() == "b"
    z_a = out[:, is_a] / 0.02
    z_b = out[:, is_b] / -0.05
    # every row of a draw moves by the same multiple of its own shift
    assert np.allclose(z_a, z_a[:, :1])
    assert np.allclose(z_a[:, 0], z_b[:, 0])
    assert np.allclose(out[:, rows["genotype"].to_numpy() == "wt"], 0.0)
    assert abs(np.std(z_a[:, 0]) - 1.0) < 0.1


def test_shift_widens_intervals_and_leaves_differences():
    rows = _rows(("wt", "a", "b"), concs=(1.0,))
    table = _shift(rows, 0.04)
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.01, size=(20000, len(rows)))
    out = apply_shared_shift(base, rows, table, seed=1)
    a = rows.index[rows["genotype"] == "a"][0]
    b = rows.index[rows["genotype"] == "b"][0]
    # each genotype's spread grows in quadrature
    assert np.std(out[:, a]) == pytest.approx(np.hypot(0.01, 0.04), rel=0.03)
    # a difference between two shifted genotypes keeps the original spread
    assert np.std(out[:, a] - out[:, b]) == pytest.approx(
        np.std(base[:, a] - base[:, b]), rel=1e-6)


def test_apply_refuses_a_single_draw_and_bad_shapes():
    rows = _rows()
    table = _shift(rows, 0.03)
    with pytest.raises(ValueError, match="MAP point"):
        apply_shared_shift(np.zeros((1, len(rows))), rows, table)
    with pytest.raises(ValueError, match="shape"):
        apply_shared_shift(np.zeros((5, len(rows) + 1)), rows, table)


def test_seed_makes_it_reproducible():
    rows = _rows()
    table = _shift(rows, 0.03)
    s = np.zeros((50, len(rows)))
    assert np.array_equal(apply_shared_shift(s, rows, table, seed=7),
                          apply_shared_shift(s, rows, table, seed=7))
    assert not np.array_equal(apply_shared_shift(s, rows, table, seed=7),
                              apply_shared_shift(s, rows, table, seed=8))


# --- through the extraction functions (the real hill_geno component) ---

def _model(genotypes, concs=(1.0,)):
    rows = [{"genotype": g, "titrant_name": "iptg", "titrant_conc": float(c),
             "map_theta_group": i}
            for i, g in enumerate(genotypes) for c in concs]
    model = MagicMock(spec=ModelOrchestrator)
    model._theta = "hill_geno"
    tm = MagicMock()
    tm.df = pd.DataFrame(rows)
    model.training_tm = tm
    model.growth_tm = tm
    return model


def _flat(theta, S):
    theta = np.tile(np.asarray(theta, dtype=float), (S, 1))
    k = theta.shape[1]
    return {"theta_hill_n": np.ones((S, k)),
            "theta_log_hill_K": np.zeros((S, k)),
            "theta_theta_high": theta.copy(),
            "theta_theta_low": theta.copy()}


def test_extract_theta_curves_applies_the_shift():
    model = _model(["wt", "mut"])
    post = _flat([0.6, 0.4], S=4000)
    rows = _rows(("wt", "mut"), concs=(1.0,))
    plain = extract_theta_curves(model, post, num_samples=None)
    shifted = extract_theta_curves(model, post, num_samples=None,
                                   shared_shift=_shift(rows, 0.05),
                                   shift_seed=2)
    wt = shifted["genotype"] == "wt"
    assert np.allclose(shifted.loc[wt, "q0.975"], plain.loc[wt, "q0.975"])
    width = (shifted.loc[~wt, "q0.975"] - shifted.loc[~wt, "q0.025"]).iloc[0]
    assert width == pytest.approx(2 * 1.96 * 0.05, rel=0.05)
    assert shifted.loc[~wt, "q0.5"].iloc[0] == pytest.approx(0.4, abs=0.005)


def test_epistasis_keeps_one_copy_of_a_uniform_shift():
    model = _model(_GENOTYPES)
    post = _flat([0.5, 0.4, 0.3, 0.25], S=4000)
    rows = _rows(_GENOTYPES, concs=(1.0,))
    plain = extract_theta_epistasis(model, post, scale="add")
    shifted = extract_theta_epistasis(model, post, scale="add",
                                      shared_shift=_shift(rows, 0.05),
                                      shift_seed=4)
    # ep = X_ab - X_a - X_b + X_wt: the shift enters s - s - s + 0 = -s
    assert plain["q0.975"].iloc[0] == pytest.approx(plain["q0.025"].iloc[0])
    width = shifted["q0.975"].iloc[0] - shifted["q0.025"].iloc[0]
    assert width == pytest.approx(2 * 1.96 * 0.05, rel=0.05)
    assert shifted["q0.5"].iloc[0] == pytest.approx(plain["q0.5"].iloc[0],
                                                    abs=0.005)


# --- the CLI guard ---

def test_predict_theta_refuses_shift_for_unmeasured_genotypes(tmp_path):
    from tfscreen.tfmodel.scripts.predict_theta_cli import predict_theta
    orch = MagicMock()
    orch._theta = "hill_geno"
    orch.training_tm.df = _rows(("wt", "A1B"))
    gfile = tmp_path / "g.txt"
    gfile.write_text("C2D\n")
    with patch("tfscreen.tfmodel.scripts.predict_theta_cli.read_configuration",
               return_value=(orch, {})), \
         patch("tfscreen.tfmodel.scripts.predict_theta_cli.resolve_param_file",
               side_effect=lambda pf, o, op: pf):
        with pytest.raises(ValueError, match="training genotypes only"):
            predict_theta("cfg.yaml", "post.h5", genotypes_file=str(gfile),
                          shared_shift=str(tmp_path / "shift.csv"),
                          out_prefix=str(tmp_path / "out"))
