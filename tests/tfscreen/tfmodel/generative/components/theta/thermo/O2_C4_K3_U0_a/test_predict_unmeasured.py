"""
Tests for lac_dimer lnK_mut.predict_unmeasured and lnK_nn_prior.predict_unmeasured.
"""

import pytest
import numpy as np
import pandas as pd

from tfscreen.tfmodel.generative.components.theta.thermo.O2_C4_K3_U0_a.PK import (
    predict_unmeasured as lnK_mut_predict,
)
from tfscreen.tfmodel.generative.components.theta.thermo.O2_C4_K3_U0_a.PnnC import (
    predict_unmeasured as lnK_nn_predict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MUT_LABELS  = ["M42I", "K84L"]
PAIR_LABELS = ["K84L/M42I"]

TITRANT_NAMES = ["IPTG"]

# Physical constants matching the lnK_mut defaults
TF_TOTAL = 6.5e-7
OP_TOTAL = 2.5e-8

# Biophysically plausible WT values (matching lnK_mut get_hyperparameters defaults)
WT_LN_K_OP = 23.0
WT_LN_K_HL = -9.0
WT_LN_K_E  = 33.4


def _make_titrant_df(concs=None):
    if concs is None:
        concs = [0.0, 1e-6, 1e-5, 1e-4]
    return pd.DataFrame({
        "titrant_name": ["IPTG"] * len(concs),
        "titrant_conc": concs,
    })


def _make_lnK_mut_posteriors(T, M, P=0, S=5, seed=0):
    """Build fake posterior dict matching lnK_mut parameter names."""
    rng = np.random.default_rng(seed)
    post = {}
    post["theta_ln_K_op_wt"] = np.full(S, WT_LN_K_OP, dtype=np.float32)
    post["theta_ln_K_HL_wt"] = np.full(S, WT_LN_K_HL, dtype=np.float32)
    post["theta_ln_K_E_wt"]  = np.full((S, T), WT_LN_K_E, dtype=np.float32)
    post["theta_d_ln_K_op"]  = rng.standard_normal((S, M)).astype(np.float32) * 0.5
    post["theta_d_ln_K_HL"]  = rng.standard_normal((S, M)).astype(np.float32) * 0.5
    post["theta_d_ln_K_E"]   = rng.standard_normal((S, T, M)).astype(np.float32) * 0.5
    if P > 0:
        post["theta_epi_ln_K_op"] = np.zeros((S, P), dtype=np.float32)
        post["theta_epi_ln_K_HL"] = np.zeros((S, P), dtype=np.float32)
        post["theta_epi_ln_K_E"]  = np.zeros((S, T, P), dtype=np.float32)
    return post


def _make_lnK_nn_posteriors(T, M, P=0, S=5, seed=0):
    """Build fake posterior dict matching lnK_nn_prior parameter names.

    KEY DIFFERENCE: d_ln_K_E is (S, M) not (S, T, M).
    """
    rng = np.random.default_rng(seed)
    post = {}
    post["theta_ln_K_op_wt"] = np.full(S, WT_LN_K_OP, dtype=np.float32)
    post["theta_ln_K_HL_wt"] = np.full(S, WT_LN_K_HL, dtype=np.float32)
    post["theta_ln_K_E_wt"]  = np.full((S, T), WT_LN_K_E, dtype=np.float32)
    post["theta_d_ln_K_op"]  = rng.standard_normal((S, M)).astype(np.float32) * 0.5
    post["theta_d_ln_K_HL"]  = rng.standard_normal((S, M)).astype(np.float32) * 0.5
    post["theta_d_ln_K_E"]   = rng.standard_normal((S, M)).astype(np.float32) * 0.5  # no T!
    if P > 0:
        post["theta_epi_ln_K_op"] = np.zeros((S, P), dtype=np.float32)
        post["theta_epi_ln_K_HL"] = np.zeros((S, P), dtype=np.float32)
        post["theta_epi_ln_K_E"]  = np.zeros((S, P), dtype=np.float32)  # no T!
    return post


def _common_kwargs(post, pair_labels=None):
    return dict(
        target_genotypes=["wt", "M42I", "K84L", "M42I/K84L"],
        titrant_names=TITRANT_NAMES,
        manual_titrant_df=_make_titrant_df(),
        mut_labels=MUT_LABELS,
        pair_labels=pair_labels or [],
        param_posteriors=post,
        q_to_get={"median": 0.5, "lower": 0.025, "upper": 0.975},
        tf_total=TF_TOTAL,
        op_total=OP_TOTAL,
    )


# ---------------------------------------------------------------------------
# lnK_mut
# ---------------------------------------------------------------------------

class TestLnKMutPredict:

    def test_output_shape_and_columns(self):
        post = _make_lnK_mut_posteriors(T=1, M=2)
        result = lnK_mut_predict(**_common_kwargs(post))
        n_genos = 4
        n_concs = 4
        assert len(result) == n_genos * n_concs
        for col in ["genotype", "titrant_name", "titrant_conc",
                    "median", "lower", "upper"]:
            assert col in result.columns

    def test_wt_theta_in_unit_interval(self):
        post = _make_lnK_mut_posteriors(T=1, M=2)
        result = lnK_mut_predict(**_common_kwargs(post))
        wt_rows = result[result["genotype"] == "wt"]
        assert (wt_rows["median"] >= 0).all()
        assert (wt_rows["median"] <= 1).all()
        assert not wt_rows["median"].isna().any()

    def test_unknown_mutation_is_nan(self):
        post = _make_lnK_mut_posteriors(T=1, M=2)
        kw = _common_kwargs(post)
        kw["target_genotypes"] = ["wt", "Z99Q"]
        result = lnK_mut_predict(**kw)
        wt_rows  = result[result["genotype"] == "wt"]
        bad_rows = result[result["genotype"] == "Z99Q"]
        assert not wt_rows["median"].isna().any()
        assert bad_rows["median"].isna().all()

    def test_zero_conc_handled(self):
        post = _make_lnK_mut_posteriors(T=1, M=2)
        kw = _common_kwargs(post)
        kw["manual_titrant_df"] = _make_titrant_df(concs=[0.0])
        result = lnK_mut_predict(**kw)
        assert np.isfinite(result["median"].values).all()

    def test_upper_ge_median_ge_lower(self):
        post = _make_lnK_mut_posteriors(T=1, M=2, S=50, seed=7)
        result = lnK_mut_predict(**_common_kwargs(post))
        assert (result["upper"] >= result["median"]).all()
        assert (result["median"] >= result["lower"]).all()

    def test_epistasis_shifts_double_mutant(self):
        T, M, P, S = 1, 2, 1, 30
        post = _make_lnK_mut_posteriors(T, M, P=P, S=S, seed=3)
        # Large epistasis on K_op for the pair
        post["theta_epi_ln_K_op"] = np.full((S, P), 5.0, dtype=np.float32)

        kw_no_epi = _common_kwargs(post, pair_labels=[])
        kw_no_epi["target_genotypes"] = ["M42I/K84L"]
        res_no_epi = lnK_mut_predict(**kw_no_epi)

        kw_epi = _common_kwargs(post, pair_labels=PAIR_LABELS)
        kw_epi["target_genotypes"] = ["M42I/K84L"]
        res_epi = lnK_mut_predict(**kw_epi)

        assert not np.allclose(res_no_epi["median"].values,
                               res_epi["median"].values, atol=1e-3)


# ---------------------------------------------------------------------------
# lnK_nn_prior — mirrors the lnK_mut tests but uses (S, M) d_ln_K_E
# ---------------------------------------------------------------------------

class TestLnKNnPriorPredict:

    def test_output_shape_and_columns(self):
        post = _make_lnK_nn_posteriors(T=1, M=2)
        result = lnK_nn_predict(**_common_kwargs(post))
        assert len(result) == 4 * 4
        for col in ["genotype", "titrant_name", "titrant_conc", "median"]:
            assert col in result.columns

    def test_wt_theta_in_unit_interval(self):
        post = _make_lnK_nn_posteriors(T=1, M=2)
        result = lnK_nn_predict(**_common_kwargs(post))
        wt_rows = result[result["genotype"] == "wt"]
        assert (wt_rows["median"] >= 0).all()
        assert (wt_rows["median"] <= 1).all()
        assert not wt_rows["median"].isna().any()

    def test_unknown_mutation_is_nan(self):
        post = _make_lnK_nn_posteriors(T=1, M=2)
        kw = _common_kwargs(post)
        kw["target_genotypes"] = ["wt", "BAD"]
        result = lnK_nn_predict(**kw)
        assert result.loc[result["genotype"] == "BAD", "median"].isna().all()
        assert not result.loc[result["genotype"] == "wt", "median"].isna().any()

    def test_multiple_titrants_broadcast_correctly(self):
        """With T=2 titrants, K_E assembly must broadcast d_ln_K_E (S,M) across T."""
        T, M, S = 2, 2, 10
        post = _make_lnK_nn_posteriors(T, M, S=S)
        kw = dict(
            target_genotypes=["wt", "M42I"],
            titrant_names=["IPTG", "TMAIPP"],
            manual_titrant_df=pd.DataFrame({
                "titrant_name": ["IPTG", "TMAIPP"],
                "titrant_conc": [1e-5, 1e-5],
            }),
            mut_labels=MUT_LABELS,
            pair_labels=[],
            param_posteriors=post,
            q_to_get={"median": 0.5},
            tf_total=TF_TOTAL,
            op_total=OP_TOTAL,
        )
        result = lnK_nn_predict(**kw)
        assert len(result) == 4   # 2 genotypes × 2 titrants
        assert not result["median"].isna().any()

    def test_scalar_d_KE_vs_titrant_d_KE(self):
        """lnK_nn_prior with d_E scalar should equal lnK_mut when d_E is replicated T times."""
        T, M, S = 2, 2, 5
        rng = np.random.default_rng(99)
        d_E_scalar = rng.standard_normal((S, M)).astype(np.float32)

        post_nn = _make_lnK_nn_posteriors(T, M, S=S)
        post_nn["theta_d_ln_K_E"] = d_E_scalar   # (S, M)

        post_mut = _make_lnK_mut_posteriors(T, M, S=S)
        # Copy WT from post_nn
        for k in ["theta_ln_K_op_wt", "theta_ln_K_HL_wt", "theta_ln_K_E_wt",
                  "theta_d_ln_K_op", "theta_d_ln_K_HL"]:
            post_mut[k] = post_nn[k].copy()
        # Replicate scalar d_E across T dimension
        post_mut["theta_d_ln_K_E"] = np.broadcast_to(
            d_E_scalar[:, None, :], (S, T, M)
        ).astype(np.float32)

        kw = dict(
            target_genotypes=["wt", "M42I", "K84L"],
            titrant_names=["IPTG", "TMAIPP"],
            manual_titrant_df=pd.DataFrame({
                "titrant_name": ["IPTG", "TMAIPP"],
                "titrant_conc": [1e-5, 1e-4],
            }),
            mut_labels=MUT_LABELS,
            pair_labels=[],
            q_to_get={"median": 0.5},
            tf_total=TF_TOTAL,
            op_total=OP_TOTAL,
        )
        res_nn  = lnK_nn_predict(param_posteriors=post_nn,  **kw)
        res_mut = lnK_mut_predict(param_posteriors=post_mut, **kw)

        np.testing.assert_allclose(
            res_nn["median"].values, res_mut["median"].values, atol=1e-5
        )
