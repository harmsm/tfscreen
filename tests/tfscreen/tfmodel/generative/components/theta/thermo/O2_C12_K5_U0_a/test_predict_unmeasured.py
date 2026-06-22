"""
Tests for mwc_dimer lnK_mut.predict_unmeasured and lnK_nn_prior.predict_unmeasured.
"""

import pytest
import numpy as np
import pandas as pd

from tfscreen.tfmodel.generative.components.theta.thermo.O2_C12_K5_U0_a.PK import (
    predict_unmeasured as lnK_mut_predict,
)
from tfscreen.tfmodel.generative.components.theta.thermo.O2_C12_K5_U0_a.PnnC import (
    predict_unmeasured as lnK_nn_predict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MUT_LABELS  = ["M42I", "K84L"]
PAIR_LABELS = ["K84L/M42I"]

TITRANT_NAMES = ["IPTG"]

# Physical constants (monomer units for tf_total, consistent with MWC convention)
TF_TOTAL = 6.5e-7
OP_TOTAL = 2.5e-8
CONC_SCALE = 1e-3   # concentrations in mM → M

# Biophysically plausible WT values (from lnK_mut get_hyperparameters defaults)
WT_LN_K_H_L =  1.84   # K_h_l ≈ 6.3
WT_LN_K_H_O = 19.86   # K_h_o ≈ 4.2e8
WT_LN_K_H_E = 10.93   # K_h_e ≈ 5.6e4
WT_LN_K_L_O = -2.30   # K_l_o ≈ 0.1
WT_LN_K_L_E = 13.54   # K_l_e ≈ 7.6e5


def _make_titrant_df(concs=None):
    if concs is None:
        concs = [0.0, 0.1, 1.0, 10.0]   # in mM
    return pd.DataFrame({
        "titrant_name": ["IPTG"] * len(concs),
        "titrant_conc": concs,
    })


def _make_lnK_mut_posteriors(T, M, P=0, S=5, seed=0):
    """Fake posterior dict for mwc_dimer lnK_mut."""
    rng = np.random.default_rng(seed)
    post = {}
    post["theta_ln_K_h_l_wt"] = np.full(S, WT_LN_K_H_L, dtype=np.float32)
    post["theta_ln_K_h_o_wt"] = np.full(S, WT_LN_K_H_O, dtype=np.float32)
    post["theta_ln_K_l_o_wt"] = np.full(S, WT_LN_K_L_O, dtype=np.float32)
    post["theta_ln_K_h_e_wt"] = np.full((S, T), WT_LN_K_H_E, dtype=np.float32)
    post["theta_ln_K_l_e_wt"] = np.full((S, T), WT_LN_K_L_E, dtype=np.float32)
    post["theta_d_ln_K_h_l"] = rng.standard_normal((S, M)).astype(np.float32) * 0.3
    post["theta_d_ln_K_h_o"] = rng.standard_normal((S, M)).astype(np.float32) * 0.3
    post["theta_d_ln_K_l_o"] = rng.standard_normal((S, M)).astype(np.float32) * 0.3
    post["theta_d_ln_K_h_e"] = rng.standard_normal((S, T, M)).astype(np.float32) * 0.3
    post["theta_d_ln_K_l_e"] = rng.standard_normal((S, T, M)).astype(np.float32) * 0.3
    if P > 0:
        for key in ["theta_epi_ln_K_h_l", "theta_epi_ln_K_h_o", "theta_epi_ln_K_l_o"]:
            post[key] = np.zeros((S, P), dtype=np.float32)
        for key in ["theta_epi_ln_K_h_e", "theta_epi_ln_K_l_e"]:
            post[key] = np.zeros((S, T, P), dtype=np.float32)
    return post


def _make_lnK_nn_posteriors(T, M, P=0, S=5, seed=0):
    """Fake posterior dict for mwc_dimer lnK_nn_prior.

    KEY DIFFERENCE: d_ln_K_h_e and d_ln_K_l_e are (S, M) not (S, T, M).
    """
    rng = np.random.default_rng(seed)
    post = {}
    post["theta_ln_K_h_l_wt"] = np.full(S, WT_LN_K_H_L, dtype=np.float32)
    post["theta_ln_K_h_o_wt"] = np.full(S, WT_LN_K_H_O, dtype=np.float32)
    post["theta_ln_K_l_o_wt"] = np.full(S, WT_LN_K_L_O, dtype=np.float32)
    post["theta_ln_K_h_e_wt"] = np.full((S, T), WT_LN_K_H_E, dtype=np.float32)
    post["theta_ln_K_l_e_wt"] = np.full((S, T), WT_LN_K_L_E, dtype=np.float32)
    post["theta_d_ln_K_h_l"] = rng.standard_normal((S, M)).astype(np.float32) * 0.3
    post["theta_d_ln_K_h_o"] = rng.standard_normal((S, M)).astype(np.float32) * 0.3
    post["theta_d_ln_K_l_o"] = rng.standard_normal((S, M)).astype(np.float32) * 0.3
    post["theta_d_ln_K_h_e"] = rng.standard_normal((S, M)).astype(np.float32) * 0.3  # no T
    post["theta_d_ln_K_l_e"] = rng.standard_normal((S, M)).astype(np.float32) * 0.3  # no T
    if P > 0:
        for key in ["theta_epi_ln_K_h_l", "theta_epi_ln_K_h_o", "theta_epi_ln_K_l_o"]:
            post[key] = np.zeros((S, P), dtype=np.float32)
        for key in ["theta_epi_ln_K_h_e", "theta_epi_ln_K_l_e"]:
            post[key] = np.zeros((S, P), dtype=np.float32)   # no T!
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
        conc_unit_scale=CONC_SCALE,
    )


# ---------------------------------------------------------------------------
# lnK_mut
# ---------------------------------------------------------------------------

class TestMwcLnKMutPredict:

    def test_output_shape_and_columns(self):
        post = _make_lnK_mut_posteriors(T=1, M=2)
        result = lnK_mut_predict(**_common_kwargs(post))
        assert len(result) == 4 * 4   # 4 genotypes × 4 concs
        for col in ["genotype", "titrant_name", "titrant_conc", "median"]:
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
        kw["target_genotypes"] = ["wt", "UNKNOWN"]
        result = lnK_mut_predict(**kw)
        assert result.loc[result["genotype"] == "UNKNOWN", "median"].isna().all()
        assert not result.loc[result["genotype"] == "wt", "median"].isna().any()

    def test_upper_ge_median_ge_lower(self):
        post = _make_lnK_mut_posteriors(T=1, M=2, S=40, seed=5)
        result = lnK_mut_predict(**_common_kwargs(post))
        assert (result["upper"] >= result["median"]).all()
        assert (result["median"] >= result["lower"]).all()

    def test_zero_conc_handled(self):
        post = _make_lnK_mut_posteriors(T=1, M=2)
        kw = _common_kwargs(post)
        kw["manual_titrant_df"] = _make_titrant_df(concs=[0.0])
        result = lnK_mut_predict(**kw)
        assert np.isfinite(result["median"].values).all()

    def test_epistasis_shifts_double_mutant(self):
        T, M, P, S = 1, 2, 1, 30
        post = _make_lnK_mut_posteriors(T, M, P=P, S=S, seed=4)
        # Large epistasis offset on K_h_l for the pair
        post["theta_epi_ln_K_h_l"] = np.full((S, P), 4.0, dtype=np.float32)

        kw_no_epi = _common_kwargs(post, pair_labels=[])
        kw_no_epi["target_genotypes"] = ["M42I/K84L"]
        res_no_epi = lnK_mut_predict(**kw_no_epi)

        kw_epi = _common_kwargs(post, pair_labels=PAIR_LABELS)
        kw_epi["target_genotypes"] = ["M42I/K84L"]
        res_epi = lnK_mut_predict(**kw_epi)

        assert not np.allclose(res_no_epi["median"].values,
                               res_epi["median"].values, atol=1e-3)


# ---------------------------------------------------------------------------
# lnK_nn_prior
# ---------------------------------------------------------------------------

class TestMwcLnKNnPriorPredict:

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

    def test_multiple_titrants_broadcast_correctly(self):
        """d_ln_K_h_e and d_ln_K_l_e are (S,M); must broadcast correctly to (S,T,N)."""
        T, M, S = 2, 2, 8
        post = _make_lnK_nn_posteriors(T, M, S=S)
        kw = dict(
            target_genotypes=["wt", "M42I"],
            titrant_names=["IPTG", "TMAIPP"],
            manual_titrant_df=pd.DataFrame({
                "titrant_name": ["IPTG", "TMAIPP"],
                "titrant_conc": [1.0, 1.0],   # mM
            }),
            mut_labels=MUT_LABELS,
            pair_labels=[],
            param_posteriors=post,
            q_to_get={"median": 0.5},
            tf_total=TF_TOTAL,
            op_total=OP_TOTAL,
            conc_unit_scale=CONC_SCALE,
        )
        result = lnK_nn_predict(**kw)
        assert len(result) == 4
        assert not result["median"].isna().any()

    def test_scalar_d_he_le_vs_titrant(self):
        """lnK_nn_prior should match lnK_mut when d_h_e = d_l_e are replicated T times."""
        T, M, S = 2, 2, 5
        rng = np.random.default_rng(77)
        d_he_scalar = rng.standard_normal((S, M)).astype(np.float32)
        d_le_scalar = rng.standard_normal((S, M)).astype(np.float32)

        post_nn = _make_lnK_nn_posteriors(T, M, S=S)
        post_nn["theta_d_ln_K_h_e"] = d_he_scalar
        post_nn["theta_d_ln_K_l_e"] = d_le_scalar

        post_mut = _make_lnK_mut_posteriors(T, M, S=S)
        for k in ["theta_ln_K_h_l_wt", "theta_ln_K_h_o_wt", "theta_ln_K_l_o_wt",
                  "theta_ln_K_h_e_wt", "theta_ln_K_l_e_wt",
                  "theta_d_ln_K_h_l", "theta_d_ln_K_h_o", "theta_d_ln_K_l_o"]:
            post_mut[k] = post_nn[k].copy()
        post_mut["theta_d_ln_K_h_e"] = np.broadcast_to(
            d_he_scalar[:, None, :], (S, T, M)
        ).astype(np.float32)
        post_mut["theta_d_ln_K_l_e"] = np.broadcast_to(
            d_le_scalar[:, None, :], (S, T, M)
        ).astype(np.float32)

        kw = dict(
            target_genotypes=["wt", "M42I", "K84L"],
            titrant_names=["IPTG", "TMAIPP"],
            manual_titrant_df=pd.DataFrame({
                "titrant_name": ["IPTG", "TMAIPP"],
                "titrant_conc": [0.1, 1.0],
            }),
            mut_labels=MUT_LABELS,
            pair_labels=[],
            q_to_get={"median": 0.5},
            tf_total=TF_TOTAL,
            op_total=OP_TOTAL,
            conc_unit_scale=CONC_SCALE,
        )
        res_nn  = lnK_nn_predict(param_posteriors=post_nn,  **kw)
        res_mut = lnK_mut_predict(param_posteriors=post_mut, **kw)

        np.testing.assert_allclose(
            res_nn["median"].values, res_mut["median"].values, atol=1e-5
        )
