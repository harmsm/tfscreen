import pytest
import pandas as pd
import jax.numpy as jnp
import numpy as np
import os
import yaml
import tfscreen
from unittest.mock import MagicMock, call, patch, ANY

# Import the module under test
from tfscreen.tfmodel.model_orchestrator import (
    ModelOrchestrator,
    _read_growth_df,
    _read_binding_df,
    _build_growth_tm,
    _build_binding_tm,
    _setup_batching,
    _check_theta_transformation_compatibility,
    get_batch
)
from tfscreen.tfmodel.generative.components.growth.linear import _parse_condition_label
from tfscreen.tfmodel.tensors.tensor_manager import TensorManager
from tfscreen.tfmodel.analysis.extraction import (
    _extract_param_est,
    extract_parameters,
    extract_theta_curves,
    extract_growth_predictions
)

# ----------------------------------------------------------------------------
# 1. Tests for _build_growth_tm
# ----------------------------------------------------------------------------

@pytest.fixture
def dummy_growth_df():
    return pd.DataFrame({
        "replicate": [1],
        "library": ["lib"],
        "t_sel": [0.0],
        "t_pre": [0.0],
        "genotype": ["wt"],
        "condition_pre": ["A"],
        "condition_sel": ["B"],
        "titrant_name": ["T1"],
        "titrant_conc": [0.1],
        "ln_cfu": [1.0],
        "ln_cfu_std": [0.1],
        "map_theta_group": [0],
        "treatment": [0],
        "condition_sel_reduced": [0]
    })

def test_build_growth_tm(dummy_growth_df):
    res = _build_growth_tm(dummy_growth_df.copy())
    assert isinstance(res, TensorManager)
    assert "ln_cfu" in res.tensors

# ----------------------------------------------------------------------------
# 2. Tests for _build_binding_tm
# ----------------------------------------------------------------------------

def test_build_binding_tm():
    df = pd.DataFrame({
        "genotype": ["wt"],
        "titrant_name": ["T1"],
        "titrant_conc": [0.1],
        "theta_obs": [0.5],
        "theta_std": [0.1]
    })
    res = _build_binding_tm(df)
    assert isinstance(res, TensorManager)
    assert "theta_obs" in res.tensors

# ----------------------------------------------------------------------------
# 3. Tests for _setup_batching
# ----------------------------------------------------------------------------

def test_setup_batching_logic():
    growth_genos = ["wt", "m1"]
    binding_genos = ["wt"]
    batch_size = 1
    res = _setup_batching(growth_genos, binding_genos, batch_size)
    assert "batch_idx" in res
    assert "scale_vector" in res
    assert res["num_binding"] == 1
    
    res = _setup_batching(growth_genos, binding_genos, 100)
    assert len(res["batch_idx"]) == 2

    # Test batch_size=None and num_binding == 0
    res = _setup_batching(["g1"], [], None)
    assert res["num_binding"] == 0
    assert res["batch_size"] == 1

# ----------------------------------------------------------------------------
# 4. Tests for _read_growth_df
# ----------------------------------------------------------------------------

@pytest.fixture
def base_growth_df():
    cols = ["ln_cfu", "ln_cfu_std", "replicate", "t_pre", "t_sel",
            "genotype", "titrant_name", "condition_pre", "condition_sel",
            "titrant_conc"]
    return pd.DataFrame(
        {c: [1.0] for c in cols if c != "genotype"}
        | {"genotype": ["wt"], "library": ["lib"]}
    )

def test_read_growth_df(mocker, base_growth_df):
    mocker.patch("tfscreen.util.io.read_dataframe", return_value=base_growth_df)
    mocker.patch("tfscreen.genetics.set_categorical_genotype", return_value=base_growth_df)
    mocker.patch("tfscreen.util.dataframe.get_scaled_cfu", return_value=base_growth_df)
    mocker.patch("tfscreen.util.dataframe.check_columns")
    mocker.patch("tfscreen.tfmodel.model_orchestrator.add_group_columns", return_value=base_growth_df)
    
    res = _read_growth_df("path.csv")
    assert "replicate" in res.columns

def test_read_growth_df_no_replicate(mocker, base_growth_df):
    df_no_rep = base_growth_df.drop(columns=["replicate"])
    mocker.patch("tfscreen.util.io.read_dataframe", return_value=df_no_rep)
    mocker.patch("tfscreen.genetics.set_categorical_genotype", return_value=df_no_rep)
    mocker.patch("tfscreen.util.dataframe.get_scaled_cfu", return_value=df_no_rep)
    mocker.patch("tfscreen.util.dataframe.check_columns")
    mocker.patch("tfscreen.tfmodel.model_orchestrator.add_group_columns", return_value=df_no_rep)

    res = _read_growth_df("path.csv")
    assert "replicate" in res.columns
    assert res["replicate"].iloc[0] == 1

def test_read_growth_df_coerces_time_to_float(mocker, base_growth_df):
    """t_pre and t_sel stored as int64 (e.g. from pd.read_csv with integer values)
    must be cast to float64 so that downstream merges against float prediction grids
    don't produce int/float dtype warnings."""
    df_int_time = base_growth_df.copy()
    df_int_time["t_pre"] = df_int_time["t_pre"].astype(int)
    df_int_time["t_sel"] = df_int_time["t_sel"].astype(int)
    assert df_int_time["t_pre"].dtype == int
    assert df_int_time["t_sel"].dtype == int

    mocker.patch("tfscreen.util.io.read_dataframe", return_value=df_int_time)
    mocker.patch("tfscreen.util.dataframe.get_scaled_cfu", return_value=df_int_time)
    mocker.patch("tfscreen.util.dataframe.check_columns")
    mocker.patch("tfscreen.tfmodel.model_orchestrator.add_group_columns", return_value=df_int_time)

    res = _read_growth_df("path.csv")
    assert res["t_pre"].dtype == float, f"expected float, got {res['t_pre'].dtype}"
    assert res["t_sel"].dtype == float, f"expected float, got {res['t_sel'].dtype}"

# ----------------------------------------------------------------------------
# 5. Tests for _read_binding_df
# ----------------------------------------------------------------------------

def test_read_binding_df(mocker):
    growth_df = pd.DataFrame({"genotype": ["A"], "titrant_name": ["T1"]})
    binding_df = pd.DataFrame({
        "genotype": ["A"], "titrant_name": ["T1"],
        "theta_obs": [0.5], "theta_std": [0.1], "titrant_conc": [1.0]
    })
    mocker.patch("tfscreen.util.io.read_dataframe", return_value=binding_df)
    mocker.patch("tfscreen.genetics.set_categorical_genotype", return_value=binding_df)
    mocker.patch("tfscreen.util.dataframe.check_columns")
    mocker.patch(
        "tfscreen.tfmodel.model_orchestrator.add_group_columns",
        return_value=binding_df,
    )

    res = _read_binding_df("path.csv", growth_df=growth_df)
    assert "theta_obs" in res.columns

def test_read_binding_df_missing_col_error(mocker):
    growth_df = pd.DataFrame({"genotype": ["A"], "titrant_name": ["T1"]})
    binding_df = pd.DataFrame({
        "genotype": ["A"], "titrant_name": ["T1"],
        "theta_obs": [0.5], "theta_std": [0.1], "titrant_conc": [1.0]
    })
    mocker.patch("tfscreen.util.io.read_dataframe", return_value=binding_df)
    mocker.patch("tfscreen.genetics.set_categorical_genotype", return_value=binding_df)
    mocker.patch("tfscreen.util.dataframe.check_columns", side_effect=ValueError("missing col"))
    with pytest.raises(ValueError, match="missing col"):
        _read_binding_df("path.csv", growth_df=growth_df)


def test_read_binding_df_extra_pairs_dropped_with_warning(capsys):
    """binding rows whose (genotype, titrant_name) is absent from growth_df are dropped."""
    from tfscreen.genetics import set_categorical_genotype
    import tfscreen.util.dataframe

    growth_df = pd.DataFrame({
        "genotype": ["wt", "wt"],
        "titrant_name": ["iptg", "iptg"],
        "titrant_conc": [0.0, 1.0],
        "ln_cfu": [10.0, 10.0],
        "ln_cfu_std": [0.1, 0.1],
        "t_pre": [1.0, 1.0],
        "t_sel": [1.0, 1.0],
        "replicate": [1, 1],
        "condition_pre": ["kan", "kan"],
        "condition_sel": ["kan", "kan"],
    })
    growth_df = set_categorical_genotype(growth_df, standardize=True)
    growth_df = tfscreen.util.dataframe.add_group_columns(
        growth_df, ["genotype", "titrant_name"], "map_theta_group"
    )

    # binding has "wt" (in growth) and "M42I" (not in growth)
    binding_df = pd.DataFrame({
        "genotype": ["wt", "M42I"],
        "titrant_name": ["iptg", "iptg"],
        "titrant_conc": [1.0, 1.0],
        "theta_obs": [0.1, 0.5],
        "theta_std": [0.02, 0.02],
    })

    result = _read_binding_df(binding_df, growth_df=growth_df)
    captured = capsys.readouterr()
    assert "will be dropped" in captured.out

    # Only the "wt" row should survive
    assert set(result["genotype"]) == {"wt"}
    assert len(result) == 1


def test_read_binding_df_preserves_canonical_genotype_order():
    # Regression test: the pd.merge inside add_group_columns converts the
    # categorical genotype column to object dtype.  _read_binding_df must
    # restore canonical ordering (wt first, then singles by site) so that
    # _build_binding_tm assigns the same genotype indices that _setup_batching
    # expects from the growth_tm.  Without the fix, genotypes appear in
    # alphabetical order (e.g. H74A=0 instead of wt=0), which misaligns the
    # binding theta_obs tensor with the batch_idx used by the Hill model.

    # Build a minimal growth_df with the four genotypes in canonical order.
    # set_categorical_genotype puts wt first, then singles by site number.
    from tfscreen.genetics import set_categorical_genotype
    growth_df = pd.DataFrame({
        "genotype": ["wt", "wt", "M42I", "M42I", "H74A", "H74A", "K84L", "K84L"],
        "titrant_name": ["iptg"] * 8,
        "titrant_conc": [0.0, 1.0] * 4,
        "ln_cfu": [10.0] * 8,
        "ln_cfu_std": [0.1] * 8,
        "t_pre": [1.0] * 8,
        "t_sel": [1.0] * 8,
        "replicate": [1] * 8,
        "condition_pre": ["kan"] * 8,
        "condition_sel": ["kan"] * 8,
    })
    growth_df = set_categorical_genotype(growth_df, standardize=True)
    growth_df = tfscreen.util.dataframe.add_group_columns(
        growth_df, ["genotype", "titrant_name"], "map_theta_group"
    )

    binding_df = pd.DataFrame({
        "genotype": ["wt", "M42I", "H74A", "K84L"],
        "titrant_name": ["iptg"] * 4,
        "titrant_conc": [1.0] * 4,
        "theta_obs": [0.1, 0.5, 0.7, 0.9],
        "theta_std": [0.02] * 4,
    })

    result = _read_binding_df(binding_df, growth_df=growth_df)

    # After the fix the genotype column must be categorical with canonical order.
    assert hasattr(result["genotype"], "cat"), "genotype column must be categorical"
    categories = list(result["genotype"].cat.categories)
    assert categories == ["wt", "M42I", "H74A", "K84L"], (
        f"Expected canonical order ['wt', 'M42I', 'H74A', 'K84L'], got {categories}"
    )

    # The map_theta_group values must match what the growth_df assigned so that
    # _setup_batching and the binding tensor stay aligned.
    # With 1 titrant name each (genotype, titrant_name) pair gets a consecutive
    # index in canonical genotype order: wt=0, M42I=1, H74A=2, K84L=3.
    expected = {
        "wt": 0, "M42I": 1, "H74A": 2, "K84L": 3
    }
    for geno, exp_idx in expected.items():
        row = result[result["genotype"] == geno]
        assert row["map_theta_group"].iloc[0] == exp_idx, (
            f"map_theta_group for {geno}: expected {exp_idx}, "
            f"got {row['map_theta_group'].iloc[0]}"
        )


# ----------------------------------------------------------------------------
# 7. Tests for _extract_param_est
# ----------------------------------------------------------------------------

def test_extract_param_est():
    input_df = pd.DataFrame({"map_id": [0, 1], "gene_name": ["G1", "G2"]})
    samples = np.array([[10.0, 20.0], [10.0, 20.0]])
    param_posteriors = {"p_v": samples}
    res = _extract_param_est(input_df, ["v"], "map_id", ["gene_name"], "p_", param_posteriors, {"m": 0.5})
    assert "v" in res

# ----------------------------------------------------------------------------
# 8. Tests for ModelOrchestrator
# ----------------------------------------------------------------------------

@pytest.fixture
def initialized_model_class():
    model = MagicMock(spec=ModelOrchestrator)
    model._theta = "hill_geno"
    model._condition_growth = "hierarchical"
    model._growth_transition = "instant"
    model._ln_cfu0 = "hierarchical"
    model._dk_geno = "hierarchical_geno"
    model._activity = "hierarchical_geno"
    model._transformation = "single"
    model._theta_growth_noise = "zero"
    model._theta_binding_noise = "zero"
    model._spiked_genotypes = None
    model._growth_shares_replicates = False
    model._batch_size = 1
    model.growth_tm = MagicMock()
    model.training_tm = model.growth_tm
    model.binding_tm = MagicMock()
    model.growth_df = pd.DataFrame()
    model.mut_labels = []
    model.pair_labels = []
    return model

def create_mock_tm(is_growth=True):
    tm = MagicMock()
    tm.map_sizes = {}
    if is_growth:
        tm.map_sizes = {"condition_rep": 1}
        tm.tensor_shape = [1, 1, 1, 1, 1, 1, 7]
        tm.tensor_dim_names = ["replicate", "time", "condition_pre", "condition_sel", "titrant_name", "titrant_conc", "genotype"]
        tm.tensor_dim_labels = [
            np.array([1]), np.array([0]), np.array(["A"]), np.array(["B"]), 
            np.array(["T"]), np.array([1.0]), np.array(["wt", "m1", "m2", "m3", "m4", "m5", "m6"])
        ]
        tm.tensors = {
            "ln_cfu": np.zeros((1, 1, 1, 1, 1, 1, 7)),
            "ln_cfu_std": np.zeros((1, 1, 1, 1, 1, 1, 7)),
            "t_pre": np.zeros((1, 1, 1, 1, 1, 1, 7)),
            "t_sel": np.zeros((1, 1, 1, 1, 1, 1, 7)),
            "map_condition_pre": np.zeros((1, 1, 1, 1, 1, 1, 7)),
            "map_condition_sel": np.zeros((1, 1, 1, 1, 1, 1, 7)),
            "good_mask": np.ones((1, 1, 1, 1, 1, 1, 7), dtype=bool)
        }
    else:
        tm.tensor_shape = [1, 1, 1]
        tm.tensor_dim_names = ["name", "titrant_conc", "genotype"]
        tm.tensor_dim_labels = [np.array(["T"]), np.array([1.0]), np.array(["wt"])]
        tm.tensors = {
            "theta_obs": np.zeros((1, 1, 1)),
            "theta_std": np.zeros((1, 1, 1)),
            "good_mask": np.ones((1, 1, 1), dtype=bool)
        }
    return tm

def test_model_class_invalid_component(mocker):
    mock_growth_tm = create_mock_tm(is_growth=True)
    mock_binding_tm = create_mock_tm(is_growth=False)
    mocker.patch("tfscreen.tfmodel.model_orchestrator._read_growth_df")
    mocker.patch("tfscreen.tfmodel.model_orchestrator._build_growth_tm", return_value=mock_growth_tm)
    mocker.patch("tfscreen.tfmodel.model_orchestrator._read_binding_df")
    mocker.patch("tfscreen.tfmodel.model_orchestrator._build_binding_tm", return_value=mock_binding_tm)
    mocker.patch("tfscreen.tfmodel.model_orchestrator.populate_dataclass", return_value=MagicMock())
    mocker.patch("tfscreen.tfmodel.model_orchestrator._setup_batching", return_value={"batch_idx": jnp.array([0]), "batch_size": 1, "scale_vector": jnp.array([1.0]), "num_binding": 0})
    
    # 741: dk_geno unrecognized
    with pytest.raises(ValueError, match="not recognized"):
        ModelOrchestrator("g.csv", "b.csv", dk_geno="invalid_name")

    # 735: key not in model_registry
    with patch("tfscreen.tfmodel.model_orchestrator.model_registry", {}):
        with pytest.raises(ValueError, match="is not in model_registry"):
            ModelOrchestrator("g.csv", "b.csv")

def test_initialize_data_spiked(mocker):
    mock_growth_tm = create_mock_tm(is_growth=True)
    mock_growth_tm.tensor_shape[6] = 2
    mock_growth_tm.tensor_dim_labels[6] = np.array(["wt", "m1"])
    for k in mock_growth_tm.tensors:
        mock_growth_tm.tensors[k] = np.zeros((1,1,1,1,1,1,2))
    
    mock_binding_tm = create_mock_tm(is_growth=False)

    mocker.patch("tfscreen.tfmodel.model_orchestrator._read_growth_df")
    mocker.patch("tfscreen.tfmodel.model_orchestrator._build_growth_tm", return_value=mock_growth_tm)
    mocker.patch("tfscreen.tfmodel.model_orchestrator._read_binding_df")
    mocker.patch("tfscreen.tfmodel.model_orchestrator._build_binding_tm", return_value=mock_binding_tm)
    mocker.patch("tfscreen.tfmodel.model_orchestrator.populate_dataclass", return_value=MagicMock())
    mocker.patch("tfscreen.tfmodel.model_orchestrator._setup_batching", return_value={"batch_idx": jnp.array([0,1]), "batch_size": 2, "scale_vector": jnp.array([1.0, 1.0]), "num_binding": 0})
    mocker.patch("tfscreen.tfmodel.model_orchestrator.ModelOrchestrator._initialize_classes")

    model = ModelOrchestrator("g.csv", "b.csv", spiked_genotypes=["m1"])
    assert model._spiked_genotypes == ["m1"]
    
    model = ModelOrchestrator("g.csv", "b.csv", spiked_genotypes="m1")
    assert model._spiked_genotypes == ["m1"]
    
    with pytest.raises(ValueError, match="not found"):
        ModelOrchestrator("g.csv", "b.csv", spiked_genotypes="m2")

def test_initialize_classes_logic(mocker):
    mock_growth_tm = create_mock_tm(is_growth=True)
    mock_binding_tm = create_mock_tm(is_growth=False)
    mocker.patch("tfscreen.tfmodel.model_orchestrator._read_growth_df")
    mocker.patch("tfscreen.tfmodel.model_orchestrator._build_growth_tm", return_value=mock_growth_tm)
    mocker.patch("tfscreen.tfmodel.model_orchestrator._read_binding_df")
    mocker.patch("tfscreen.tfmodel.model_orchestrator._build_binding_tm", return_value=mock_binding_tm)
    mocker.patch("tfscreen.tfmodel.model_orchestrator._setup_batching", return_value={"batch_idx": jnp.array([0]), "batch_size": 1, "scale_vector": jnp.array([1.0]), "num_binding": 0})
    mocker.patch("tfscreen.tfmodel.model_orchestrator.populate_dataclass", return_value=MagicMock())

    with patch.dict("tfscreen.tfmodel.model_orchestrator.model_registry", {
        "condition_growth": {"hierarchical": MagicMock(), "independent": MagicMock()},
        "growth_transition": {"instant": MagicMock(), "linear": MagicMock()},
        "ln_cfu0": {"hierarchical": MagicMock()},
        "dk_geno": {"hierarchical_geno": MagicMock()},
        "activity": {"hierarchical_geno": MagicMock(), "horseshoe_geno": MagicMock()},
        "theta": {"categorical_geno": MagicMock(), "hill_geno": MagicMock(), "fixed": MagicMock()},
        "transformation": {"logit_norm": MagicMock(), "single": MagicMock()},
        "theta_rescale": {"passthrough": MagicMock()},
        "theta_growth_noise": {"zero": MagicMock(), "logit_normal": MagicMock()},
        "theta_binding_noise": {"zero": MagicMock()},
        "growth_noise": {"zero": MagicMock()},
        "sample_offset": {"zero": MagicMock()},
        "observe_binding": MagicMock(),
        "observe_growth": MagicMock()
    }, clear=True):
        for k in ["condition_growth", "growth_transition", "ln_cfu0", "dk_geno", "activity", "theta", "transformation", "theta_growth_noise", "theta_binding_noise", "growth_noise", "sample_offset"]:
            for sub_k in tfscreen.tfmodel.model_orchestrator.model_registry[k]:
                tfscreen.tfmodel.model_orchestrator.model_registry[k][sub_k].get_priors.return_value = {}
                tfscreen.tfmodel.model_orchestrator.model_registry[k][sub_k].get_guesses.return_value = {}

        model = ModelOrchestrator("g.csv", "b.csv", theta="categorical_geno", transformation="logit_norm", condition_growth="independent", growth_transition="instant")
        assert model._theta == "categorical_geno"


# ---------------------------------------------------------------------------
# presplit forwarding to component get_priors/get_guesses
# ---------------------------------------------------------------------------

class _LnCfu0ModuleWithPresplit:
    """
    A minimal, real (non-Mock) component module whose get_priors/get_guesses
    declare a `presplit` parameter, so inspect.signature can actually detect
    it -- MagicMock's auto-generated (*args, **kwargs) signature can't be
    used to exercise this branch (see test below).
    """
    recorded = {}

    @staticmethod
    def get_priors(data=None, presplit=None):
        _LnCfu0ModuleWithPresplit.recorded["priors_data"] = data
        _LnCfu0ModuleWithPresplit.recorded["priors_presplit"] = presplit
        return {}

    @staticmethod
    def get_guesses(name, data, presplit=None):
        _LnCfu0ModuleWithPresplit.recorded["guesses_data"] = data
        _LnCfu0ModuleWithPresplit.recorded["guesses_presplit"] = presplit
        return {}

    define_model = MagicMock()
    guide = MagicMock()


def test_initialize_classes_forwards_presplit_to_declaring_component(mocker):
    """
    _initialize_classes must forward self._data.presplit to a component's
    get_priors/get_guesses only when its signature declares a `presplit`
    parameter -- currently just the ln_cfu0 component.  Components that
    don't declare it (everything else, here plain MagicMocks) are
    unaffected and keep working exactly as before.
    """
    mock_growth_tm = create_mock_tm(is_growth=True)
    mock_binding_tm = create_mock_tm(is_growth=False)
    mocker.patch("tfscreen.tfmodel.model_orchestrator._read_growth_df")
    mocker.patch("tfscreen.tfmodel.model_orchestrator._build_growth_tm", return_value=mock_growth_tm)
    mocker.patch("tfscreen.tfmodel.model_orchestrator._read_binding_df")
    mocker.patch("tfscreen.tfmodel.model_orchestrator._build_binding_tm", return_value=mock_binding_tm)
    mocker.patch("tfscreen.tfmodel.model_orchestrator._setup_batching", return_value={"batch_idx": jnp.array([0]), "batch_size": 1, "scale_vector": jnp.array([1.0]), "num_binding": 0})

    fake_data = MagicMock()
    fake_data.presplit = "PRESPLIT_SENTINEL"
    mocker.patch("tfscreen.tfmodel.model_orchestrator.populate_dataclass", return_value=fake_data)

    _LnCfu0ModuleWithPresplit.recorded.clear()

    with patch.dict("tfscreen.tfmodel.model_orchestrator.model_registry", {
        "condition_growth": {"independent": MagicMock()},
        "growth_transition": {"instant": MagicMock()},
        "ln_cfu0": {"hierarchical": _LnCfu0ModuleWithPresplit},
        "dk_geno": {"hierarchical_geno": MagicMock()},
        "activity": {"hierarchical_geno": MagicMock()},
        "theta": {"categorical_geno": MagicMock()},
        "transformation": {"logit_norm": MagicMock()},
        "theta_rescale": {"passthrough": MagicMock()},
        "theta_growth_noise": {"zero": MagicMock()},
        "theta_binding_noise": {"zero": MagicMock()},
        "growth_noise": {"zero": MagicMock()},
        "sample_offset": {"zero": MagicMock()},
        "observe_binding": MagicMock(),
        "observe_growth": MagicMock(),
    }, clear=True):
        for k in ["condition_growth", "growth_transition", "dk_geno", "activity",
                 "theta", "transformation", "theta_growth_noise",
                 "theta_binding_noise", "growth_noise", "sample_offset"]:
            for sub_k in tfscreen.tfmodel.model_orchestrator.model_registry[k]:
                mod = tfscreen.tfmodel.model_orchestrator.model_registry[k][sub_k]
                mod.get_priors.return_value = {}
                mod.get_guesses.return_value = {}

        ModelOrchestrator(
            "g.csv", "b.csv",
            theta="categorical_geno", transformation="logit_norm",
            condition_growth="independent", growth_transition="instant",
            ln_cfu0="hierarchical", activity="hierarchical_geno",
            theta_growth_noise="zero",
        )

    # presplit forwarded to the component that declares it...
    assert _LnCfu0ModuleWithPresplit.recorded["priors_presplit"] == "PRESPLIT_SENTINEL"
    assert _LnCfu0ModuleWithPresplit.recorded["guesses_presplit"] == "PRESPLIT_SENTINEL"
    # ...alongside the usual `data` kwarg, unaffected by the new plumbing.
    assert _LnCfu0ModuleWithPresplit.recorded["priors_data"] is fake_data.growth
    assert _LnCfu0ModuleWithPresplit.recorded["guesses_data"] is fake_data.growth


# ---------------------------------------------------------------------------
# dk_geno="pinned" wiring
# ---------------------------------------------------------------------------

def test_dk_geno_pinned_requires_pins_file():
    """
    dk_geno='pinned' without dk_geno_pins_file must raise before any data
    I/O happens (mirrors
    test_model_orchestrator_rejects_categorical_geno_with_empirical:
    "g.csv"/"b.csv" are never touched).
    """
    with pytest.raises(ValueError, match="requires dk_geno_pins_file"):
        ModelOrchestrator("g.csv", "b.csv", dk_geno="pinned")


def test_dk_geno_pins_file_without_pinned_component_raises():
    """A stray dk_geno_pins_file with any other dk_geno component must raise."""
    with pytest.raises(ValueError, match="dk_geno != 'pinned'"):
        ModelOrchestrator(
            "g.csv", "b.csv",
            dk_geno="hierarchical_geno",
            dk_geno_pins_file="pins.csv",
        )


class _SpyPinnedModule:
    """
    Thin real-object wrapper around dk_geno.pinned.

    inspect.signature(component_module.get_priors) must see the real
    "dk_geno_values" parameter for _initialize_classes's branch selection to
    fire (a bare MagicMock's signature is just (*args, **kwargs), which
    would silently fall through to the wrong branch). Wrapping with an
    explicit method of the same shape gets us a real signature while still
    recording what was actually passed.
    """

    def __init__(self, real_module):
        self._real_module = real_module
        self.calls = []

    def get_priors(self, dk_geno_values=None):
        self.calls.append(dk_geno_values)
        return self._real_module.get_priors(dk_geno_values=dk_geno_values)

    def __getattr__(self, name):
        return getattr(self._real_module, name)


def test_dk_geno_pinned_builds_values_and_wires_into_priors(mocker, tmp_path):
    """
    End-to-end: a pinned dk_geno_pins_file is read, converted to a
    (num_genotype,) array ordered by the growth genotype axis, and passed
    into the "pinned" component's get_priors(dk_geno_values=...).
    """
    from tfscreen.tfmodel.generative.components.dk_geno import pinned as real_pinned

    csv_path = tmp_path / "pins.csv"
    csv_path.write_text("genotype,dk_geno\nm1,-0.02\nm3,0.01\n")

    mock_growth_tm = create_mock_tm(is_growth=True)
    mock_binding_tm = create_mock_tm(is_growth=False)
    mocker.patch("tfscreen.tfmodel.model_orchestrator._read_growth_df")
    mocker.patch("tfscreen.tfmodel.model_orchestrator._build_growth_tm", return_value=mock_growth_tm)
    mocker.patch("tfscreen.tfmodel.model_orchestrator._read_binding_df")
    mocker.patch("tfscreen.tfmodel.model_orchestrator._build_binding_tm", return_value=mock_binding_tm)
    mocker.patch("tfscreen.tfmodel.model_orchestrator.populate_dataclass", return_value=MagicMock())
    mocker.patch(
        "tfscreen.tfmodel.model_orchestrator._setup_batching",
        return_value={
            "batch_idx": jnp.arange(7), "batch_size": 7,
            "scale_vector": jnp.ones(7), "num_binding": 0,
        },
    )

    spy = _SpyPinnedModule(real_pinned)

    with patch.dict("tfscreen.tfmodel.model_orchestrator.model_registry", {
        "condition_growth": {"hierarchical": MagicMock()},
        "growth_transition": {"instant": MagicMock()},
        "ln_cfu0": {"hierarchical": MagicMock()},
        "dk_geno": {"pinned": spy},
        "activity": {"hierarchical_geno": MagicMock()},
        "theta": {"hill_geno": MagicMock()},
        "transformation": {"single": MagicMock()},
        "theta_rescale": {"passthrough": MagicMock()},
        "theta_growth_noise": {"zero": MagicMock()},
        "theta_binding_noise": {"zero": MagicMock()},
        "growth_noise": {"zero": MagicMock()},
        "sample_offset": {"zero": MagicMock()},
        "observe_binding": MagicMock(),
        "observe_growth": MagicMock(),
    }, clear=True):
        for k in ["condition_growth", "growth_transition", "ln_cfu0", "activity",
                  "theta", "transformation", "theta_growth_noise",
                  "theta_binding_noise", "growth_noise", "sample_offset"]:
            for sub_k in tfscreen.tfmodel.model_orchestrator.model_registry[k]:
                tfscreen.tfmodel.model_orchestrator.model_registry[k][sub_k].get_priors.return_value = {}
                tfscreen.tfmodel.model_orchestrator.model_registry[k][sub_k].get_guesses.return_value = {}

        model = ModelOrchestrator(
            "g.csv", "b.csv",
            dk_geno="pinned",
            dk_geno_pins_file=str(csv_path),
            condition_growth="hierarchical",
            growth_transition="instant",
            activity="hierarchical_geno",
            theta="hill_geno",
            transformation="single",
            theta_growth_noise="zero",
        )

    expected = np.array([0.0, -0.02, 0.0, 0.01, 0.0, 0.0, 0.0])

    # _dk_geno_values built during _initialize_data, ordered wt,m1..m6.
    assert np.allclose(np.asarray(model._dk_geno_values), expected)

    # The component received that exact array via get_priors(dk_geno_values=...).
    assert len(spy.calls) == 1
    assert np.allclose(np.asarray(spy.calls[0]), expected)

    # Round-trips through the settings dict for YAML config serialisation.
    assert model.settings["dk_geno_pins_file"] == str(csv_path)
    assert model.settings["dk_geno"] == "pinned"


@pytest.mark.parametrize("transformation_key, expected_needs_population", [
    ("single", False),
    ("logit_norm", False),
    ("empirical", True),
])
def test_transformation_control_kwargs_carry_needs_population_flag(
        mocker, transformation_key, expected_needs_population):
    """
    main_control_kwargs["transformation"] must be a 3-tuple whose third
    element reflects the *real* transformation component's
    NEEDS_FULL_POPULATION_THETA flag (empirical=True; single/logit_norm=False)
    — this is the wiring jax_model relies on to decide whether it needs a
    population-wide theta reference for the congression correction.
    """
    import tfscreen.tfmodel.generative.components.transformation as transformation_pkg

    mock_growth_tm = create_mock_tm(is_growth=True)
    mock_binding_tm = create_mock_tm(is_growth=False)
    mocker.patch("tfscreen.tfmodel.model_orchestrator._read_growth_df")
    mocker.patch("tfscreen.tfmodel.model_orchestrator._build_growth_tm", return_value=mock_growth_tm)
    mocker.patch("tfscreen.tfmodel.model_orchestrator._read_binding_df")
    mocker.patch("tfscreen.tfmodel.model_orchestrator._build_binding_tm", return_value=mock_binding_tm)
    mocker.patch("tfscreen.tfmodel.model_orchestrator._setup_batching", return_value={"batch_idx": jnp.array([0]), "batch_size": 1, "scale_vector": jnp.array([1.0]), "num_binding": 0})
    mocker.patch("tfscreen.tfmodel.model_orchestrator.populate_dataclass", return_value=MagicMock())

    real_transformation_module = getattr(transformation_pkg, transformation_key)

    with patch.dict("tfscreen.tfmodel.model_orchestrator.model_registry", {
        "condition_growth": {"independent": MagicMock()},
        "growth_transition": {"instant": MagicMock()},
        "ln_cfu0": {"hierarchical": MagicMock()},
        "dk_geno": {"hierarchical_geno": MagicMock()},
        "activity": {"horseshoe_geno": MagicMock()},
        # Use a mocked "hill_geno" stand-in (not "categorical_geno") so this
        # wiring test doesn't trip the real theta/transformation compatibility
        # check exercised separately below.
        "theta": {"hill_geno": MagicMock()},
        "transformation": {transformation_key: real_transformation_module},
        "theta_rescale": {"passthrough": MagicMock()},
        "theta_growth_noise": {"logit_normal": MagicMock()},
        "theta_binding_noise": {"zero": MagicMock()},
        "growth_noise": {"zero": MagicMock()},
        "sample_offset": {"zero": MagicMock()},
        "observe_binding": MagicMock(),
        "observe_growth": MagicMock()
    }, clear=True):
        for k in ["condition_growth", "growth_transition", "ln_cfu0", "dk_geno",
                  "activity", "theta", "theta_growth_noise", "theta_binding_noise",
                  "growth_noise", "sample_offset"]:
            for sub_k in tfscreen.tfmodel.model_orchestrator.model_registry[k]:
                tfscreen.tfmodel.model_orchestrator.model_registry[k][sub_k].get_priors.return_value = {}
                tfscreen.tfmodel.model_orchestrator.model_registry[k][sub_k].get_guesses.return_value = {}

        model = ModelOrchestrator(
            "g.csv", "b.csv",
            theta="hill_geno",
            transformation=transformation_key,
            condition_growth="independent",
            growth_transition="instant",
        )

        transformation_control = model.main_control_kwargs["transformation"]
        assert len(transformation_control) == 3
        assert transformation_control[2] is expected_needs_population


# ---------------------------------------------------------------------------
# _check_theta_transformation_compatibility
# ---------------------------------------------------------------------------

def test_check_theta_transformation_compatibility_rejects_categorical_geno_empirical():
    """categorical_geno cannot supply a full-population theta reference, so
    pairing it with transformation='empirical' must raise, not silently
    reproduce the population-CDF bug."""
    with pytest.raises(ValueError, match="categorical_geno.*empirical|empirical.*categorical_geno"):
        _check_theta_transformation_compatibility("categorical_geno", "empirical")


@pytest.mark.parametrize("transformation_key", ["single", "logit_norm"])
def test_check_theta_transformation_compatibility_allows_categorical_geno_elsewhere(
        transformation_key):
    """categorical_geno is fine with transformation models that don't need a
    population-wide theta reference."""
    _check_theta_transformation_compatibility("categorical_geno", transformation_key)


@pytest.mark.parametrize("theta_key", ["hill_geno", "hill_mut", "some_thermo_variant"])
def test_check_theta_transformation_compatibility_allows_other_theta_with_empirical(
        theta_key):
    """Any theta component other than the known-incompatible ones must be
    allowed with transformation='empirical' (including forward-compatibility
    with theta components this check has never heard of)."""
    _check_theta_transformation_compatibility(theta_key, "empirical")


def test_model_orchestrator_rejects_categorical_geno_with_empirical():
    """
    End-to-end: constructing a ModelOrchestrator with the incompatible pair
    must raise before any data loading happens (no growth_df/binding_df I/O
    needed for this to fire — "g.csv"/"b.csv" are never touched).
    """
    with pytest.raises(ValueError, match="categorical_geno"):
        ModelOrchestrator(
            "g.csv", "b.csv",
            theta="categorical_geno",
            transformation="empirical",
        )


def test_check_theta_transformation_compatibility_binding_only_skips_check():
    """
    In binding-only mode jax_model returns before ever reaching the
    transformation/congression code path (see generative/model.py's early
    `if binding_only: ... return`), so the transformation setting is inert
    and categorical_geno + empirical must be allowed.  Regression test for
    the smoke-test failure where tfs-configure-model's binding-only pipeline
    (theta="categorical_geno", transformation left at its "empirical"
    default) was incorrectly rejected by this check.
    """
    _check_theta_transformation_compatibility(
        "categorical_geno", "empirical", binding_only=True
    )


def test_model_class_properties(initialized_model_class):
    model = initialized_model_class
    model._jax_model = "jm"
    model._jax_model_guide = "jmg"
    model._data = "d"
    model._priors = "p"
    model._init_params = "ip"
    model._batch_size = 1
    model._activity = "a"
    model._theta = "t"
    model._transformation = "tr"
    model._theta_rescale = "rs"
    model._condition_growth = "cg"
    model._growth_transition = "gt"
    model._ln_cfu0 = "ln"
    model._dk_geno = "dk"
    model._dk_geno_pins_file = None
    model._theta_growth_noise = "gn"
    model._theta_binding_noise = "bn"
    model._growth_noise = "grn"
    model._sample_offset = "so"
    model._spiked_genotypes = ["s"]
    model._binding_only = False
    model._growth_shares_replicates = False
    model._epistasis = True
    model._thermo_data = None
    model._binding_weight = 1.0

    assert ModelOrchestrator.jax_model.fget(model) == "jm"
    assert ModelOrchestrator.jax_model_guide.fget(model) == "jmg"
    assert ModelOrchestrator.data.fget(model) == "d"
    assert ModelOrchestrator.priors.fget(model) == "p"
    assert ModelOrchestrator.init_params.fget(model) == "ip"
    assert ModelOrchestrator.settings.fget(model)["activity"] == "a"
    assert ModelOrchestrator.settings.fget(model)["theta"] == "t"
    assert ModelOrchestrator.settings.fget(model)["transformation"] == "tr"
    assert ModelOrchestrator.settings.fget(model)["theta_rescale"] == "rs"
    assert ModelOrchestrator.settings.fget(model)["condition_growth"] == "cg"
    assert ModelOrchestrator.settings.fget(model)["growth_transition"] == "gt"
    assert ModelOrchestrator.settings.fget(model)["ln_cfu0"] == "ln"
    assert ModelOrchestrator.settings.fget(model)["dk_geno"] == "dk"
    assert ModelOrchestrator.settings.fget(model)["dk_geno_pins_file"] is None
    assert ModelOrchestrator.settings.fget(model)["theta_growth_noise"] == "gn"
    assert ModelOrchestrator.settings.fget(model)["theta_binding_noise"] == "bn"
    assert ModelOrchestrator.settings.fget(model)["growth_noise"] == "grn"
    assert ModelOrchestrator.settings.fget(model)["spiked_genotypes"] == ["s"]
    assert ModelOrchestrator.settings.fget(model)["growth_shares_replicates"] == False
    assert ModelOrchestrator.settings.fget(model)["epistasis"] == True
    assert ModelOrchestrator.get_batch.fget(model) == get_batch

def test_extract_parameters_full(initialized_model_class):
    model = initialized_model_class
    model._theta = "hill_geno"
    model._condition_growth = "hierarchical"
    model._dk_geno = "hierarchical_geno"
    model._activity = "hierarchical_geno"
    model._transformation = "logit_norm"
    model.growth_tm.df = pd.DataFrame({
        "genotype": ["wt"], "titrant_name": ["T"], 
        "replicate": [1], "condition_pre": ["A"],
        "map_theta_group": [0], "map_ln_cfu0": [0],
        "map_genotype": [0], "titrant_name_idx": [0], "titrant_conc_idx": [0],
        "titrant_conc": [1.0]
    })
    model.growth_tm.map_groups = {"condition_rep": pd.DataFrame({"replicate":[1], "condition_rep":["A"], "map_condition_rep":[0]})}
    model.growth_tm.tensor_dim_names = ["replicate", "time", "condition_pre", "condition_sel", "titrant_name", "titrant_conc", "genotype"]
    model.growth_tm.tensor_dim_labels = [[], [], [], [], [], [1.0], []]
    
    post = {
        "theta_hill_n": np.zeros((1, 1, 1)), "theta_log_hill_K": np.zeros((1, 1, 1)), 
        "theta_theta_high": np.zeros((1, 1, 1)), "theta_theta_low": np.zeros((1, 1, 1)),
        "ln_cfu0": np.zeros((1, 1, 1)), "dk_geno": np.zeros((1, 1, 1)),
        "condition_growth_m": np.zeros((1, 1, 1)), "condition_growth_k": np.zeros((1, 1, 1)),
        "activity": np.zeros((1, 1, 1)),
        "transformation_lam": np.zeros((1,)), "transformation_mu": np.zeros((1, 1)), "transformation_sigma": np.zeros((1, 1))
    }
    res = extract_parameters(model, post)
    assert "hill_n" in res
    assert "activity" in res
    assert "lam" in res

def test_extract_parameters_errors(initialized_model_class):
    model = initialized_model_class
    with pytest.raises(ValueError, match="should be a 1-D array-like"):
        extract_parameters(model, {}, q_to_get={"not": "valid"})

def test_extract_parameters_npz(initialized_model_class, tmpdir):
    model = initialized_model_class
    model._transformation = "single"
    model._theta_growth_noise = "zero"
    model._theta_binding_noise = "zero"
    model._condition_growth = "zero"
    model._dk_geno = "zero"
    model._activity = "fixed"
    model.growth_tm.df = pd.DataFrame({"genotype": ["wt"], "titrant_name": ["T"], "titrant_conc": [1.0], "map_theta_group": [0]})
    post = {"theta_hill_n": np.zeros((1, 1)), "theta_log_hill_K": np.zeros((1, 1)),
            "theta_theta_high": np.zeros((1, 1)), "theta_theta_low": np.zeros((1, 1))}
    path = os.path.join(tmpdir, "post.npz")
    np.savez(path, **post)
    res = extract_parameters(model, path)
    assert "hill_n" in res

def test_extract_parameters_categorical(initialized_model_class):
    model = initialized_model_class
    model._theta = "categorical_geno"
    model._condition_growth = "none"
    model._dk_geno = "none"
    model._activity = "fixed"
    model.growth_tm.df = pd.DataFrame({"genotype": ["wt"], "titrant_name": ["T"], "titrant_conc": [1.0], "map_theta": [0]})
    post = {"theta_theta": np.zeros((1, 1))}
    res = extract_parameters(model, post)
    assert "theta" in res

def test_extract_theta_curves_full(initialized_model_class, tmpdir):
    model = initialized_model_class
    model._theta = "hill_geno"
    model.growth_tm.df = pd.DataFrame({"genotype": ["wt"], "titrant_name": ["T"], "titrant_conc": [1.0], "map_theta_group": [0]})
    post = {
        "theta_hill_n": np.zeros((1, 1, 1)), "theta_log_hill_K": np.zeros((1, 1, 1)),
        "theta_theta_high": np.zeros((1, 1, 1)), "theta_theta_low": np.zeros((1, 1, 1))
    }
    res = extract_theta_curves(model, post)
    assert len(res) > 0
    
    # Test path for extract_theta_curves (1047)
    path = os.path.join(tmpdir, "theta.npz")
    np.savez(path, **post)
    res = extract_theta_curves(model, path)
    assert len(res) > 0

def test_extract_theta_curves_manual(initialized_model_class):
    model = initialized_model_class
    model._theta = "hill_geno"
    model.growth_tm.df = pd.DataFrame({"genotype": ["wt"], "titrant_name": ["T"], "map_theta_group": [0]})
    post = {
        "theta_hill_n": np.zeros((1, 1, 1)), "theta_log_hill_K": np.zeros((1, 1, 1)),
        "theta_theta_high": np.zeros((1, 1, 1)), "theta_theta_low": np.zeros((1, 1, 1))
    }
    
    # Test manual_titrant_df (1074-1085)
    manual = pd.DataFrame({"titrant_name": ["T"], "titrant_conc": [0.5]})
    res = extract_theta_curves(model, post, manual_titrant_df=manual)
    assert len(res) == 1
    
    # Test manual_titrant_df with genotype
    manual["genotype"] = "wt"
    res = extract_theta_curves(model, post, manual_titrant_df=manual)
    assert len(res) == 1
    
    # Trigger ValueError by passing a genotype that doesn't exist
    manual["genotype"] = "missing"
    with pytest.raises(ValueError, match="were not found in the model data"):
        extract_theta_curves(model, post, manual_titrant_df=manual)

def test_extract_theta_curves_errors(initialized_model_class):
    model = initialized_model_class
    model._theta = "_simple"
    with pytest.raises(ValueError, match="does not support this interface"):
        extract_theta_curves(model, {})
    model._theta = "hill_geno"
    with pytest.raises(ValueError, match="should be a 1-D array-like"):
        extract_theta_curves(model, {}, q_to_get={"not": "valid"})

def test_extract_growth_predictions_full(initialized_model_class, tmpdir):
    model = initialized_model_class
    model._transformation = "none"
    model.growth_df = pd.DataFrame({
        "replicate": [1], 
        "genotype": ["wt"],
        "condition_pre": ["A"],
        "condition_sel": ["B"],
        "titrant_name": ["T"],
        "titrant_conc": [1.0],
        "t_pre": [1.0],
        "t_sel": [1.0],
        "ln_cfu": [10.0],
        "ln_cfu_std": [0.1],
        "replicate_idx": [0], 
        "time_idx": [0], 
        "condition_pre_idx": [0],
        "condition_sel_idx": [0], 
        "titrant_name_idx": [0], 
        "titrant_conc_idx": [0],
        "genotype_idx": [0],
    })
    post = {"growth_pred": np.zeros((1, 1, 1, 1, 1, 1, 1, 1))}
    res = extract_growth_predictions(model, post)
    assert "q0.5" in res.columns

    with pytest.raises(ValueError, match="'growth_pred' not found"):
        extract_growth_predictions(model, {})
    with pytest.raises(ValueError, match="should be a 1-D array-like"):
        extract_growth_predictions(model, {"growth_pred": None}, q_to_get={"not": "valid"})
        
    path = os.path.join(tmpdir, "gp.npz")
    np.savez(path, growth_pred=np.zeros((1,1,1,1,1,1,1,1)))
    res = extract_growth_predictions(model, path)
    assert len(res) > 0

def test_get_random_idx_logic(initialized_model_class):
    model = initialized_model_class
    model.data = MagicMock()
    model.data.num_genotype = 10
    model.data.num_binding = 1
    model.data.batch_idx = jnp.arange(10)
    model.data.growth.batch_idx = model.data.batch_idx
    model.data.not_binding_idx = jnp.arange(1, 10)
    model.data.not_binding_batch_size = 9
    model._batch_size = 2 # At least num_binding
    
    with pytest.raises(ValueError, match="must be called with an integer batch key"):
        ModelOrchestrator.get_random_idx(model)
        
    res = ModelOrchestrator.get_random_idx(model, batch_key=42)
    assert res.shape == (2,)
    
    # num_batches > 1 (1351-1360)
    res = ModelOrchestrator.get_random_idx(model, num_batches=2)
    assert res.shape == (2, 2)


# ----------------------------------------------------------------------------
# Tests for binding_weight
# ----------------------------------------------------------------------------

_SETUP_BATCHING_RETURN = {
    "batch_idx": np.array([0, 1]),
    "batch_size": 2,
    "scale_vector": np.array([1.0, 1.0]),
    "num_binding": 1,
    "not_binding_idx": np.array([1]),
    "not_binding_batch_size": 1,
}

_BINDING_WEIGHT_PATCHES = [
    "tfscreen.tfmodel.model_orchestrator._read_growth_df",
    "tfscreen.tfmodel.model_orchestrator._read_binding_df",
]


def _make_model_for_binding_weight(mocker, n_growth, n_binding, binding_weight=None):
    """
    Build a ModelOrchestrator whose growth_tm.df has n_growth rows and
    binding_tm.df has n_binding rows.  Everything else is mocked out.
    """
    mock_growth_tm = create_mock_tm(is_growth=True)
    mock_growth_tm.df = pd.DataFrame({"x": range(n_growth)})

    mock_binding_tm = create_mock_tm(is_growth=False)
    mock_binding_tm.df = pd.DataFrame({"x": range(n_binding)})

    for path in _BINDING_WEIGHT_PATCHES:
        mocker.patch(path)
    mocker.patch(
        "tfscreen.tfmodel.model_orchestrator._build_growth_tm",
        return_value=mock_growth_tm,
    )
    mocker.patch(
        "tfscreen.tfmodel.model_orchestrator._build_binding_tm",
        return_value=mock_binding_tm,
    )
    mocker.patch(
        "tfscreen.tfmodel.model_orchestrator._setup_batching",
        return_value=_SETUP_BATCHING_RETURN.copy(),
    )
    mocker.patch(
        "tfscreen.tfmodel.model_orchestrator.populate_dataclass",
        return_value=MagicMock(),
    )
    mocker.patch(
        "tfscreen.tfmodel.model_orchestrator.ModelOrchestrator._initialize_classes"
    )

    return ModelOrchestrator("g.csv", "b.csv", binding_weight=binding_weight)


def test_binding_weight_auto_computed(mocker):
    """binding_weight=None should be resolved to N_growth / N_binding after init."""
    n_growth, n_binding = 1000, 10
    model = _make_model_for_binding_weight(mocker, n_growth, n_binding)
    assert model._binding_weight == pytest.approx(n_growth / n_binding)


def test_binding_weight_explicit_preserved(mocker):
    """An explicit binding_weight should be stored unchanged."""
    model = _make_model_for_binding_weight(mocker, 1000, 10, binding_weight=42.5)
    assert model._binding_weight == pytest.approx(42.5)


def test_binding_weight_explicit_in_settings(mocker):
    """binding_weight should appear in settings with the resolved value."""
    model = _make_model_for_binding_weight(mocker, 200, 5, binding_weight=7.0)
    model._binding_only = False
    model._condition_growth = "linear"
    model._growth_transition = "instant"
    model._ln_cfu0 = "hierarchical"
    model._dk_geno = "hierarchical_geno"
    model._activity = "horseshoe_geno"
    model._theta = "hill_geno"
    model._transformation = "empirical"
    model._theta_rescale = "passthrough"
    model._theta_growth_noise = "zero"
    model._theta_binding_noise = "zero"
    model._spiked_genotypes = None
    model._growth_shares_replicates = False
    model._epistasis = False
    model._thermo_data = None
    model._batch_size = None
    s = ModelOrchestrator.settings.fget(model)
    assert "binding_weight" in s
    assert s["binding_weight"] == pytest.approx(7.0)


def test_binding_weight_auto_in_settings(mocker):
    """Auto-computed weight (not None) should appear in settings after init."""
    n_growth, n_binding = 500, 25
    model = _make_model_for_binding_weight(mocker, n_growth, n_binding)
    model._binding_only = False
    model._condition_growth = "linear"
    model._growth_transition = "instant"
    model._ln_cfu0 = "hierarchical"
    model._dk_geno = "hierarchical_geno"
    model._activity = "horseshoe_geno"
    model._theta = "hill_geno"
    model._transformation = "empirical"
    model._theta_rescale = "passthrough"
    model._theta_growth_noise = "zero"
    model._theta_binding_noise = "zero"
    model._spiked_genotypes = None
    model._growth_shares_replicates = False
    model._epistasis = False
    model._thermo_data = None
    model._batch_size = None
    s = ModelOrchestrator.settings.fget(model)
    assert s["binding_weight"] == pytest.approx(n_growth / n_binding)


def test_binding_weight_applied_to_scale_vector(mocker):
    """
    The scale_vector passed to BindingData should be the original value
    multiplied by binding_weight.
    """
    from tfscreen.tfmodel.data_class import BindingData

    explicit_weight = 50.0
    base_scale = np.array([1.0])  # scale for 1 binding genotype

    batching_return = _SETUP_BATCHING_RETURN.copy()
    batching_return["scale_vector"] = np.array([1.0, 1.0])  # index 0 = binding
    batching_return["num_binding"] = 1

    mock_growth_tm = create_mock_tm(is_growth=True)
    mock_growth_tm.df = pd.DataFrame({"x": range(500)})
    mock_binding_tm = create_mock_tm(is_growth=False)
    mock_binding_tm.df = pd.DataFrame({"x": range(10)})

    for path in _BINDING_WEIGHT_PATCHES:
        mocker.patch(path)
    mocker.patch(
        "tfscreen.tfmodel.model_orchestrator._build_growth_tm",
        return_value=mock_growth_tm,
    )
    mocker.patch(
        "tfscreen.tfmodel.model_orchestrator._build_binding_tm",
        return_value=mock_binding_tm,
    )
    mocker.patch(
        "tfscreen.tfmodel.model_orchestrator._setup_batching",
        return_value=batching_return,
    )
    mocker.patch(
        "tfscreen.tfmodel.model_orchestrator.ModelOrchestrator._initialize_classes"
    )

    captured = {}

    def capture_populate(cls, sources):
        if cls is BindingData:
            for src in sources:
                if isinstance(src, dict) and "scale_vector" in src:
                    captured["scale_vector"] = np.array(src["scale_vector"])
        return MagicMock()

    mocker.patch(
        "tfscreen.tfmodel.model_orchestrator.populate_dataclass",
        side_effect=capture_populate,
    )

    ModelOrchestrator("g.csv", "b.csv", binding_weight=explicit_weight)

    assert "scale_vector" in captured, "BindingData populate_dataclass was never called"
    expected = base_scale * explicit_weight
    np.testing.assert_allclose(captured["scale_vector"], expected)


# ---------------------------------------------------------------------------
# Tests for condition-label extraction logic in _setup_model
# ---------------------------------------------------------------------------

class TestConditionLabelExtraction:
    """
    Verify that _setup_model correctly extracts ordered condition labels from
    growth_tm and routes them to get_priors().

    These tests exercise the extraction logic directly without needing to
    build a full ModelOrchestrator, by calling the relevant code path via
    a stripped-down mock orchestrator.
    """

    def _make_cond_rep_df(self, names, indices):
        """Return a condition_rep map_groups DataFrame."""
        return pd.DataFrame({
            "condition_rep": names,
            "map_condition_rep": indices,
        })

    def test_labels_extracted_in_index_order(self):
        """
        Labels must be sorted by map_condition_rep so parameter positions
        match the ordering established by the TensorManager.
        """
        # Rows are intentionally NOT in index order
        df = self._make_cond_rep_df(
            names=["pheS+4CP", "pheS-4CP", "kanR+kan", "kanR-kan"],
            indices=[2, 3, 0, 1],
        )
        result = list(df.sort_values("map_condition_rep")["condition_rep"])
        assert result == ["kanR+kan", "kanR-kan", "pheS+4CP", "pheS-4CP"]

    def test_labels_correct_when_already_ordered(self):
        df = self._make_cond_rep_df(
            names=["kanR+kan", "kanR-kan", "pheS+4CP", "pheS-4CP"],
            indices=[0, 1, 2, 3],
        )
        result = list(df.sort_values("map_condition_rep")["condition_rep"])
        assert result == ["kanR+kan", "kanR-kan", "pheS+4CP", "pheS-4CP"]

    def test_single_condition(self):
        df = self._make_cond_rep_df(["sel+media"], [0])
        result = list(df.sort_values("map_condition_rep")["condition_rep"])
        assert result == ["sel+media"]

    def test_shares_replicates_false_includes_replicate_but_still_sorts(self):
        """When shares_replicates=False the df has (replicate, condition_rep) rows."""
        df = pd.DataFrame({
            "replicate": [1, 2, 1, 2],
            "condition_rep": ["pheS+4CP", "pheS+4CP", "pheS-4CP", "pheS-4CP"],
            "map_condition_rep": [1, 3, 0, 2],
        })
        result = list(df.sort_values("map_condition_rep")["condition_rep"])
        # Sorted by index: 0=pheS-4CP, 1=pheS+4CP, 2=pheS-4CP, 3=pheS+4CP
        assert result[0] == "pheS-4CP"
        assert result[1] == "pheS+4CP"

    def test_linear_get_priors_signature_has_is_selection(self):
        """
        The orchestrator detects is_selection support via inspect.signature.
        Verify that the linear component's get_priors has this parameter
        (preferred path) and still retains condition_labels for legacy compat.
        """
        import inspect
        from tfscreen.tfmodel.generative.components.growth.linear import get_priors
        sig = inspect.signature(get_priors)
        assert "is_selection" in sig.parameters, (
            "linear.get_priors must have an 'is_selection' parameter so "
            "ModelOrchestrator can pass inferred selection flags directly"
        )
        assert "condition_labels" in sig.parameters, (
            "linear.get_priors must retain 'condition_labels' for backward compat"
        )

    def test_setup_model_extracts_is_selection_and_calls_get_priors(self, mocker):
        """
        _initialize_classes must call get_priors(is_selection=...) using
        flags inferred from the data structure, not from parsing condition
        name strings.  kanR+kan only in condition_sel → True; kanR-kan in
        condition_pre → False.
        """
        import inspect
        import tfscreen.tfmodel.model_orchestrator as mo
        from tfscreen.tfmodel.generative.components.growth import linear as real_linear

        captured = {}

        def capturing_get_priors(condition_labels=None, is_selection=None):
            captured["is_selection"] = is_selection
            captured["condition_labels"] = condition_labels
            return real_linear.get_priors(is_selection=is_selection,
                                          condition_labels=condition_labels)

        # cond_rep_df as produced after Phase 1 (library column present)
        cond_rep_df = pd.DataFrame({
            "library": ["lib", "lib"],
            "condition_rep": ["kanR+kan", "kanR-kan"],
            "map_condition_rep": [0, 1],
        })
        mock_growth_tm = MagicMock()
        mock_growth_tm.map_groups = {"condition_rep": cond_rep_df}

        # Minimal growth_df so _infer_is_selection can derive flags:
        # kanR-kan in condition_pre → False; kanR+kan only in condition_sel → True
        growth_df = pd.DataFrame({
            "library": ["lib", "lib"],
            "condition_pre": ["kanR-kan", "kanR-kan"],
            "condition_sel": ["kanR+kan", "kanR-kan"],
        })

        # Stub ALL registry entries except we intercept condition_growth.get_priors
        all_stubs = {}
        for comp_key, comp_val in mo.model_registry.items():
            if not isinstance(comp_val, dict):
                continue
            all_stubs[comp_key] = {}
            for variant_name, variant_mod in comp_val.items():
                stub = MagicMock()
                stub.get_priors.return_value = MagicMock()
                stub.get_guesses.return_value = {}
                stub.define_model = MagicMock()
                stub.guide = MagicMock()
                stub.calculate_growth = MagicMock()
                stub.rescale = MagicMock()
                stub.observe = MagicMock()
                all_stubs[comp_key][variant_name] = stub

        # Override condition_growth / linear with our spy
        all_stubs.setdefault("condition_growth", {})["linear"] = MagicMock(
            get_priors=capturing_get_priors,
            get_guesses=MagicMock(return_value={}),
            define_model=MagicMock(),
            guide=MagicMock(),
            calculate_growth=MagicMock(),
        )

        orchestrator = MagicMock(spec=ModelOrchestrator)
        orchestrator.growth_tm = mock_growth_tm
        orchestrator.growth_df = growth_df
        orchestrator._data = MagicMock()
        orchestrator._binding_only = False
        orchestrator._batch_size = None
        orchestrator._condition_growth = "linear"
        orchestrator._growth_transition = "instant"
        orchestrator._ln_cfu0 = "hierarchical"
        orchestrator._dk_geno = "fixed"
        orchestrator._activity = "fixed"
        orchestrator._theta = "hill_geno"
        orchestrator._transformation = "single"
        orchestrator._theta_growth_noise = "zero"
        orchestrator._theta_binding_noise = "zero"
        orchestrator._growth_noise = "zero"
        orchestrator._sample_offset = "zero"
        orchestrator._theta_rescale = "passthrough"

        with patch.object(mo, "model_registry", all_stubs):
            try:
                mo.ModelOrchestrator._initialize_classes(orchestrator)
            except Exception:
                pass  # only is_selection matters

        assert "is_selection" in captured, (
            "_initialize_classes did not pass is_selection to get_priors"
        )
        # kanR+kan at index 0 → selective (True); kanR-kan at index 1 → not selective (False)
        assert captured["is_selection"] == [True, False], (
            f"Expected [True, False], got {captured['is_selection']}"
        )
        assert captured["condition_labels"] is None


# ---------------------------------------------------------------------------
# Sentinel tests: genotype ordering must survive the data pipeline
# ---------------------------------------------------------------------------

class TestGenotypeOrderSentinel:
    """
    Sentinel tests verifying that genotype ordering is maintained as GENETIC
    order (wt first, then by residue number) through the full data pipeline.

    Uses a deliberately pathological genotype set where:
      insertion order:    A10G, wt, A2G
      alphabetical order: A10G, A2G, wt
      genetic order:      wt, A2G, A10G   <- required

    Any operation that loses the CategoricalDtype and re-creates it via
    pd.Categorical() would produce alphabetical order instead, giving wrong
    genotype_idx assignments and silently scrambling parameters.
    """

    @staticmethod
    def _build_raw_df(libraries=("lib",), condition_blocks=(("no-kan", "kan"),)):
        """Minimal growth_df with pathological genotype insertion order."""
        rows = []
        for lib in libraries:
            for geno in ["A10G", "wt", "A2G"]:   # non-alpha, non-genetic order
                for cp, cs in condition_blocks:
                    rows.append({
                        "genotype": geno,
                        "library": lib,
                        "replicate": 1,
                        "titrant_name": "iptg",
                        "titrant_conc": 0.0,
                        "condition_pre": cp,
                        "condition_sel": cs,
                        "t_pre": 30.0,
                        "t_sel": 100.0,
                        "ln_cfu": 10.0,
                        "ln_cfu_std": 0.1,
                    })
        return pd.DataFrame(rows)

    def test_read_growth_df_produces_genetic_categorical_order(self):
        """After _read_growth_df, genotype categories must be in genetic order."""
        result = _read_growth_df(self._build_raw_df())
        cats = list(result["genotype"].cat.categories)
        assert cats == ["wt", "A2G", "A10G"], (
            f"Expected ['wt','A2G','A10G'] (genetic), got {cats}. "
            "Alphabetical would be ['A10G','A2G','wt'] — categorical dtype "
            "was probably lost and re-created by pd.Categorical()."
        )

    def test_read_growth_df_genotype_stays_categorical(self):
        """Categorical dtype on genotype must survive all ops in _read_growth_df."""
        result = _read_growth_df(self._build_raw_df())
        assert isinstance(result["genotype"].dtype, pd.CategoricalDtype), (
            "genotype lost CategoricalDtype during _read_growth_df. "
            "Any subsequent pd.Categorical() call produces alphabetical order."
        )

    def test_read_growth_df_row_order_unchanged(self):
        """Row insertion order must not change inside _read_growth_df."""
        df = self._build_raw_df()
        expected = list(df["genotype"])
        result = _read_growth_df(df)
        assert list(result["genotype"]) == expected

    def test_build_growth_tm_tensor_genotype_axis_genetic_order(self):
        """
        After _build_growth_tm the genotype tensor dimension must be in
        genetic order, not alphabetical or insertion order.
        """
        df = self._build_raw_df()
        growth_df = _read_growth_df(df)
        tm = _build_growth_tm(growth_df)

        geno_dim = tm.tensor_dim_names.index("genotype")
        assert list(tm.tensor_dim_labels[geno_dim]) == ["wt", "A2G", "A10G"], (
            f"Tensor genotype axis: {list(tm.tensor_dim_labels[geno_dim])}"
        )

    def test_genotype_idx_consistent_with_tensor_labels(self):
        """
        genotype_idx in the DataFrame must match each genotype's position
        in the tensor genotype dimension.  Misalignment causes parameters
        for genotype X to be silently applied to genotype Y's data.
        """
        df = self._build_raw_df()
        growth_df = _read_growth_df(df)
        tm = _build_growth_tm(growth_df)

        geno_dim = tm.tensor_dim_names.index("genotype")
        label_to_pos = {str(g): i for i, g in enumerate(tm.tensor_dim_labels[geno_dim])}

        for _, row in tm.df.iterrows():
            geno = str(row["genotype"])
            idx = int(row["genotype_idx"])
            expected = label_to_pos[geno]
            assert idx == expected, (
                f"Genotype '{geno}': genotype_idx={idx} but "
                f"tensor label position={expected}."
            )

    def test_condition_sel_reduced_does_not_reorder_rows(self):
        """
        The condition_sel_reduced groupby in _read_growth_df adds a column
        but must not change row order (which would shift genotype_idx assignments).
        """
        df = self._build_raw_df(
            condition_blocks=[("no-kan", "kan"), ("no-kan", "no-kan")]
        )
        expected_genos = list(df["genotype"])
        result = _read_growth_df(df)
        assert list(result["genotype"]) == expected_genos, (
            "Row order changed during condition_sel_reduced computation."
        )

    def test_genotype_categorical_survives_condition_sel_reduced(self):
        """
        CategoricalDtype must survive the condition_sel_reduced groupby/map.
        If dropped and re-created, genetic order is replaced by alphabetical.
        """
        df = self._build_raw_df(
            condition_blocks=[("no-kan", "kan"), ("no-kan", "no-kan")]
        )
        result = _read_growth_df(df)
        assert isinstance(result["genotype"].dtype, pd.CategoricalDtype), (
            "genotype lost CategoricalDtype after condition_sel_reduced was computed."
        )


# ---------------------------------------------------------------------------
# Phase 1 tests: library as a proper grouping key
# ---------------------------------------------------------------------------

class TestLibraryGroupingPhase1:
    """
    Tests for Phase 1 of the library-column refactor.  Library must be added
    to the condition_sel_reduced groupby and to the ln_cfu0 map tensor.

    Tests marked xfail(strict=True) document the desired new behavior and
    will fail until the refactor is implemented.  Remove the marker after
    each test starts passing.

    The regression guard (single-library unchanged) should pass now and
    continue to pass after the refactor.
    """

    @staticmethod
    def _two_library_df(shared_condition_sel="kan"):
        """Two libraries with identical condition names but distinct ln_cfu."""
        rows = []
        for lib, base in [("libA", 10.0), ("libB", 20.0)]:
            for geno in ["wt", "A2G"]:
                rows.append({
                    "genotype": geno,
                    "library": lib,
                    "replicate": 1,
                    "titrant_name": "iptg",
                    "titrant_conc": 0.0,
                    "condition_pre": "no-kan",
                    "condition_sel": shared_condition_sel,
                    "t_pre": 30.0,
                    "t_sel": 100.0,
                    "ln_cfu": base,
                    "ln_cfu_std": 0.1,
                })
        return pd.DataFrame(rows)

    def test_single_library_regression(self):
        """
        Single-library behavior must be unchanged after Phase 1.
        Genotype order must still be genetic.
        """
        df = TestGenotypeOrderSentinel._build_raw_df()
        growth_df = _read_growth_df(df)
        tm = _build_growth_tm(growth_df)

        geno_dim = tm.tensor_dim_names.index("genotype")
        assert list(tm.tensor_dim_labels[geno_dim]) == ["wt", "A2G", "A10G"]

    def test_two_libraries_genotype_order_preserved(self):
        """
        Adding a second library must not change genotype ordering — both
        libraries share the same genotype set and must be in genetic order.
        """
        df = self._two_library_df()
        growth_df = _read_growth_df(df)
        tm = _build_growth_tm(growth_df)

        geno_dim = tm.tensor_dim_names.index("genotype")
        assert list(tm.tensor_dim_labels[geno_dim]) == ["wt", "A2G"], (
            f"Unexpected genotype order: {list(tm.tensor_dim_labels[geno_dim])}"
        )

    def test_two_libraries_have_distinct_ln_cfu0_slots(self):
        """
        Same genotype in two different libraries must have different ln_cfu0
        parameter slots.  After Phase 1, 'library' is added to the ln_cfu0
        map tensor so (libA, rep1, no-kan, wt) and (libB, rep1, no-kan, wt)
        receive different indices.
        """
        df = self._two_library_df()
        growth_df = _read_growth_df(df)
        tm = _build_growth_tm(growth_df)

        libA_wt = int(
            tm.df[(tm.df["genotype"] == "wt") & (tm.df["library"] == "libA")]
            ["map_ln_cfu0"].iloc[0]
        )
        libB_wt = int(
            tm.df[(tm.df["genotype"] == "wt") & (tm.df["library"] == "libB")]
            ["map_ln_cfu0"].iloc[0]
        )
        assert libA_wt != libB_wt, (
            "libA/wt and libB/wt share ln_cfu0 slot — library is not yet "
            "included in the ln_cfu0 map tensor."
        )

    def test_condition_sel_reduced_is_per_library(self):
        """
        condition_sel_reduced numbering must be computed within each
        (library, condition_pre) group so that each library's first unique
        condition_sel gets index 0.

        Currently, the mapper is built globally per condition_pre, so a
        condition that only appears in libB gets a non-zero index because
        libA's conditions are numbered first.  After Phase 1, each
        (library, condition_pre) group numbers independently from 0.
        """
        rows = []
        # libA: two distinct condition_sel values under no-kan
        for cs in ["sel-A", "ctrl"]:
            rows.append({
                "genotype": "wt", "library": "libA", "replicate": 1,
                "titrant_name": "iptg", "titrant_conc": 0.0,
                "condition_pre": "no-kan", "condition_sel": cs,
                "t_pre": 30.0, "t_sel": 100.0, "ln_cfu": 10.0, "ln_cfu_std": 0.1,
            })
        # libB: one condition_sel unique to it (not in libA)
        rows.append({
            "genotype": "wt", "library": "libB", "replicate": 1,
            "titrant_name": "iptg", "titrant_conc": 0.0,
            "condition_pre": "no-kan", "condition_sel": "sel-B",
            "t_pre": 30.0, "t_sel": 100.0, "ln_cfu": 10.0, "ln_cfu_std": 0.1,
        })
        df = pd.DataFrame(rows)
        growth_df = _read_growth_df(df)

        # Before Phase 1: global mapper gives "sel-B" index 2 (after sel-A=0, ctrl=1).
        # After Phase 1: per-library mapper gives "sel-B" index 0 within its own group.
        libB_sel_b_reduced = int(
            growth_df[
                (growth_df["library"] == "libB") & (growth_df["condition_sel"] == "sel-B")
            ]["condition_sel_reduced"].iloc[0]
        )
        assert libB_sel_b_reduced == 0, (
            f"libB/'sel-B' got condition_sel_reduced={libB_sel_b_reduced}, expected 0. "
            "The global mapper assigns higher indices to conditions that appear "
            "after other libraries' conditions — library is not yet in the groupby."
        )


# ---------------------------------------------------------------------------
# Phase 2 tests: is_selection inference from data structure
# ---------------------------------------------------------------------------

class TestIsSelectionInference:
    """
    Tests for Phase 2: inferring which conditions are selective based on
    data structure.  The rule:
      condition only in condition_sel (never in condition_pre) → selective
      condition that appears in condition_pre → not selective

    All tests are xfail until _infer_is_selection is implemented in
    model_orchestrator.py.  The ImportError that results from the missing
    function counts as the expected failure under xfail(strict=True).
    Remove the marker once each test passes.
    """

    @staticmethod
    def _build_df(blocks, library="lib"):
        """Build a minimal growth_df from (condition_pre, condition_sel) pairs."""
        rows = []
        for cp, cs in blocks:
            rows.append({
                "genotype": "wt",
                "library": library,
                "replicate": 1,
                "titrant_name": "iptg",
                "titrant_conc": 0.0,
                "condition_pre": cp,
                "condition_sel": cs,
                "t_pre": 30.0,
                "t_sel": 100.0,
                "ln_cfu": 10.0,
                "ln_cfu_std": 0.1,
            })
        return pd.DataFrame(rows)

    def test_standard_case(self):
        """
        Condition only in condition_pre → False.
        Condition only in condition_sel → True.
        """
        from tfscreen.tfmodel.model_orchestrator import _infer_is_selection
        df = self._build_df([
            ("no-kan", "kan"),    # no-kan in pre → False; kan only in sel → True
            ("no-kan", "no-kan"), # control arm: no-kan appears in pre → still False
        ])
        result = _infer_is_selection(df)
        assert result["no-kan"] is False
        assert result["kan"] is True

    def test_control_arm_same_condition_in_pre_and_sel(self):
        """condition_pre == condition_sel → that condition is non-selective."""
        from tfscreen.tfmodel.model_orchestrator import _infer_is_selection
        df = self._build_df([
            ("LB", "sel"),
            ("LB", "LB"),   # control arm
        ])
        result = _infer_is_selection(df)
        assert result["LB"] is False
        assert result["sel"] is True

    def test_multiple_null_conditions_both_false(self):
        """Two distinct non-selective conditions must independently get False."""
        from tfscreen.tfmodel.model_orchestrator import _infer_is_selection
        df = self._build_df([
            ("LB", "kan"),
            ("M9", "kan"),   # second distinct null medium
        ])
        result = _infer_is_selection(df)
        assert result["LB"] is False
        assert result["M9"] is False
        assert result["kan"] is True

    def test_per_library_same_name_different_role(self):
        """
        Same condition name appearing in condition_pre for libA but only in
        condition_sel for libB gets different is_selection per library.
        """
        from tfscreen.tfmodel.model_orchestrator import _infer_is_selection
        rows = []
        # libA: "shared" is used as pre-growth (non-selective)
        rows.append({"genotype": "wt", "library": "libA",
                     "condition_pre": "shared", "condition_sel": "sel",
                     "replicate": 1, "titrant_name": "iptg", "titrant_conc": 0.0,
                     "t_pre": 30.0, "t_sel": 100.0, "ln_cfu": 10.0, "ln_cfu_std": 0.1})
        # libB: "shared" only appears in condition_sel (selective)
        rows.append({"genotype": "wt", "library": "libB",
                     "condition_pre": "LB", "condition_sel": "shared",
                     "replicate": 1, "titrant_name": "iptg", "titrant_conc": 0.0,
                     "t_pre": 30.0, "t_sel": 100.0, "ln_cfu": 10.0, "ln_cfu_std": 0.1})
        df = pd.DataFrame(rows)
        result = _infer_is_selection(df, per_library=True)
        assert result[("libA", "shared")] is False
        assert result[("libB", "shared")] is True

    def test_inferred_agrees_with_legacy_parse_for_plus_minus_names(self):
        """
        For condition names with standard +/- notation, the new inference rule
        must agree with the legacy _parse_condition_label() parser.
        """
        from tfscreen.tfmodel.model_orchestrator import _infer_is_selection
        df = self._build_df([
            ("kanR-kan", "kanR+kan"),   # - in pre (control), + in sel (selection)
            ("kanR-kan", "kanR-kan"),   # control arm
        ])
        result = _infer_is_selection(df)
        for cond, is_sel in result.items():
            legacy = _parse_condition_label(cond)
            assert is_sel is legacy, (
                f"Inference disagreed with legacy parser for '{cond}': "
                f"inferred={is_sel}, legacy={legacy}"
            )


# ---------------------------------------------------------------------------
# Combined library/condition_pre key tests
# ---------------------------------------------------------------------------

class TestCombinedConditionPreKey:
    """
    Tests for the internal `_condition_pre_key` = library + "/" + condition_pre
    column used inside _build_growth_tm and _build_presplit_tm.

    The combined key ensures that two libraries sharing the same condition_pre
    name get DISTINCT tensor slots, preventing silent cross-library pooling.
    The '/' separator is never split on downstream — the key is opaque.
    """

    @staticmethod
    def _growth_df(libraries_conditions):
        """
        Build a minimal growth_df.

        Parameters
        ----------
        libraries_conditions : list of (library, condition_pre, condition_sel)
        """
        rows = []
        for lib, cp, cs in libraries_conditions:
            rows.append({
                "library": lib, "replicate": 1,
                "genotype": "wt", "titrant_name": "iptg", "titrant_conc": 0.0,
                "condition_pre": cp, "condition_sel": cs,
                "t_pre": 30.0, "t_sel": 100.0,
                "ln_cfu": 10.0, "ln_cfu_std": 0.1,
            })
        return pd.DataFrame(rows)

    def test_single_library_combined_key_formed(self):
        """_condition_pre_key = library/condition_pre is added to growth_tm.df."""
        df = self._growth_df([("kanR", "-kan", "+kan")])
        growth_df = _read_growth_df(df)
        tm = _build_growth_tm(growth_df)

        assert "_condition_pre_key" in tm.df.columns
        assert tm.df["_condition_pre_key"].iloc[0] == "kanR/-kan"

    def test_single_library_num_condition_pre_unchanged(self):
        """Single library with one condition_pre → num_condition_pre == 1."""
        df = self._growth_df([("kanR", "-kan", "+kan")])
        growth_df = _read_growth_df(df)
        tm = _build_growth_tm(growth_df)

        cp_dim = tm.tensor_dim_names.index("condition_pre")
        assert len(tm.tensor_dim_labels[cp_dim]) == 1

    def test_two_libraries_shared_condition_pre_distinct_tensor_slots(self):
        """
        Two libraries both using condition_pre='-kan' must occupy DIFFERENT
        condition_pre tensor slots.  Before the combined key this collapsed
        them into one slot, silently pooling unrelated data.
        """
        df = self._growth_df([
            ("libA", "-kan", "+kan"),
            ("libB", "-kan", "+kan"),   # same condition_pre, different library
        ])
        growth_df = _read_growth_df(df)
        tm = _build_growth_tm(growth_df)

        cp_dim = tm.tensor_dim_names.index("condition_pre")
        assert len(tm.tensor_dim_labels[cp_dim]) == 2, (
            "Two libraries sharing condition_pre='-kan' should produce "
            "two distinct condition_pre tensor slots ('libA/-kan' and 'libB/-kan'), "
            "not one collapsed slot."
        )

    def test_two_libraries_shared_condition_pre_labels_contain_library(self):
        """Tensor condition_pre labels carry the library prefix."""
        df = self._growth_df([
            ("libA", "-kan", "+kan"),
            ("libB", "-kan", "+kan"),
        ])
        growth_df = _read_growth_df(df)
        tm = _build_growth_tm(growth_df)

        cp_dim = tm.tensor_dim_names.index("condition_pre")
        labels = set(tm.tensor_dim_labels[cp_dim])
        assert "libA/-kan" in labels
        assert "libB/-kan" in labels

    def test_two_libraries_shared_condition_pre_data_not_pooled(self):
        """
        Data from two libraries sharing condition_pre must land in separate
        tensor positions.  The condition_pre_idx in tm.df must differ between
        the two libraries.
        """
        df = pd.DataFrame([
            {"library": "libA", "replicate": 1, "genotype": "wt",
             "titrant_name": "iptg", "titrant_conc": 0.0,
             "condition_pre": "-kan", "condition_sel": "+kan",
             "t_pre": 30.0, "t_sel": 100.0, "ln_cfu": 10.0, "ln_cfu_std": 0.1},
            {"library": "libB", "replicate": 1, "genotype": "wt",
             "titrant_name": "iptg", "titrant_conc": 0.0,
             "condition_pre": "-kan", "condition_sel": "+kan",
             "t_pre": 30.0, "t_sel": 100.0, "ln_cfu": 20.0, "ln_cfu_std": 0.1},
        ])
        growth_df = _read_growth_df(df)
        tm = _build_growth_tm(growth_df)

        idx_a = int(tm.df[tm.df["library"] == "libA"]["condition_pre_idx"].iloc[0])
        idx_b = int(tm.df[tm.df["library"] == "libB"]["condition_pre_idx"].iloc[0])
        assert idx_a != idx_b, (
            f"libA and libB both got condition_pre_idx={idx_a} — "
            "their data would overwrite each other in the tensor."
        )

    def test_distinct_condition_pre_names_not_affected(self):
        """
        Libraries with distinct condition_pre names keep num_condition_pre
        equal to the number of unique (library, condition_pre) pairs —
        same behaviour as before.
        """
        df = self._growth_df([
            ("kanR", "-kan", "+kan"),
            ("pheS", "-4CP", "+4CP"),  # different names — no collision risk
        ])
        growth_df = _read_growth_df(df)
        tm = _build_growth_tm(growth_df)

        cp_dim = tm.tensor_dim_names.index("condition_pre")
        assert len(tm.tensor_dim_labels[cp_dim]) == 2

    def test_slash_in_condition_name_is_safe(self):
        """
        A condition_pre name that itself contains '/' produces a valid combined
        key without any parsing ambiguity — the key is treated as opaque.
        """
        df = self._growth_df([("lib", "no/sel", "+sel")])
        growth_df = _read_growth_df(df)
        tm = _build_growth_tm(growth_df)

        cp_dim = tm.tensor_dim_names.index("condition_pre")
        labels = list(tm.tensor_dim_labels[cp_dim])
        assert labels == ["lib/no/sel"]

    def test_presplit_combined_key_matches_growth_tm_categories(self):
        """
        _build_presplit_tm must use the same combined condition_pre categories
        as growth_tm so that the presplit condition_pre_idx values are
        consistent with those used in the growth tensor.
        """
        rows_growth = []
        rows_presplit = []
        for lib, cp, cs in [("libA", "-kan", "+kan"), ("libB", "-kan", "+kan")]:
            for geno in ["wt", "A2G"]:
                rows_growth.append({
                    "library": lib, "replicate": 1, "genotype": geno,
                    "titrant_name": "iptg", "titrant_conc": 0.0,
                    "condition_pre": cp, "condition_sel": cs,
                    "t_pre": 30.0, "t_sel": 100.0,
                    "ln_cfu": 10.0, "ln_cfu_std": 0.1,
                })
                rows_presplit.append({
                    "library": lib, "replicate": 1, "genotype": geno,
                    "condition_pre": cp,
                    "ln_cfu": 9.5, "ln_cfu_std": 0.1,
                })

        growth_df_raw = pd.DataFrame(rows_growth)
        presplit_df_raw = pd.DataFrame(rows_presplit)

        from tfscreen.tfmodel.model_orchestrator import _build_presplit_tm, _read_presplit_df
        growth_df = _read_growth_df(growth_df_raw)
        tm_growth = _build_growth_tm(growth_df)
        presplit_df = _read_presplit_df(presplit_df_raw, growth_df)
        tm_presplit = _build_presplit_tm(presplit_df, tm_growth)

        # Both tensors must share the same condition_pre dimension labels
        cp_growth   = list(tm_growth.tensor_dim_labels[
            tm_growth.tensor_dim_names.index("condition_pre")])
        cp_presplit = list(tm_presplit.tensor_dim_labels[
            tm_presplit.tensor_dim_names.index("condition_pre")])
        assert cp_growth == cp_presplit, (
            f"growth condition_pre labels: {cp_growth}\n"
            f"presplit condition_pre labels: {cp_presplit}"
        )

# ---------------------------------------------------------------------------
# base_growth_df: _read_base_growth_df / _build_base_growth_arrays /
# _derive_k_ref_guess
# ---------------------------------------------------------------------------

class TestBaseGrowthDf:
    """
    Tests for the base_growth_df dk_geno-calibration input: a direct,
    uncertainty-weighted observation of growth rate tied to a new k_ref
    scalar plus the existing per-genotype dk_geno latent (see model.py's
    base_growth_obs block).
    """

    @staticmethod
    def _growth_df(genotypes=("wt", "A2G", "K3R")):
        rows = []
        for geno in genotypes:
            rows.append({
                "library": "kanR", "replicate": 1, "genotype": geno,
                "titrant_name": "iptg", "titrant_conc": 0.0,
                "condition_pre": "-kan", "condition_sel": "+kan",
                "t_pre": 30.0, "t_sel": 100.0,
                "ln_cfu": 10.0, "ln_cfu_std": 0.1,
            })
        return pd.DataFrame(rows)

    def test_basic_construction(self):
        from tfscreen.tfmodel.model_orchestrator import _read_base_growth_df

        growth_df = _read_growth_df(self._growth_df())
        base_growth_df_raw = pd.DataFrame({
            "genotype": ["wt", "A2G"],
            "rate": [0.02, 0.015],
            "rate_std": [0.001, 0.002],
        })

        result = _read_base_growth_df(base_growth_df_raw, growth_df)
        assert set(result["genotype"]) == {"wt", "A2G"}
        assert len(result) == 2

    def test_missing_required_column_raises(self):
        from tfscreen.tfmodel.model_orchestrator import _read_base_growth_df

        growth_df = _read_growth_df(self._growth_df())
        bad_df = pd.DataFrame({"genotype": ["wt"], "rate": [0.02]})  # no rate_std

        with pytest.raises(ValueError):
            _read_base_growth_df(bad_df, growth_df)

    def test_genotype_not_in_growth_df_is_dropped(self, capsys):
        from tfscreen.tfmodel.model_orchestrator import _read_base_growth_df

        growth_df = _read_growth_df(self._growth_df())
        # A5T is validly-formatted but not one of the genotypes in growth_df.
        base_growth_df_raw = pd.DataFrame({
            "genotype": ["wt", "A5T"],
            "rate": [0.02, 0.05],
            "rate_std": [0.001, 0.001],
        })

        result = _read_base_growth_df(base_growth_df_raw, growth_df)
        assert set(result["genotype"]) == {"wt"}
        captured = capsys.readouterr()
        assert "Syncing base_growth_df to growth_df" in captured.out

    def test_missing_wt_after_filtering_raises(self):
        from tfscreen.tfmodel.model_orchestrator import _read_base_growth_df

        growth_df = _read_growth_df(self._growth_df())
        # A2G is in growth_df, but wt is not present in base_growth_df at all.
        base_growth_df_raw = pd.DataFrame({
            "genotype": ["A2G"],
            "rate": [0.015],
            "rate_std": [0.002],
        })

        with pytest.raises(ValueError, match="wt"):
            _read_base_growth_df(base_growth_df_raw, growth_df)

    def test_wt_dropped_by_growth_df_filter_raises(self):
        """wt present in base_growth_df but not in growth_df -- still an error,
        since post-filtering wt is what's actually required."""
        from tfscreen.tfmodel.model_orchestrator import _read_base_growth_df

        growth_df = _read_growth_df(self._growth_df(genotypes=("A2G",)))  # no wt at all
        base_growth_df_raw = pd.DataFrame({
            "genotype": ["wt", "A2G"],
            "rate": [0.02, 0.015],
            "rate_std": [0.001, 0.002],
        })

        with pytest.raises(ValueError, match="wt"):
            _read_base_growth_df(base_growth_df_raw, growth_df)

    def test_multiple_rows_per_genotype_combined_by_inverse_variance(self):
        from tfscreen.tfmodel.model_orchestrator import _read_base_growth_df

        growth_df = _read_growth_df(self._growth_df())
        # Two independent measurements of wt: precision-weighted mean.
        base_growth_df_raw = pd.DataFrame({
            "genotype": ["wt", "wt"],
            "rate": [0.02, 0.03],
            "rate_std": [0.01, 0.02],
        })

        result = _read_base_growth_df(base_growth_df_raw, growth_df)
        assert len(result) == 1

        precision = np.array([1 / 0.01**2, 1 / 0.02**2])
        expected_rate = float(np.sum(np.array([0.02, 0.03]) * precision) / precision.sum())
        expected_std = float(np.sqrt(1.0 / precision.sum()))

        row = result.iloc[0]
        assert row["genotype"] == "wt"
        assert row["rate"] == pytest.approx(expected_rate)
        assert row["rate_std"] == pytest.approx(expected_std)
        # Combined precision must exceed either individual measurement's.
        assert row["rate_std"] < 0.01

    def test_build_base_growth_arrays_shapes_and_values(self):
        from tfscreen.tfmodel.model_orchestrator import (
            _read_base_growth_df,
            _build_base_growth_arrays,
        )

        growth_df = _read_growth_df(self._growth_df())
        tm = _build_growth_tm(growth_df)
        genotype_labels = tm.tensor_dim_labels[tm.tensor_dim_names.index("genotype")]

        base_growth_df_raw = pd.DataFrame({
            "genotype": ["wt", "A2G"],
            "rate": [0.02, 0.015],
            "rate_std": [0.001, 0.002],
        })
        processed = _read_base_growth_df(base_growth_df_raw, growth_df)
        arrays = _build_base_growth_arrays(processed, genotype_labels)

        assert arrays["rate_obs"].shape == (len(genotype_labels),)
        assert arrays["rate_std"].shape == (len(genotype_labels),)
        assert arrays["good_mask"].shape == (len(genotype_labels),)

        label_to_idx = {str(g): i for i, g in enumerate(genotype_labels)}
        wt_idx = label_to_idx["wt"]
        a2g_idx = label_to_idx["A2G"]
        k3r_idx = label_to_idx["K3R"]  # not in base_growth_df_raw

        assert arrays["good_mask"][wt_idx]
        assert arrays["rate_obs"][wt_idx] == pytest.approx(0.02)
        assert arrays["rate_std"][wt_idx] == pytest.approx(0.001)

        assert arrays["good_mask"][a2g_idx]
        assert arrays["rate_obs"][a2g_idx] == pytest.approx(0.015)

        assert not arrays["good_mask"][k3r_idx]

    def test_derive_k_ref_guess_returns_wt_rate(self):
        from tfscreen.tfmodel.model_orchestrator import (
            _read_base_growth_df,
            _derive_k_ref_guess,
        )

        growth_df = _read_growth_df(self._growth_df())
        base_growth_df_raw = pd.DataFrame({
            "genotype": ["wt", "A2G"],
            "rate": [0.021, 0.015],
            "rate_std": [0.001, 0.002],
        })
        processed = _read_base_growth_df(base_growth_df_raw, growth_df)

        guess = _derive_k_ref_guess(processed)
        assert guess == pytest.approx(0.021)

    def test_base_growth_df_wires_k_ref_into_growth_priors(self, mocker):
        """
        End-to-end: supplying base_growth_df must derive a k_ref guess from
        wt's rate and thread it into growth_priors.base_growth as a
        BaseGrowthPriors(k_ref_loc=guess, k_ref_scale=_K_REF_PRIOR_SCALE)
        instance. Without base_growth_df, growth_priors.base_growth must
        stay None (its dataclass default).
        """
        from tfscreen.tfmodel.model_orchestrator import _K_REF_PRIOR_SCALE
        from tfscreen.tfmodel.tensors.populate_dataclass import (
            populate_dataclass as real_populate_dataclass,
        )
        from tfscreen.tfmodel.data_class import DataClass, BaseGrowthPriors

        mock_growth_tm = create_mock_tm(is_growth=True)
        mock_binding_tm = create_mock_tm(is_growth=False)
        mocker.patch("tfscreen.tfmodel.model_orchestrator._read_growth_df")
        mocker.patch("tfscreen.tfmodel.model_orchestrator._build_growth_tm", return_value=mock_growth_tm)
        mocker.patch("tfscreen.tfmodel.model_orchestrator._read_binding_df")
        mocker.patch("tfscreen.tfmodel.model_orchestrator._build_binding_tm", return_value=mock_binding_tm)
        mocker.patch(
            "tfscreen.tfmodel.model_orchestrator._setup_batching",
            return_value={
                "batch_idx": jnp.arange(7), "batch_size": 7,
                "scale_vector": jnp.ones(7), "num_binding": 0,
            },
        )
        mocker.patch(
            "tfscreen.tfmodel.model_orchestrator._read_base_growth_df",
            return_value=pd.DataFrame({
                "genotype": ["wt"], "rate": [0.0234], "rate_std": [0.001],
            }),
        )
        mocker.patch(
            "tfscreen.tfmodel.model_orchestrator._build_base_growth_arrays",
            return_value={
                "rate_obs": np.zeros(7), "rate_std": np.ones(7),
                "good_mask": np.zeros(7, dtype=bool),
            },
        )

        # Real populate_dataclass everywhere except DataClass, which would
        # otherwise require a fully-realistic set of growth/binding tensors
        # that this test has no need to build.
        def _populate_side_effect(target_dataclass, sources):
            if target_dataclass is DataClass:
                return MagicMock()
            return real_populate_dataclass(target_dataclass, sources)

        mocker.patch(
            "tfscreen.tfmodel.model_orchestrator.populate_dataclass",
            side_effect=_populate_side_effect,
        )

        registry = {
            "condition_growth": {"hierarchical": MagicMock()},
            "growth_transition": {"instant": MagicMock()},
            "ln_cfu0": {"hierarchical": MagicMock()},
            "dk_geno": {"hierarchical_geno": MagicMock()},
            "activity": {"hierarchical_geno": MagicMock()},
            "theta": {"hill_geno": MagicMock()},
            "transformation": {"single": MagicMock()},
            "theta_rescale": {"passthrough": MagicMock()},
            "theta_growth_noise": {"zero": MagicMock()},
            "theta_binding_noise": {"zero": MagicMock()},
            "growth_noise": {"zero": MagicMock()},
            "sample_offset": {"zero": MagicMock()},
            "observe_binding": MagicMock(),
            "observe_growth": MagicMock(),
        }
        with patch.dict("tfscreen.tfmodel.model_orchestrator.model_registry", registry, clear=True):
            for k in ["condition_growth", "growth_transition", "ln_cfu0", "dk_geno",
                      "activity", "theta", "transformation", "theta_growth_noise",
                      "theta_binding_noise", "growth_noise", "sample_offset"]:
                for sub_k in tfscreen.tfmodel.model_orchestrator.model_registry[k]:
                    mod = tfscreen.tfmodel.model_orchestrator.model_registry[k][sub_k]
                    mod.get_priors.return_value = {}
                    mod.get_guesses.return_value = {}

            model = ModelOrchestrator(
                "g.csv", "b.csv",
                base_growth_df="base_growth.csv",
                condition_growth="hierarchical",
                growth_transition="instant",
                activity="hierarchical_geno",
                theta="hill_geno",
                transformation="single",
                theta_growth_noise="zero",
            )

        base_growth_priors = model._priors.growth.base_growth
        assert isinstance(base_growth_priors, BaseGrowthPriors)
        assert base_growth_priors.k_ref_loc == pytest.approx(0.0234)
        assert base_growth_priors.k_ref_scale == _K_REF_PRIOR_SCALE

        with patch.dict("tfscreen.tfmodel.model_orchestrator.model_registry", registry, clear=True):
            for k in ["condition_growth", "growth_transition", "ln_cfu0", "dk_geno",
                      "activity", "theta", "transformation", "theta_growth_noise",
                      "theta_binding_noise", "growth_noise", "sample_offset"]:
                for sub_k in tfscreen.tfmodel.model_orchestrator.model_registry[k]:
                    mod = tfscreen.tfmodel.model_orchestrator.model_registry[k][sub_k]
                    mod.get_priors.return_value = {}
                    mod.get_guesses.return_value = {}

            model_no_base_growth = ModelOrchestrator(
                "g.csv", "b.csv",
                condition_growth="hierarchical",
                growth_transition="instant",
                activity="hierarchical_geno",
                theta="hill_geno",
                transformation="single",
                theta_growth_noise="zero",
            )

        assert model_no_base_growth._priors.growth.base_growth is None
