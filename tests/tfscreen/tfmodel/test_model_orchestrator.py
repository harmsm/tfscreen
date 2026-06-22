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
    get_batch
)
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
    return pd.DataFrame({c: [1.0] for c in cols if c != "genotype"} | {"genotype": ["wt"]})

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
    assert "M42I" in captured.out

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

    def test_linear_get_priors_signature_has_condition_labels(self):
        """
        The orchestrator detects condition_labels support via inspect.signature.
        Verify that the linear component's get_priors actually has this param.
        """
        import inspect
        from tfscreen.tfmodel.generative.components.growth.linear import get_priors
        sig = inspect.signature(get_priors)
        assert "condition_labels" in sig.parameters, (
            "linear.get_priors must have a 'condition_labels' parameter so "
            "ModelOrchestrator can pass condition names for per-condition m priors"
        )

    def test_setup_model_extracts_labels_and_calls_get_priors(self, mocker):
        """
        _setup_model must call get_priors(condition_labels=...) for a
        condition_growth component that accepts that parameter.
        """
        import inspect
        import tfscreen.tfmodel.model_orchestrator as mo
        from tfscreen.tfmodel.generative.components.growth import linear as real_linear

        captured = {}

        def capturing_get_priors(condition_labels=None):
            captured["condition_labels"] = condition_labels
            return real_linear.get_priors(condition_labels=condition_labels)

        cond_rep_df = pd.DataFrame({
            "condition_rep": ["kanR+kan", "kanR-kan"],
            "map_condition_rep": [0, 1],
        })
        mock_growth_tm = MagicMock()
        mock_growth_tm.map_groups = {"condition_rep": cond_rep_df}

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
                pass  # only condition_labels matters

        assert "condition_labels" in captured, (
            "_setup_model did not pass condition_labels to get_priors"
        )
        assert captured["condition_labels"] == ["kanR+kan", "kanR-kan"]