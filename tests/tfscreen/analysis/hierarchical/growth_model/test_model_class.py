import pytest
import pandas as pd
import jax.numpy as jnp
import numpy as np
import os
import yaml
import tfscreen
from unittest.mock import MagicMock, call, patch, ANY

# Import the module under test
from tfscreen.analysis.hierarchical.growth_model.model_class import (
    ModelClass,
    _read_growth_df,
    _read_binding_df,
    _build_growth_tm,
    _build_binding_tm,
    _setup_batching,
    _extract_param_est,
    get_batch
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
    assert isinstance(res, tfscreen.analysis.hierarchical.TensorManager)
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
    assert isinstance(res, tfscreen.analysis.hierarchical.TensorManager)
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
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class.add_group_columns", return_value=base_growth_df)
    
    res = _read_growth_df("path.csv")
    assert "replicate" in res.columns

def test_read_growth_df_no_replicate(mocker, base_growth_df):
    df_no_rep = base_growth_df.drop(columns=["replicate"])
    mocker.patch("tfscreen.util.io.read_dataframe", return_value=df_no_rep)
    mocker.patch("tfscreen.genetics.set_categorical_genotype", return_value=df_no_rep)
    mocker.patch("tfscreen.util.dataframe.get_scaled_cfu", return_value=df_no_rep)
    mocker.patch("tfscreen.util.dataframe.check_columns")
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class.add_group_columns", return_value=df_no_rep)
    
    res = _read_growth_df("path.csv")
    assert "replicate" in res.columns
    assert res["replicate"].iloc[0] == 1

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
    
    res = _read_binding_df("path.csv", growth_df=growth_df)
    assert "theta_obs" in res.columns

def test_read_binding_df_errors(mocker):
    growth_df = pd.DataFrame({"genotype": ["A"], "titrant_name": ["T1"]})
    binding_df = pd.DataFrame({
        "genotype": ["B"], "titrant_name": ["T1"], 
        "theta_obs": [0.5], "theta_std": [0.1], "titrant_conc": [1.0]
    })
    mocker.patch("tfscreen.util.io.read_dataframe", return_value=binding_df)
    mocker.patch("tfscreen.genetics.set_categorical_genotype", return_value=binding_df)
    mocker.patch("tfscreen.util.dataframe.check_columns")
    
    with patch("pandas.Index.isin", return_value=np.array([False])):
        with pytest.raises(ValueError, match="not seen"):
            _read_binding_df("path.csv", growth_df=growth_df)
        
    mocker.patch("tfscreen.util.dataframe.check_columns", side_effect=ValueError("missing col"))
    with pytest.raises(ValueError, match="missing col"):
        _read_binding_df("path.csv", growth_df=growth_df)

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
# 8. Tests for ModelClass
# ----------------------------------------------------------------------------

@pytest.fixture
def initialized_model_class():
    model = MagicMock(spec=ModelClass)
    model._theta = "hill"
    model._condition_growth = "hierarchical"
    model._ln_cfu0 = "hierarchical"
    model._dk_geno = "hierarchical"
    model._activity = "hierarchical"
    model._transformation = "none"
    model._theta_growth_noise = "none"
    model._theta_binding_noise = "none"
    model._spiked_genotypes = None
    model._batch_size = 1
    model.growth_tm = MagicMock()
    model.binding_tm = MagicMock()
    model.growth_df = pd.DataFrame()
    return model

def create_mock_tm(is_growth=True):
    tm = MagicMock()
    tm.map_sizes = {}
    if is_growth:
        tm.map_sizes = {"condition": 1}
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
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._read_growth_df")
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._build_growth_tm", return_value=mock_growth_tm)
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._read_binding_df")
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._build_binding_tm", return_value=mock_binding_tm)
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class.populate_dataclass", return_value=MagicMock())
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._setup_batching", return_value={"batch_idx": jnp.array([0]), "batch_size": 1, "scale_vector": jnp.array([1.0]), "num_binding": 0})
    
    # 741: dk_geno unrecognized
    with pytest.raises(ValueError, match="not recognized"):
        ModelClass("g.csv", "b.csv", dk_geno="invalid_name")

    # 735: key not in model_registry
    with patch("tfscreen.analysis.hierarchical.growth_model.model_class.model_registry", {}):
        with pytest.raises(ValueError, match="is not in model_registry"):
            ModelClass("g.csv", "b.csv")

def test_initialize_data_spiked(mocker):
    mock_growth_tm = create_mock_tm(is_growth=True)
    mock_growth_tm.tensor_shape[6] = 2
    mock_growth_tm.tensor_dim_labels[6] = np.array(["wt", "m1"])
    for k in mock_growth_tm.tensors:
        mock_growth_tm.tensors[k] = np.zeros((1,1,1,1,1,1,2))
    
    mock_binding_tm = create_mock_tm(is_growth=False)

    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._read_growth_df")
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._build_growth_tm", return_value=mock_growth_tm)
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._read_binding_df")
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._build_binding_tm", return_value=mock_binding_tm)
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class.populate_dataclass", return_value=MagicMock())
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._setup_batching", return_value={"batch_idx": jnp.array([0,1]), "batch_size": 2, "scale_vector": jnp.array([1.0, 1.0]), "num_binding": 0})
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class.ModelClass._initialize_classes")

    model = ModelClass("g.csv", "b.csv", spiked_genotypes=["m1"])
    assert model._spiked_genotypes == ["m1"]
    
    model = ModelClass("g.csv", "b.csv", spiked_genotypes="m1")
    assert model._spiked_genotypes == ["m1"]
    
    with pytest.raises(ValueError, match="not found"):
        ModelClass("g.csv", "b.csv", spiked_genotypes="m2")

def test_initialize_classes_logic(mocker):
    mock_growth_tm = create_mock_tm(is_growth=True)
    mock_binding_tm = create_mock_tm(is_growth=False)
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._read_growth_df")
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._build_growth_tm", return_value=mock_growth_tm)
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._read_binding_df")
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._build_binding_tm", return_value=mock_binding_tm)
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class._setup_batching", return_value={"batch_idx": jnp.array([0]), "batch_size": 1, "scale_vector": jnp.array([1.0]), "num_binding": 0})
    mocker.patch("tfscreen.analysis.hierarchical.growth_model.model_class.populate_dataclass", return_value=MagicMock())

    with patch.dict("tfscreen.analysis.hierarchical.growth_model.model_class.model_registry", {
        "condition_growth": {"hierarchical": MagicMock(), "independent": MagicMock()},
        "ln_cfu0": {"hierarchical": MagicMock()},
        "dk_geno": {"hierarchical": MagicMock()},
        "activity": {"hierarchical": MagicMock(), "horseshoe": MagicMock()},
        "theta": {"categorical": MagicMock(), "hill": MagicMock(), "fixed": MagicMock()},
        "transformation": {"congression": MagicMock(), "none": MagicMock()},
        "theta_growth_noise": {"none": MagicMock()},
        "theta_binding_noise": {"none": MagicMock()},
        "observe_binding": MagicMock(),
        "observe_growth": MagicMock()
    }, clear=True):
        for k in ["condition_growth", "ln_cfu0", "dk_geno", "activity", "theta", "transformation", "theta_growth_noise", "theta_binding_noise"]:
            for sub_k in tfscreen.analysis.hierarchical.growth_model.model_class.model_registry[k]:
                tfscreen.analysis.hierarchical.growth_model.model_class.model_registry[k][sub_k].get_priors.return_value = {}
                tfscreen.analysis.hierarchical.growth_model.model_class.model_registry[k][sub_k].get_guesses.return_value = {}

        model = ModelClass("g.csv", "b.csv", theta="categorical", transformation="congression", condition_growth="independent")
        assert model._theta == "categorical"

def test_model_class_write_config(initialized_model_class, tmpdir):
    model = initialized_model_class
    model.settings = {"a":1}
    ModelClass.write_config(model, "g.csv", "b.csv", os.path.join(tmpdir, "out"))
    assert os.path.exists(os.path.join(tmpdir, "out_config.yaml"))

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
    model._condition_growth = "cg"
    model._ln_cfu0 = "ln"
    model._dk_geno = "dk"
    model._theta_growth_noise = "gn"
    model._theta_binding_noise = "bn"
    model._spiked_genotypes = ["s"]
    
    assert ModelClass.jax_model.fget(model) == "jm"
    assert ModelClass.jax_model_guide.fget(model) == "jmg"
    assert ModelClass.data.fget(model) == "d"
    assert ModelClass.priors.fget(model) == "p"
    assert ModelClass.init_params.fget(model) == "ip"
    assert ModelClass.settings.fget(model)["activity"] == "a"
    assert ModelClass.settings.fget(model)["theta"] == "t"
    assert ModelClass.settings.fget(model)["transformation"] == "tr"
    assert ModelClass.settings.fget(model)["condition_growth"] == "cg"
    assert ModelClass.settings.fget(model)["ln_cfu0"] == "ln"
    assert ModelClass.settings.fget(model)["dk_geno"] == "dk"
    assert ModelClass.settings.fget(model)["theta_growth_noise"] == "gn"
    assert ModelClass.settings.fget(model)["theta_binding_noise"] == "bn"
    assert ModelClass.settings.fget(model)["spiked_genotypes"] == ["s"]
    assert ModelClass.get_batch.fget(model) == get_batch

def test_extract_parameters_full(initialized_model_class):
    model = initialized_model_class
    model._theta = "hill"
    model._condition_growth = "hierarchical"
    model._dk_geno = "hierarchical"
    model._activity = "hierarchical"
    model._transformation = "congression"
    model.growth_tm.df = pd.DataFrame({
        "genotype": ["wt"], "titrant_name": ["T"], 
        "replicate": [1], "condition_pre": ["A"],
        "map_theta_group": [0], "map_ln_cfu0": [0],
        "map_genotype": [0], "titrant_name_idx": [0], "titrant_conc_idx": [0],
        "titrant_conc": [1.0]
    })
    model.growth_tm.map_groups = {"condition": pd.DataFrame({"replicate":[1], "condition":["A"], "map_condition":[0]})}
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
    res = ModelClass.extract_parameters(model, post)
    assert "hill_n" in res
    assert "activity" in res
    assert "lam" in res

def test_extract_parameters_errors(initialized_model_class):
    model = initialized_model_class
    with pytest.raises(ValueError, match="should be a dictionary"):
        ModelClass.extract_parameters(model, {}, q_to_get="not_a_dict")

def test_extract_parameters_npz(initialized_model_class, tmpdir):
    model = initialized_model_class
    model._theta = "hill"
    model._condition_growth = "none"
    model._dk_geno = "none"
    model._activity = "fixed"
    model.growth_tm.df = pd.DataFrame({"genotype": ["wt"], "titrant_name": ["T"], "map_theta_group": [0], "map_ln_cfu0": [0]})
    post = {"theta_hill_n": np.zeros((1, 1)), "theta_log_hill_K": np.zeros((1, 1)), 
            "theta_theta_high": np.zeros((1, 1)), "theta_theta_low": np.zeros((1, 1)),
            "ln_cfu0": np.zeros((1, 1))}
    path = os.path.join(tmpdir, "post.npz")
    np.savez(path, **post)
    res = ModelClass.extract_parameters(model, path)
    assert "hill_n" in res

def test_extract_parameters_categorical(initialized_model_class):
    model = initialized_model_class
    model._theta = "categorical"
    model._condition_growth = "none"
    model._dk_geno = "none"
    model._activity = "fixed"
    model.growth_tm.df = pd.DataFrame({"genotype": ["wt"], "titrant_name": ["T"], "titrant_conc": [1.0], "map_theta": [0]})
    post = {"theta_theta": np.zeros((1, 1))}
    res = ModelClass.extract_parameters(model, post)
    assert "theta" in res

def test_extract_theta_curves_full(initialized_model_class, tmpdir):
    model = initialized_model_class
    model._theta = "hill"
    model.growth_tm.df = pd.DataFrame({"genotype": ["wt"], "titrant_name": ["T"], "titrant_conc": [1.0], "map_theta_group": [0]})
    post = {
        "theta_hill_n": np.zeros((1, 1, 1)), "theta_log_hill_K": np.zeros((1, 1, 1)),
        "theta_theta_high": np.zeros((1, 1, 1)), "theta_theta_low": np.zeros((1, 1, 1))
    }
    res = ModelClass.extract_theta_curves(model, post)
    assert len(res) > 0
    
    # Test path for extract_theta_curves (1047)
    path = os.path.join(tmpdir, "theta.npz")
    np.savez(path, **post)
    res = ModelClass.extract_theta_curves(model, path)
    assert len(res) > 0

def test_extract_theta_curves_manual(initialized_model_class):
    model = initialized_model_class
    model._theta = "hill"
    model.growth_tm.df = pd.DataFrame({"genotype": ["wt"], "titrant_name": ["T"], "map_theta_group": [0]})
    post = {
        "theta_hill_n": np.zeros((1, 1, 1)), "theta_log_hill_K": np.zeros((1, 1, 1)),
        "theta_theta_high": np.zeros((1, 1, 1)), "theta_theta_low": np.zeros((1, 1, 1))
    }
    
    # Test manual_titrant_df (1074-1085)
    manual = pd.DataFrame({"titrant_name": ["T"], "titrant_conc": [0.5]})
    res = ModelClass.extract_theta_curves(model, post, manual_titrant_df=manual)
    assert len(res) == 1
    
    # Test manual_titrant_df with genotype (1087)
    manual["genotype"] = "wt"
    res = ModelClass.extract_theta_curves(model, post, manual_titrant_df=manual)
    assert len(res) == 1
    
    # Test mapping error (1103-1109)
    # Trigger 1106 by mocking Index.map to raise
    with patch("pandas.Index.map", side_effect=Exception("forced failure")):
        with pytest.raises(ValueError, match=r"Some \(genotype, titrant_name\) pairs"):
            ModelClass.extract_theta_curves(model, post, manual_titrant_df=manual)

    # Trigger 1113 by passing a genotype that doesn't exist
    manual["genotype"] = "missing"
    with pytest.raises(ValueError, match="were not found in the model data"):
        ModelClass.extract_theta_curves(model, post, manual_titrant_df=manual)

def test_extract_theta_curves_errors(initialized_model_class):
    model = initialized_model_class
    model._theta = "categorical"
    with pytest.raises(ValueError, match="only available for models where theta='hill'"):
        ModelClass.extract_theta_curves(model, {})
    model._theta = "hill"
    with pytest.raises(ValueError, match="should be a dictionary"):
        ModelClass.extract_theta_curves(model, {}, q_to_get="not_a_dict")

def test_extract_growth_predictions_full(initialized_model_class, tmpdir):
    model = initialized_model_class
    model._transformation = "none"
    model.growth_df = pd.DataFrame({
        "genotype": ["wt"], "titrant_name": ["T"], "titrant_conc": [1.0],
        "replicate_idx": [0], "time_idx": [0], "condition_pre_idx": [0],
        "condition_sel_idx": [0], "titrant_name_idx": [0], "titrant_conc_idx": [0],
        "genotype_idx": [0]
    })
    post = {"growth_pred": np.zeros((1, 1, 1, 1, 1, 1, 1, 1))}
    res = ModelClass.extract_growth_predictions(model, post)
    assert "median" in res.columns
    
    with pytest.raises(ValueError, match="'growth_pred' not found"):
        ModelClass.extract_growth_predictions(model, {})
    with pytest.raises(ValueError, match="should be a dictionary"):
        ModelClass.extract_growth_predictions(model, {"growth_pred": None}, q_to_get="bad")
        
    path = os.path.join(tmpdir, "gp.npz")
    np.savez(path, growth_pred=np.zeros((1,1,1,1,1,1,1,1)))
    res = ModelClass.extract_growth_predictions(model, path)
    assert len(res) > 0

def test_get_random_idx_logic(initialized_model_class):
    model = initialized_model_class
    model.data = MagicMock()
    model.data.growth.batch_idx = jnp.array([0, 1])
    model.data.not_binding_idx = jnp.array([2, 3])
    model.data.not_binding_batch_size = 1
    
    with pytest.raises(ValueError, match="must be called with an integer batch key"):
        ModelClass.get_random_idx(model)
        
    res = ModelClass.get_random_idx(model, batch_key=42)
    assert res.shape == (2,)
    
    # num_batches > 1 (1351-1360)
    res = ModelClass.get_random_idx(model, num_batches=2)
    assert res.shape == (2, 2)

def test_load_config_errors(tmpdir):
    # 1409: FileNotFoundError
    with pytest.raises(FileNotFoundError):
        ModelClass.load_config("missing.yaml")
        
    path = os.path.join(tmpdir, "bad.yaml")
    with open(path, "w") as f:
        yaml.dump({"growth_df": "g.csv"}, f)
    with pytest.raises(ValueError, match="Missing required field: binding_df"):
        ModelClass.load_config(path)

    with open(path, "w") as f:
        yaml.dump({"growth_df": "g.csv", "binding_df": "b.csv", "settings": {}}, f)
    with pytest.raises(ValueError, match="Missing required field: tfscreen_version"):
        ModelClass.load_config(path)