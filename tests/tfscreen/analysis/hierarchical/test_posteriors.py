import pytest
import numpy as np
import h5py
from tfscreen.analysis.hierarchical.posteriors import load_posteriors, get_posterior_samples

def test_load_posteriors_dict():
    """Test loading posteriors from a dictionary."""
    posteriors = {"param1": np.array([1, 2, 3]), "param2": np.array([4, 5, 6])}
    q, p = load_posteriors(posteriors)
    
    assert p == posteriors
    assert "median" in q
    assert q["median"] == 0.5
    assert len(q) == 9

def test_load_posteriors_npz(tmp_path):
    """Test loading posteriors from an .npz file."""
    d = tmp_path / "post.npz"
    np.savez(d, param1=np.array([1, 2, 3]))
    
    q, p = load_posteriors(str(d))
    
    assert "param1" in p
    assert np.array_equal(p["param1"], [1, 2, 3])
    assert "median" in q

def test_load_posteriors_h5(tmp_path):
    """Test loading posteriors from an .h5 file."""
    d = tmp_path / "post.h5"
    with h5py.File(d, "w") as f:
        f.create_dataset("param1", data=np.array([1, 2, 3]))
    
    # Passing the file path
    q, p = load_posteriors(str(d))
    assert "param1" in p
    assert np.array_equal(p["param1"][:], [1, 2, 3])
    p.close() # Close since we opened it via path
    
    # Passing the file object
    with h5py.File(d, "r") as f:
        q, p = load_posteriors(f)
        assert "param1" in p
        assert np.array_equal(p["param1"][:], [1, 2, 3])

def test_load_posteriors_custom_q():
    """Test loading with custom quantiles."""
    posteriors = {"param1": np.array([1, 2, 3])}
    custom_q = {"test": 0.123}
    q, p = load_posteriors(posteriors, q_to_get=custom_q)
    
    assert q == custom_q
    assert p == posteriors

def test_load_posteriors_errors():
    """Test validation errors."""
    posteriors = {"param1": np.array([1, 2, 3])}
    
    with pytest.raises(ValueError, match="q_to_get should be a dictionary"):
        load_posteriors(posteriors, q_to_get=[0.5])
    
    # Test invalid file path (np.load will raise error)
    with pytest.raises(Exception):
        load_posteriors("non_existent_file.npz")

def test_get_posterior_samples():
    """Test extracting posterior samples with fallbacks."""
    posteriors = {
        "param1": np.array([1, 2]),
        "param2_auto_loc": np.array([3, 4]),
        "param3_mean": np.array([5, 6])
    }
    
    # Exact match
    assert np.array_equal(get_posterior_samples(posteriors, "param1"), [1, 2])
    
    # _auto_loc fallback
    assert np.array_equal(get_posterior_samples(posteriors, "param2"), [3, 4])
    
    # _mean fallback
    assert np.array_equal(get_posterior_samples(posteriors, "param3"), [5, 6])
    
    # Not found
    with pytest.raises(KeyError, match="Parameter 'bad_param' not found"):
        get_posterior_samples(posteriors, "bad_param")
