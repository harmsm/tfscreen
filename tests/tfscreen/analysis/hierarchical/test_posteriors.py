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

def test_get_posterior_samples_auto_loc_priority():
    """_auto_loc should be preferred over _mean when both exist."""
    posteriors = {
        "param_auto_loc": np.array([10, 20]),
        "param_mean": np.array([30, 40]),
    }
    result = get_posterior_samples(posteriors, "param")
    assert np.array_equal(result, [10, 20])

def test_get_posterior_samples_error_truncates_long_key_list():
    """Error message truncates key list when more than 10 keys exist."""
    posteriors = {f"key_{i}": np.array([i]) for i in range(15)}
    with pytest.raises(KeyError, match=r"\.\.\."):
        get_posterior_samples(posteriors, "missing_param")

def test_get_posterior_samples_error_no_truncation_short_key_list():
    """Error message does not truncate when <=10 keys exist."""
    posteriors = {f"key_{i}": np.array([i]) for i in range(5)}
    with pytest.raises(KeyError) as exc_info:
        get_posterior_samples(posteriors, "missing_param")
    assert "..." not in str(exc_info.value)

def test_load_posteriors_h5_oserror_retries(tmp_path, mocker):
    """OSError on h5py.File is retried up to max_retries-1 times."""
    d = tmp_path / "post.h5"
    with h5py.File(d, "w") as f:
        f.create_dataset("x", data=np.array([1]))

    mock_sleep = mocker.patch("time.sleep")

    # Build a class (not a Mock) so isinstance(..., h5py.File) in posteriors.py
    # still receives a proper type and doesn't raise TypeError.
    _real_open = h5py.File
    _call_count = {"n": 0}
    _real_file_holder = {}

    class _SometimesFailing(h5py.File):
        def __new__(cls, *args, **kwargs):
            _call_count["n"] += 1
            if _call_count["n"] <= 2:
                raise OSError("busy")
            obj = _real_open(*args, **kwargs)
            _real_file_holder["f"] = obj
            return obj

    mocker.patch.object(h5py, "File", _SometimesFailing)

    q, p = load_posteriors(str(d))
    assert mock_sleep.call_count == 2
    assert "x" in p
    p.close()

def test_load_posteriors_h5_oserror_exhausted(tmp_path, mocker):
    """OSError is re-raised after all retries are exhausted."""
    d = tmp_path / "post.h5"

    mocker.patch("time.sleep")

    class _AlwaysFails(h5py.File):
        def __new__(cls, *args, **kwargs):
            raise OSError("always fails")

    mocker.patch.object(h5py, "File", _AlwaysFails)

    with pytest.raises(OSError, match="always fails"):
        load_posteriors(str(d))
