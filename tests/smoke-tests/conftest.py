import pytest
import os
import numpy as np

@pytest.fixture(scope="session")
def smoke_test_dir():
    """Return the absolute path to the smoke tests directory."""
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope="session")
def growth_smoke_csv(smoke_test_dir):
    """Return the path to the growth smoke test data."""
    return os.path.join(smoke_test_dir, "test_data", "growth-smoke.csv")

@pytest.fixture(scope="session")
def binding_smoke_csv(smoke_test_dir):
    """Return the path to the binding smoke test data."""
    return os.path.join(smoke_test_dir, "test_data", "binding-smoke.csv")

@pytest.fixture(scope="session")
def struct_smoke_h5_path(tmp_path_factory, growth_smoke_csv):
    """
    Session-scoped fixture: write a single HDF5 ensemble file covering all four
    lac-dimer structures, with residue numbers drawn from the smoke growth CSV.
    """
    import pandas as pd
    import h5py

    df = pd.read_csv(growth_smoke_csv)
    resnums = sorted(set(
        int(r) for r in df["resid_1"].dropna()
        if str(r).strip() not in ("", "nan")
    ))
    if not resnums:
        resnums = [42]

    tmp = tmp_path_factory.mktemp("struct_smoke")
    h5_path = str(tmp / "ensemble.h5")
    L = len(resnums)
    structure_names = ("H", "HD", "L", "LE2")

    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset('structure_names',
                          data=np.array(list(structure_names), dtype=h5py.string_dtype()))
        for seed_i, name in enumerate(structure_names):
            rng = np.random.RandomState(seed_i)
            grp = hf.create_group(name)
            grp.create_dataset('logP',
                               data=rng.randn(L, 20).astype(np.float32))
            grp.create_dataset('residue_nums',
                               data=np.asarray(resnums, dtype=np.int32))
            grp.create_dataset('dist_matrix',
                               data=np.abs(rng.randn(L, L)).astype(np.float32))
            grp.create_dataset('n_chains_bearing_mut', data=np.int32(2))

    return h5_path
