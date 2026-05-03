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
def struct_smoke_npz_paths(tmp_path_factory, growth_smoke_csv):
    """
    Session-scoped fixture: create one NPZ per lac-dimer structure covering all
    residue numbers found in the smoke growth CSV.
    """
    import pandas as pd
    df = pd.read_csv(growth_smoke_csv)
    resnums = sorted(set(
        int(r) for r in df["resid_1"].dropna()
        if str(r).strip() not in ("", "nan")
    ))
    if not resnums:
        resnums = [42]

    tmp = tmp_path_factory.mktemp("struct_smoke")
    L = len(resnums)
    structure_names = ("H", "HD", "L", "LE2")
    paths = {}
    for seed_i, name in enumerate(structure_names):
        rng = np.random.RandomState(seed_i)
        path = str(tmp / f"{name}.npz")
        np.savez(
            path,
            logP=rng.randn(L, 20).astype(np.float32),
            residue_nums=np.asarray(resnums, dtype=np.int32),
            dist_matrix=np.abs(rng.randn(L, L)).astype(np.float32),
            n_chains_bearing_mut=np.int32(2),
        )
        paths[name] = path
    return paths
