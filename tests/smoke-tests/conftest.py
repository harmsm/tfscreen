import pytest
import os
import pyro

@pytest.fixture(autouse=True)
def clear_pyro_param_store():
    """Clear Pyro param store before each test to prevent contamination."""
    pyro.clear_param_store()
    yield
    pyro.clear_param_store()

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
