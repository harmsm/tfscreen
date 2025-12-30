import pytest
import os

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
