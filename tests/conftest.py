import pytest

import os
import glob

def get_files(base_dir):
    """
    Traverse base_dir and return a dictionary that keys all files and some
    rudimentary *.ext expressions to absolute paths to those files. They keys
    will be things like "some_dir/test0/rocket.txt" mapping to
    "c:/some_dir/life/base_dir/some_dir/test0/rocket.txt". The idea is to have
    easy-to-read cross-platform keys within unit tests.

    Classes of keys:

        + some_dir/test0/rocket.txt maps to a file (str)
        + some_dir/test0/ maps to the test0 directory itself (str)
        + some_dir/test0/*.txt maps to all .txt (list)
        + some_dir/test0/* maps to all files or directories in the directory
          (list)

    Note that base_dir is *not* included in the keys. All are relative to that
    directory by :code:`os.path.basename(__file__)/{base_dir}`.

    Parameters
    ----------
    base_dir : str
        base directory for search. should be relative to test file location.

    Returns
    -------
    output : dict
        dictionary keying string paths to absolute paths
    """

    containing_dir = os.path.dirname(os.path.realpath(__file__))
    starting_dir = os.path.abspath(os.path.join(containing_dir,base_dir))

    base_length = len(starting_dir.split(os.sep))

    # Traverse starting_dir
    output = {}
    for root, dirs, files in os.walk(starting_dir):

        # path relative to base_dir as a list
        this_path = root.split(os.sep)[base_length:]

        # Build paths to specific files
        local_files = []
        for file in files:
            local_files.append(os.path.join(root,file))
            new_key = this_path[:]
            new_key.append(file)
            output["/".join(new_key)] = local_files[-1]

        # Build paths to patterns of file types
        patterns = {}
        ext = list(set([f.split(".")[-1] for f in local_files]))
        for e in ext:
            new_key = this_path[:]
            new_key.append(f"*.{e}")
            output["/".join(new_key)] = glob.glob(os.path.join(root,f"*.{e}"))

        # Build path to all files in this directory
        new_key = this_path[:]
        new_key.append("*")
        output["/".join(new_key)] = glob.glob(os.path.join(root,f"*"))

        # Build paths to directories in this directory
        for this_dir in dirs:
            new_key = this_path[:]
            new_key.append(this_dir)
            # dir without terminating /
            output["/".join(new_key)] = os.path.join(root,this_dir)

            # dir with terminating /
            new_key.append("")
            output["/".join(new_key)] = os.path.join(root,this_dir)

    # make sure output is sorted stably
    for k in output:
        if issubclass(type(output[k]),str):
            continue

        new_output = list(output[k])
        new_output.sort()
        output[k] = new_output

    return output

@pytest.fixture(scope="module")
def dummy():
    """
    """
    return None

## Skip marker for tests that require PyTorch/Pyro (used during the NumPyro→Pyro port).
## Tests decorated with @pytest.mark.requires_torch are skipped if torch is not installed.
try:
    import torch  # noqa: F401
    import pyro   # noqa: F401
    _torch_available = True
except ImportError:
    _torch_available = False

requires_torch = pytest.mark.skipif(
    not _torch_available,
    reason="PyTorch/Pyro not installed (NumPyro env)"
)

## Code for skipping slow tests and numerical equivalence tests.
def pytest_addoption(parser):
    parser.addoption("--runslow",
                     action="store_true",
                     default=False,
                     help="run slow tests")
    parser.addoption("--runnuts",
                     action="store_true",
                     default=False,
                     help="run NUTS equivalence tests (very slow, requires reference fixtures)")

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "nuts: mark test as a NUTS equivalence test (requires --runnuts)")
    config.addinivalue_line("markers", "requires_torch: skip if PyTorch/Pyro not installed")

def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_nuts = pytest.mark.skip(reason="need --runnuts option to run")
    for item in items:
        if "slow" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(skip_slow)
        if "nuts" in item.keywords and not config.getoption("--runnuts"):
            item.add_marker(skip_nuts)
