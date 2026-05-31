
from .generalized_main import (  # noqa: F401
    generalized_main
)


def read_lines(path):
    """Return non-empty, non-comment lines from a plain-text file (one value per line)."""
    lines = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines

