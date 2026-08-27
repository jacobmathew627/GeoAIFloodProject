"""Shared pytest fixtures and import path setup."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"

# The project modules import each other by bare name (`from config import ...`),
# so src/ has to be on the path rather than being imported as a package.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

NODATA = -9999.0


@pytest.fixture
def probability_map():
    """A small probability map with a nodata hole."""
    rng = np.random.default_rng(0)
    data = rng.random((32, 32)).astype(np.float32)
    data[0:4, 0:4] = NODATA
    return data


@pytest.fixture
def identity_transform():
    """A 10 m affine transform, matching the project's master grid."""
    from rasterio.transform import Affine

    return Affine(10.0, 0.0, 627980.0, 0.0, -10.0, 1139200.0)


@pytest.fixture
def curve_number_grid():
    """A CN grid spanning forest to open water, with a NaN hole."""
    cn = np.array(
        [
            [70.0, 80.0, 88.0],
            [90.0, 100.0, 74.0],
            [np.nan, 70.0, 88.0],
        ],
        dtype=np.float32,
    )
    return cn
