"""Settings for tests."""
import os
from pathlib import Path

import pytest

collect_ignore = ['setup.py']


@pytest.fixture(scope='session')
def data_directory() -> Path:
    """
    Fixture that returns the test data directory.

    Returns
    -------
    Path
        Test data directory
    """
    root = Path(__file__).absolute().parent

    default = root / 'ci' / 'ibis-testing-data'
    datadir = os.environ.get('IBIS_TEST_DATA_DIRECTORY', default)
    datadir = Path(datadir)

    return datadir
