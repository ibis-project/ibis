import os
from pathlib import Path

import pytest

collect_ignore = ['setup.py']


@pytest.fixture(scope='session')
def data_directory():
    root = Path(__file__).absolute().parent

    default = root / 'ci' / 'ibis-testing-data'
    datadir = os.environ.get('IBIS_TEST_DATA_DIRECTORY', default)
    datadir = Path(datadir)

    if not datadir.exists():
        pytest.skip('test data directory not found')

    return datadir
