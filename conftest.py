import fnmatch
import os
import sys
import pytest

from ibis.compat import Path


collect_ignore = ['setup.py']

if sys.version_info.major == 2:
    this_directory = os.path.dirname(__file__)
    bigquery_udf = os.path.join(this_directory, 'ibis', 'bigquery', 'udf')
    for root, _, filenames in os.walk(bigquery_udf):
        for filename in filenames:
            if fnmatch.fnmatch(filename, '*.py'):
                collect_ignore.append(os.path.join(root, filename))


@pytest.fixture(scope='session')
def data_directory():
    root = Path(__file__).absolute().parent

    default = root / 'ci' / 'ibis-testing-data'
    datadir = os.environ.get('IBIS_TEST_DATA_DIRECTORY', default)
    datadir = Path(datadir)

    if not datadir.exists():
        pytest.skip('test data directory not found')

    return datadir
