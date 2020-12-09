from pathlib import Path

import pytest

import ibis
from ibis.backends.pandas.tests.conftest import PandasTest

from ibis.backends.hdf5 import HDFClient


class HDF5Test(PandasTest):
    check_names = False
    supports_divide_by_zero = True
    returned_timestamp_unit = 'ns'

    @staticmethod
    def connect(data_directory: Path) -> ibis.client.Client:
        filename = data_directory / 'functional_alltypes.h5'
        if not filename.exists():
            pytest.skip('test data set {} not found'.format(filename))
        return ibis.hdf5.connect(data_directory)


@pytest.fixture
def hdf(tmpdir, file_backends_data):
    hdf = tmpdir.mkdir('hdf_dir')
    f = hdf / 'prices.h5'

    for k, v in file_backends_data.items():
        v.to_hdf(str(f), k, format='table', data_columns=True)

    return HDFClient(tmpdir).database()
