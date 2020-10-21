import pytest

from ibis.backends.hdf5 import HDFClient


@pytest.fixture
def hdf(tmpdir, file_backends_data):
    hdf = tmpdir.mkdir('hdf_dir')
    f = hdf / 'prices.h5'

    for k, v in file_backends_data.items():
        v.to_hdf(str(f), k, format='table', data_columns=True)

    return HDFClient(tmpdir).database()
