from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ibis
from ibis.backends.pandas.tests.conftest import TestConf as PandasTest


class TestConf(PandasTest):
    check_names = False
    supports_divide_by_zero = True
    returned_timestamp_unit = 'ns'

    @staticmethod
    def connect(data_directory: Path):
        filename = data_directory / 'functional_alltypes.parquet'
        if not filename.exists():
            pytest.skip(f'test data set {filename} not found')
        return ibis.parquet.connect(data_directory)


@pytest.fixture
def parquet(tmpdir, file_backends_data):
    # create single files
    d = tmpdir.mkdir('pq')

    for k, v in file_backends_data.items():
        f = d / f'{k}.parquet'
        table = pa.Table.from_pandas(v)
        pq.write_table(table, str(f))

    return ibis.parquet.connect(tmpdir).database()
