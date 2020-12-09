from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ibis
from ..parquet import ParquetClient
from ibis.backends.pandas.tests.conftest import PandasTest


class ParquetTest(PandasTest):
    check_names = False
    supports_divide_by_zero = True
    returned_timestamp_unit = 'ns'

    @staticmethod
    def connect(data_directory: Path) -> ibis.client.Client:
        filename = data_directory / 'functional_alltypes.parquet'
        if not filename.exists():
            pytest.skip('test data set {} not found'.format(filename))
        return ibis.parquet.connect(data_directory)


@pytest.fixture
def parquet(tmpdir, file_backends_data):
    # create single files
    d = tmpdir.mkdir('pq')

    for k, v in file_backends_data.items():
        f = d / '{}.parquet'.format(k)
        table = pa.Table.from_pandas(v)
        pq.write_table(table, str(f))

    return ParquetClient(tmpdir).database()
