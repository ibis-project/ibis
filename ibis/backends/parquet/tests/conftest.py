import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ibis.backends.parquet import ParquetClient


@pytest.fixture
def parquet(tmpdir, file_backends_data):
    # create single files
    d = tmpdir.mkdir('pq')

    for k, v in file_backends_data.items():
        f = d / '{}.parquet'.format(k)
        table = pa.Table.from_pandas(v)
        pq.write_table(table, str(f))

    return ParquetClient(tmpdir).database()
