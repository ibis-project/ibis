import pytest
import ibis

pytest.importorskip('sqlalchemy')
pytest.importorskip('impala.dbapi')


pytestmark = pytest.mark.impala


def test_connection_pool_size(impala_port, hdfs, impala_host, test_data_db):
    client = ibis.impala.connect(
        port=impala_port,
        hdfs_client=hdfs,
        host=impala_host,
        database=test_data_db,
    )
    assert len(client.con.connection_pool) == 1


def test_connection_pool_size_after_close(
    impala_port, hdfs, impala_host, test_data_db
):
    client = ibis.impala.connect(
        port=impala_port,
        hdfs_client=hdfs,
        host=impala_host,
        database=test_data_db,
    )
    client.close()
    assert len(client.con.connection_pool) == 0
