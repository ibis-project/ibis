import pytest
import ibis

pytest.importorskip('sqlalchemy')
pytest.importorskip('impala.dbapi')

pytestmark = pytest.mark.impala


def test_connection_pool_size(hdfs, env, test_data_db):
    client = ibis.impala.connect(
        port=env.impala_port,
        hdfs_client=hdfs,
        host=env.impala_host,
        database=test_data_db,
    )
    assert len(client.con.connection_pool) == 1


def test_connection_pool_size_after_close(hdfs, env, test_data_db):
    client = ibis.impala.connect(
        port=env.impala_port,
        hdfs_client=hdfs,
        host=env.impala_host,
        database=test_data_db,
    )
    client.close()
    assert not client.con.connection_pool
