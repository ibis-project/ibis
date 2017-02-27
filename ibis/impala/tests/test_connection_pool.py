import pytest
import ibis

pytest.importorskip('sqlalchemy')
pytest.importorskip('impala.dbapi')

from ibis.impala.tests.common import IbisTestEnv  # noqa


@pytest.fixture
def env():
    return IbisTestEnv()


@pytest.fixture
def hdfs_client(env):
    return ibis.hdfs_connect(
        host=env.nn_host,
        port=int(env.webhdfs_port),
        auth_mechanism=env.auth_mechanism,
        user=env.webhdfs_user,
    )


@pytest.mark.impala
def test_connection_pool_size(env, hdfs_client):
    client = ibis.impala.connect(
        port=int(env.impala_port),
        hdfs_client=hdfs_client,
        host=env.impala_host,
        database=env.test_data_db,
    )
    assert client.con.connection_pool_size == 1


@pytest.mark.impala
def test_connection_pool_size_after_close(env, hdfs_client):
    client = ibis.impala.connect(
        port=int(env.impala_port),
        hdfs_client=hdfs_client,
        host=env.impala_host,
        database=env.test_data_db,
    )
    client.close()
    assert client.con.connection_pool_size == 0
