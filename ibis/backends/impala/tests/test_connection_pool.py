import ibis


def test_connection_pool_size(hdfs, env, test_data_db):
    client = ibis.impala.connect(
        port=env.impala_port,
        hdfs_client=hdfs,
        host=env.impala_host,
        database=test_data_db,
    )

    # the client cursor may or may not be GC'd, so the connection
    # pool will contain either zero or one cursor
    assert len(client.con.connection_pool) in (0, 1)


def test_connection_pool_size_after_close(hdfs, env, test_data_db):
    client = ibis.impala.connect(
        port=env.impala_port,
        hdfs_client=hdfs,
        host=env.impala_host,
        database=test_data_db,
    )
    client.close()
    assert not client.con.connection_pool
