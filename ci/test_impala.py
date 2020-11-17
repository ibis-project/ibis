import io

import ibis

hdfs_conn = ibis.impala.hdfs_connect(
    host='localhost',
    port=50070,
    auth_mechanism='NOSASL',
    verify=True,
    user='hdfs',
)
hdfs_conn.put('/tmp/test_file', io.BytesIO(b'1234'))
