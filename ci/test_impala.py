import os
import io
import pprint

import ibis


hdfs_host = os.environ.get('IBIS_TEST_NN_HOST', 'localhost'),
hdfs_port = int(os.environ.get('IBIS_TEST_WEBHDFS_PORT', 50070))
auth_mechanism = os.environ.get('IBIS_TEST_AUTH_MECH', 'NOSASL')
webhdfs_user = os.environ.get('IBIS_TEST_WEBHDFS_USER', 'hdfs')

impala_host = os.environ.get('IBIS_TEST_IMPALA_HOST', 'localhost')
impala_port = int(os.environ.get('IBIS_TEST_IMPALA_PORT', 21050))

hdfs_conn = ibis.hdfs_connect(
    host=hdfs_host,
    port=hdfs_port,
    auth_mechanism=auth_mechanism,
    verify=auth_mechanism not in ['GSSAPI', 'LDAP'],
    user=webhdfs_user,
)
impala_conn = ibis.impala.connect(
    host=impala_host,
    port=impala_port,
    auth_mechanism=auth_mechanism,
    hdfs_client=hdfs_conn,
    pool_size=16,
)
pprint.pprint(globals())
impala_conn.hdfs.put('/tmp/test_file', io.BytesIO(b'1234'))
