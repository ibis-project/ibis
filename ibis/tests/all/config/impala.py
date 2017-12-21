from ibis.tests.all.config.backendtestconfiguration import (
    BackendTestConfiguration,
    UnorderedSeriesComparator,
)

import ibis
from ibis.impala.tests.common import IbisTestEnv


ENV = IbisTestEnv()


class Impala(UnorderedSeriesComparator, BackendTestConfiguration):
    supports_arrays = True
    supports_arrays_outside_of_select = False
    check_dtype = False

    @classmethod
    def connect(cls, backend):
        hc = ibis.hdfs_connect(
            host=ENV.nn_host,
            port=ENV.webhdfs_port,
            auth_mechanism=ENV.auth_mechanism,
            verify=ENV.auth_mechanism not in ['GSSAPI', 'LDAP'],
            user=ENV.webhdfs_user
        )
        auth_mechanism = ENV.auth_mechanism
        if auth_mechanism == 'GSSAPI' or auth_mechanism == 'LDAP':
            print("Warning: ignoring invalid Certificate Authority errors")
        return ibis.impala.connect(
            host=ENV.impala_host,
            port=ENV.impala_port,
            auth_mechanism=ENV.auth_mechanism,
            hdfs_client=hc,
            database='ibis_testing'
        )
