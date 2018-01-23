from __future__ import absolute_import

import ibis
from ibis.impala.tests.common import IbisTestEnv

from ibis.tests.all.config.backendtestconfiguration import (
    BackendTestConfiguration,
    UnorderedSeriesComparator,
)


ENV = IbisTestEnv()


class Impala(UnorderedSeriesComparator, BackendTestConfiguration):

    required_modules = 'sqlalchemy', 'hdfs', 'impala.dbapi',

    supports_arrays = True
    supports_arrays_outside_of_select = False
    check_dtype = False

    @classmethod
    def connect(cls, module):
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

        return module.connect(
            host=ENV.impala_host,
            port=ENV.impala_port,
            auth_mechanism=ENV.auth_mechanism,
            hdfs_client=hc,
            database='ibis_testing'
        )
