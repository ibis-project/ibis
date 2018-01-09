from __future__ import absolute_import

import os

import ibis

from ibis.tests.all.config.backendtestconfiguration import (
    BackendTestConfiguration
)


class Clickhouse(BackendTestConfiguration):
    check_dtype = False

    @classmethod
    def connect(cls, backend):
        return ibis.clickhouse.connect(
            host=os.environ.get('IBIS_CLICKHOUSE_HOST', 'localhost'),
            port=int(os.environ.get('IBIS_CLICKHOUSE_PORT', 9000)),
            database=os.environ.get('IBIS_TEST_DATA_DB', 'ibis_testing'),
            user=os.environ.get('IBIS_CLICKHOUSE_USER', 'default'),
            password=os.environ.get('IBIS_CLICKHOUSE_PASS', ''),
        )

    @classmethod
    def functional_alltypes(cls, con):
        t = con.database().functional_alltypes
        return t.mutate(bool_col=t.bool_col == 1)
