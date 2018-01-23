from __future__ import absolute_import

import os

import pytest

import ibis

from ibis.tests.all.config.backendtestconfiguration import (
    BackendTestConfiguration
)


class CSV(BackendTestConfiguration):
    check_names = False

    @classmethod
    def connect(cls, module):
        test_data_directory = os.environ.get('IBIS_TEST_DATA_DIRECTORY')
        filename = os.path.join(test_data_directory, 'functional_alltypes.csv')
        if not os.path.exists(test_data_directory):
            pytest.skip('test data directory not found')
        if not os.path.exists(filename):
            pytest.skip(
                'test data set functional_alltypes.csv not found in '
                'test data directory'
            )
        else:
            return module.connect(test_data_directory)

    @classmethod
    def functional_alltypes(cls, con):
        schema = ibis.schema([
            ('bool_col', 'boolean'),
            ('string_col', 'string'),
        ])
        return con.table('functional_alltypes', schema=schema)
