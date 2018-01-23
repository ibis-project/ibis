from __future__ import absolute_import

import os

import pytest

from ibis.tests.all.config.backendtestconfiguration import (
    BackendTestConfiguration
)


class Parquet(BackendTestConfiguration):

    required_modules = 'pyarrow',
    check_names = False

    @classmethod
    def connect(cls, module):
        test_data_directory = os.environ.get('IBIS_TEST_DATA_DIRECTORY')
        filename = os.path.join(
            test_data_directory, 'functional_alltypes.parquet'
        )
        if not os.path.exists(test_data_directory):
            pytest.skip('test data directory not found')
        if not os.path.exists(filename):
            pytest.skip(
                'test data set functional_alltypes.parquet not found in '
                'test data directory'
            )
        else:
            return module.connect(test_data_directory)
