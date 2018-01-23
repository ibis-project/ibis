from __future__ import absolute_import

import os

import pytest

from ibis.tests.all.config.backendtestconfiguration import (
    BackendTestConfiguration
)


class SQLite(BackendTestConfiguration):

    required_modules = 'sqlalchemy',

    supports_arrays = False
    supports_arrays_outside_of_select = supports_arrays
    supports_window_operations = False

    check_dtype = False

    @classmethod
    def connect(cls, module):
        path = os.environ.get('IBIS_TEST_SQLITE_DATABASE', 'ibis_testing.db')
        if not os.path.exists(path):
            pytest.skip('SQLite testing db at {} does not exist'.format(path))
        return module.connect(path)
