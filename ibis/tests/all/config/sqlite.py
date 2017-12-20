import os

import pytest

from ibis.tests.all.config.backendtestconfiguration import (
    BackendTestConfiguration
)


class SQLite(BackendTestConfiguration):

    supports_arrays = False
    supports_window_operations = False
    check_dtype = False

    @classmethod
    def connect(cls, backend):
        path = os.environ.get('IBIS_TEST_SQLITE_DATABASE', 'ibis_testing.db')
        if not os.path.exists(path):
            pytest.skip('SQLite testing db {} does not exist'.format(path))
        else:
            con = backend.connect(path)
            return con
