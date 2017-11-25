import os

import pytest


class SQLite(object):
    check_dtype = False

    @classmethod
    def connect(cls, backend):
        path = os.environ.get('IBIS_TEST_SQLITE_DB_PATH', 'ibis_testing.db')
        if not os.path.exists(path):
            pytest.skip('SQLite testing db {} does not exist'.format(path))
        else:
            con = backend.connect(path)
            return con
