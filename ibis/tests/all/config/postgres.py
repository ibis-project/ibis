import getpass
import os

from ibis.tests.all.config.backendtestconfiguration import BackendTestConfiguration


class Postgres(BackendTestConfiguration):
    @classmethod
    def connect(cls, backend):
        PG_USER = os.environ.get(
            'IBIS_POSTGRES_USER',
            os.environ.get('PGUSER', getpass.getuser())
        )
        PG_PASS = os.environ.get(
            'IBIS_POSTGRES_PASS', os.environ.get('PGPASS')
        )
        PG_HOST = os.environ.get('PGHOST', 'localhost')
        IBIS_TEST_POSTGRES_DB = os.environ.get(
            'IBIS_TEST_POSTGRES_DB',
            os.environ.get('PGDATABASE', 'ibis_testing')
        )
        return backend.connect(
            host=PG_HOST,
            user=PG_USER,
            password=PG_PASS,
            database=IBIS_TEST_POSTGRES_DB,
        )
