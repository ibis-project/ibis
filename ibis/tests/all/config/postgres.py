import os

from ibis.tests.all.config.backendtestconfiguration import (
    BackendTestConfiguration
)


class Postgres(BackendTestConfiguration):

    required_modules = 'sqlalchemy', 'psycopg2'

    @classmethod
    def connect(cls, backend):
        user = os.environ.get('IBIS_TEST_POSTGRES_USER',
                              os.environ.get('PGUSER', 'postgres'))
        password = os.environ.get('IBIS_TEST_POSTGRES_PASSWORD',
                                  os.environ.get('PGPASS'))
        host = os.environ.get('IBIS_TEST_POSTGRES_HOST',
                              os.environ.get('PGHOST', 'localhost'))
        database = os.environ.get('IBIS_TEST_POSTGRES_DATABASE',
                                  os.environ.get('PGDATABASE', 'ibis_testing'))
        return backend.connect(host=host,
                               user=user,
                               password=password,
                               database=database)
