from __future__ import absolute_import

import getpass
import os

from ibis.tests.all.config.backendtestconfiguration import (
    BackendTestConfiguration
)


class Postgres(BackendTestConfiguration):

    required_modules = 'sqlalchemy', 'psycopg2',

    @classmethod
    def connect(cls, module):
        PG_USER = os.environ.get(
            'IBIS_TEST_POSTGRES_USER',
            os.environ.get('PGUSER', getpass.getuser())
        )
        PG_PASS = os.environ.get(
            'IBIS_TEST_POSTGRES_PASS', os.environ.get('PGPASS')
        )
        PG_HOST = os.environ.get('PGHOST', 'localhost')
        IBIS_TEST_POSTGRES_DB = os.environ.get(
            'IBIS_TEST_POSTGRES_DATABASE',
            os.environ.get('PGDATABASE', 'ibis_testing')
        )
        return module.connect(
            host=PG_HOST,
            user=PG_USER,
            password=PG_PASS,
            database=IBIS_TEST_POSTGRES_DB,
        )
