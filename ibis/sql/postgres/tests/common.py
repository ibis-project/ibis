# Copyright 2015 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import getpass
import os

import pytest

from ibis.sql.postgres.compiler import PostgreSQLExprTranslator
import ibis.sql.postgres.api as api

from sqlalchemy.dialects.postgresql import dialect as postgres_dialect


PG_USER = os.environ.get('IBIS_POSTGRES_USER', getpass.getuser())
PG_PASS = os.environ.get('IBIS_POSTGRES_PASS')


@pytest.mark.postgresql
class PostgreSQLTests(object):

    @classmethod
    def setUpClass(cls):
        cls.env = PostgreSQLTestEnv()
        cls.dialect = postgres_dialect()

        E = cls.env

        cls.con = api.connect(host=E.host, user=E.user, password=E.password,
                              database=E.database_name)
        cls.alltypes = cls.con.table('functional_alltypes')

    def _check_expr_cases(self, cases, context=None, named=False):
        for expr, expected in cases:
            result = self._translate(expr, named=named, context=context)

            compiled = result.compile(dialect=self.dialect)
            ex_compiled = expected.compile(dialect=self.dialect)

            assert str(compiled) == str(ex_compiled)

    def _translate(self, expr, named=False, context=None):
        translator = PostgreSQLExprTranslator(
            expr, context=context, named=named
        )
        return translator.get_result()

    def _to_sqla(self, table):
        return table.op().sqla_table

    def _check_e2e_cases(self, cases):
        for expr, expected in cases:
            result = self.con.execute(expr)
            assert result == expected


class PostgreSQLTestEnv(object):

    def __init__(self):
        if PG_PASS:
            creds = '{0}:{1}'.format(PG_USER, PG_PASS)
        else:
            creds = PG_USER

        self.user = PG_USER
        self.password = PG_PASS
        self.host = 'localhost'
        self.database_name = 'ibis_testing'

        self.db_url = 'postgresql://{0}@localhost/ibis_testing'.format(creds)
