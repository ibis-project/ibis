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

import os
import pytest

from ibis.sql.sqlite.compiler import SQLiteExprTranslator
import ibis.sql.sqlite.api as api
import ibis.util as util

from sqlalchemy.dialects.sqlite import dialect as sqlite_dialect


@pytest.mark.sqlite
class SQLiteTests(object):

    @classmethod
    def setUpClass(cls):
        cls.env = SQLiteTestEnv()
        cls.dialect = sqlite_dialect()
        cls.con = api.connect(cls.env.db_path)
        cls.alltypes = cls.con.table('functional_alltypes')

    def _check_expr_cases(self, cases, context=None, named=False):
        for expr, expected in cases:
            result = self._translate(expr, named=named, context=context)

            compiled = result.compile(dialect=self.dialect)
            ex_compiled = expected.compile(dialect=self.dialect)

            assert str(compiled) == str(ex_compiled)

    def _translate(self, expr, named=False, context=None):
        translator = SQLiteExprTranslator(expr, context=context, named=named)
        return translator.get_result()

    def _to_sqla(self, table):
        return table.op().sqla_table

    def _check_e2e_cases(self, cases):
        for expr, expected in cases:
            result = self.con.execute(expr)
            assert result == expected


class SQLiteTestEnv(object):

    def __init__(self):
        self.db_path = os.environ.get('IBIS_TEST_SQLITE_DB_PATH',
                                      'ibis_testing.db')
