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

import ibis.expr.types as ir
import ibis.sql.alchemy as alch
from .compiler import SQLiteDialect


class SQLiteTable(alch.AlchemyTable):
    pass


class SQLiteDatabase(alch.AlchemyClient):

    dialect = SQLiteDialect

    def __init__(self, path):
        self.name = path
        uri = 'sqlite:///{0}'.format(path)
        alch.AlchemyClient.__init__(self, uri)

    @property
    def client(self):
        return self

    def table(self, name):
        """
        Create a table expression that references a particular table in the
        SQLite database

        Parameters
        ----------
        name : string

        Returns
        -------
        table : TableExpr
        """
        alch_table = self._get_sqla_table(name)
        node = SQLiteTable(alch_table, self)
        return self._table_expr_klass(node)

    def drop_table(self):
        pass

    def create_table(self, name, expr=None):
        pass

    @property
    def _table_expr_klass(self):
        return ir.TableExpr
