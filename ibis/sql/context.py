# Copyright 2014 Cloudera Inc.
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

import ibis.expr.base as ir


#----------------------------------------------------------------------
# The QueryContext (temporary name) will store useful information like table
# alias names for converting value expressions to SQL.


class QueryContext(object):

    """

    """

    def __init__(self, indent=2):
        self.table_aliases = {}
        self.extracted_subexprs = set()
        self.subquery_memo = {}
        self.indent = indent

    @property
    def top_context(self):
        return self

    def _get_table_key(self, table):
        if isinstance(table, ir.TableExpr):
            table = table.op()
        return id(table)

    def is_extracted(self, expr):
        key = self._get_table_key(expr)
        return key in self.top_context.extracted_subexprs

    def set_extracted(self, expr):
        key = self._get_table_key(expr)
        self.extracted_subexprs.add(key)
        self.make_alias(expr)

    def get_formatted_query(self, expr):
        from ibis.sql.compiler import to_sql

        this = self.top_context

        key = self._get_table_key(expr)
        if key in this.subquery_memo:
            return this.subquery_memo[key]

        op = expr.op()
        if isinstance(op, ir.SQLQueryResult):
            result = op.query
        else:
            result = to_sql(expr, context=self.subcontext())
        this.subquery_memo[key] = result
        return result

    def make_alias(self, table_expr):
        i = len(self.table_aliases)
        alias = 't%d' % i
        self.set_alias(table_expr, alias)

    def has_alias(self, table_expr):
        key = self._get_table_key(table_expr)
        return key in self.table_aliases

    def need_aliases(self):
        return len(self.table_aliases) > 1

    def set_alias(self, table_expr, alias):
        key = self._get_table_key(table_expr)
        self.table_aliases[key] = alias

    def get_alias(self, table_expr):
        """
        Get the alias being used throughout a query to refer to a particular
        table or inline view
        """
        key = self._get_table_key(table_expr)

        top = self.top_context
        if self is top:
            if self.is_extracted(table_expr):
                return top.table_aliases.get(key)

        return self.table_aliases.get(key)

    def subcontext(self):
        return SubContext(self)


class SubContext(QueryContext):

    def __init__(self, parent):
        self.parent = parent
        super(SubContext, self).__init__(indent=parent.indent)

    @property
    def top_context(self):
        return self.parent.top_context
