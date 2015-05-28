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

import ibis.expr.operations as ops
import ibis.expr.types as ir


#----------------------------------------------------------------------
# The QueryContext (temporary name) will store useful information like table
# alias names for converting value expressions to SQL.


class QueryContext(object):

    """

    """

    def __init__(self, indent=2, parent=None):
        self.table_aliases = {}
        self.extracted_subexprs = set()
        self.subquery_memo = {}
        self.indent = indent
        self.parent = parent

        self.always_alias = False

        self.query = None

        self._table_key_memo = {}

    @property
    def top_context(self):
        if self.parent is None:
            return self
        else:
            return self.parent.top_context

    def _get_table_key(self, table):
        if isinstance(table, ir.TableExpr):
            table = table.op()

        k = id(table)
        if k in self._table_key_memo:
            return self._table_key_memo[k]
        else:
            val = table._repr()
            self._table_key_memo[k] = val
            return val

    def set_always_alias(self):
        self.always_alias = True

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
        if isinstance(op, ops.SQLQueryResult):
            result = op.query
        else:
            sub_ctx = self.subcontext()
            result = to_sql(expr, context=sub_ctx)
        this.subquery_memo[key] = result
        return result

    def make_alias(self, table_expr):
        i = len(self.table_aliases)

        key = self._get_table_key(table_expr)

        # Get total number of aliases up and down the tree at this point; if we
        # find the table prior-aliased along the way, however, we reuse that
        # alias
        ctx = self
        while ctx.parent is not None:
            ctx = ctx.parent

            if key in ctx.table_aliases:
                alias = ctx.table_aliases[key]
                self.set_alias(table_expr, alias)
                return

            i += len(ctx.table_aliases)

        alias = 't%d' % i
        self.set_alias(table_expr, alias)

    def has_alias(self, table_expr, parent_contexts=False):
        key = self._get_table_key(table_expr)
        return self._key_in(key, 'table_aliases',
                            parent_contexts=parent_contexts)

    def _key_in(self, key, memo_attr, parent_contexts=False):
        if key in getattr(self, memo_attr):
            return True

        ctx = self
        while parent_contexts and ctx.parent is not None:
            ctx = ctx.parent
            if key in getattr(ctx, memo_attr):
                return True

        return False

    def need_aliases(self):
        return self.always_alias or len(self.table_aliases) > 1

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

        if self.is_extracted(table_expr):
            return top.table_aliases.get(key)

        return self.table_aliases.get(key)

    def subcontext(self):
        return QueryContext(indent=self.indent, parent=self)

    # Maybe temporary hacks for correlated / uncorrelated subqueries

    def set_query(self, query):
        self.query = query

    def is_foreign_expr(self, expr):
        from ibis.expr.analysis import ExprValidator

        # The expression isn't foreign to us. For example, the parent table set
        # in a correlated WHERE subquery
        if self.has_alias(expr, parent_contexts=True):
            return False

        exprs = [self.query.table_set] + self.query.select_set
        validator = ExprValidator(exprs)
        return not validator.validate(expr)
