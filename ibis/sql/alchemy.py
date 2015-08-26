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

import sqlalchemy as sa
import sqlalchemy.sql as sql

from ibis.client import SQLClient
from ibis.sql.ddl import ExprTranslator, Select, Union
import ibis.common as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops


_ibis_type_to_sqla = {
    dt.Int8: sa.types.SmallInteger,
    dt.Int16: sa.types.SmallInteger,
    dt.Int32: sa.types.Integer,
    dt.Int64: sa.types.BigInteger,

    # Mantissa-based
    dt.Float: sa.types.Float(precision=24),
    dt.Double: sa.types.Float(precision=53),

    dt.Boolean: sa.types.Boolean,

    dt.String: sa.types.String,

    dt.Timestamp: sa.types.DateTime,

    dt.Decimal: sa.types.NUMERIC,
}

_sqla_type_mapping = {
    sa.types.SmallInteger: dt.Int16,
    sa.types.FLOAT: dt.Double,
    sa.types.REAL: dt.Double,

    sa.types.NullType: dt.String,
    sa.types.Text: dt.String,
}

_sqla_type_to_ibis = dict((v, k) for k, v in
                          _ibis_type_to_sqla.items())
_sqla_type_to_ibis.update(_sqla_type_mapping)


def schema_from_table(table):
    names = table.columns.keys()

    types = []
    for c in table.columns.values():
        type_class = type(c.type)

        if c.type in _sqla_type_to_ibis:
            ibis_class = _sqla_type_to_ibis[c.type]
        elif type_class in _sqla_type_to_ibis:
            ibis_class = _sqla_type_to_ibis[type_class]
        else:
            raise NotImplementedError(c.type)

        t = ibis_class(c.nullable)
        types.append(t)

    return dt.Schema(names, types)


def _fixed_arity_call(sa_func, arity):
    def formatter(translator, expr):
        op = expr.op()
        if arity != len(op.args):
            raise com.IbisError('incorrect number of args')

        trans_args = [translator.translate(arg) for arg in op.args]
        return sa_func(*trans_args)

    return formatter


_expr_rewrites = {

}


_operation_registry = {
    ops.And: _fixed_arity_call(sql.and_, 2),
    ops.Or: _fixed_arity_call(sql.or_, 2),
}


class AlchemyTable(ops.DatabaseTable):

    def __init__(self, table, source):
        schema = schema_from_table(table)
        name = table.name
        ops.DatabaseTable.__init__(self, name, schema, source)


class AlchemyClient(SQLClient):

    def _sqla_table_to_expr(self, table):
        node = AlchemyTable(table, self)
        return self._table_expr_klass(node)


class AlchemyExprTranslator(ExprTranslator):

    _registry = _operation_registry
    _rewrites = _expr_rewrites

    def name(self, translated, name, force=True):
        pass


class AlchemySelect(Select):

    def compile(self):
        # Can't tell if this is a hack or not. Revisit later
        self.context.set_query(self)

        table_set = self._compile_table_set()

        frag = self._add_select_clauses(table_set)
        frag = self._add_groupby_clauses(frag)
        frag = self._add_where_clauses(frag)
        frag = self._add_order_by_clauses(frag)

    def _compile_table_set(self):
        pass

    def _add_select_clauses(self, table_set):
        pass

    def _add_groupby_clauses(self, fragment):
        # GROUP BY and HAVING
        pass

    def _add_where_clauses(self, fragment):
        pass

    def _add_order_by(self, fragment):
        pass

    def _translate(self, expr, context=None, named=False,
                   permit_subquery=False):
        translator = AlchemyExprTranslator(expr, context=context, named=named,
                                           permit_subquery=permit_subquery)
        return translator.get_result()


class AlchemyUnion(Union):

    def compile(self):
        context = self.context

        if self.distinct:
            union_keyword = 'UNION'
        else:
            union_keyword = 'UNION ALL'

        left_set = context.get_formatted_query(self.left)
        right_set = context.get_formatted_query(self.right)

        query = '{0}\n{1}\n{2}'.format(left_set, union_keyword, right_set)
        return query
