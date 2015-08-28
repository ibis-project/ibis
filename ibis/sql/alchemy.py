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

import operator

import sqlalchemy as sa
import sqlalchemy.sql as sql

from ibis.client import SQLClient
import ibis.common as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.sql.compiler as comp
import ibis.sql.ddl as ddl
import ibis


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
    sa.types.BOOLEAN: dt.Boolean,
    sa.types.BIGINT: dt.Int64,
    sa.types.FLOAT: dt.Double,
    sa.types.REAL: dt.Double,

    sa.types.TEXT: dt.String,
    sa.types.NullType: dt.String,
    sa.types.Text: dt.String,
}

_sqla_type_to_ibis = dict((v, k) for k, v in
                          _ibis_type_to_sqla.items())
_sqla_type_to_ibis.update(_sqla_type_mapping)


def schema_from_table(table):
    # Convert SQLA table to Ibis schema
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


def table_from_schema(name, meta, schema):
    # Convert Ibis schema to SQLA table
    sqla_cols = []

    for cname, itype in zip(schema.names, schema.types):
        ctype = _ibis_type_to_sqla[type(itype)]

        col = sa.Column(cname, ctype, nullable=itype.nullable)
        sqla_cols.append(col)

    return sa.Table(name, meta, *sqla_cols)


def _fixed_arity_call(sa_func, arity):
    def formatter(translator, expr):
        if arity != len(expr.op().args):
            raise com.IbisError('incorrect number of args')

        return _varargs_call(sa_func, translator, expr)

    return formatter


def _varargs_call(sa_func, translator, expr):
    op = expr.op()
    trans_args = [translator.translate(arg) for arg in op.args]
    return sa_func(*trans_args)


def _table_column(translator, expr):
    op = expr.op()
    table = op.table
    ctx = translator.context

    # If the column does not originate from the table set in the current SELECT
    # context, we should format as a subquery
    # if translator.permit_subquery and ctx.is_foreign_expr(table):
    #     proj_expr = table.projection([field_name]).to_array()
    #     return _table_array_view(translator, proj_expr)

    sa_table = ctx.get_ref(table)
    if sa_table is None:
        sa_table = table.op().sqla_table

    return getattr(sa_table.c, op.name)


def _reduction(sa_func):
    def formatter(translator, expr):
        op = expr.op()

        # HACK: support trailing arguments
        arg, where = op.args[:2]

        return _reduction_format(translator, sa_func, arg, where)
    return formatter


def _reduction_format(translator, sa_func, arg, where):
    if where is not None:
        case = where.ifelse(arg, ibis.NA)
        arg = translator.translate(case)
    else:
        arg = translator.translate(arg)

    return sa_func(arg)


def _literal(translator, expr):
    return expr.op().value


_expr_rewrites = {

}

_operation_registry = {
    ops.And: _fixed_arity_call(sql.and_, 2),
    ops.Or: _fixed_arity_call(sql.or_, 2),

    ops.Count: _reduction(sa.func.count),
    ops.Sum: _reduction(sa.func.sum),

    ir.Literal: _literal,

    ops.TableColumn: _table_column,
}


# TODO: unit tests for each of these
_binary_ops = {
    # Binary arithmetic
    ops.Add: operator.add,
    ops.Subtract: operator.sub,
    ops.Multiply: operator.mul,
    ops.Divide: operator.truediv,
    ops.Power: operator.pow,
    ops.Modulus: operator.mod,

    # Comparisons
    ops.Equals: operator.eq,
    ops.NotEquals: operator.ne,
    ops.Less: operator.lt,
    ops.LessEqual: operator.le,
    ops.Greater: operator.gt,
    ops.GreaterEqual: operator.ge,

    # Boolean comparisons

    # TODO
}

for _k, _v in _binary_ops.items():
    _operation_registry[_k] = _fixed_arity_call(_v, 2)


def to_sqlalchemy(expr):
    builder = AlchemyQueryBuilder(expr)
    ast = builder.get_result()
    query = ast.queries[0]
    return query.compile()


class AlchemyQueryBuilder(comp.QueryBuilder):

    @property
    def _context_class(self):
        return AlchemyContext

    def _make_union(self):
        raise NotImplementedError

    def _make_select(self):
        builder = AlchemySelectBuilder(self.expr, self.context)
        return builder.get_result()


class AlchemySelectBuilder(comp.SelectBuilder):

    @property
    def _select_class(self):
        return AlchemySelect

    def _convert_group_by(self, exprs):
        return exprs


class AlchemyContext(comp.QueryContext):

    def _compile_subquery(self, expr):
        from ibis.sql.compiler import to_sql
        sub_ctx = self.subcontext()
        return to_sql(expr, context=sub_ctx)

    def record_table(self, expr):
        # Store SQLAlchemy table
        op = expr.op()

        if not isinstance(op, AlchemyTable):
            raise TypeError(type(op))

        self.set_ref(expr, op.sqla_table)


class AlchemyTable(ops.DatabaseTable):

    def __init__(self, table, source):
        self.sqla_table = table

        schema = schema_from_table(table)
        name = table.name

        ops.TableNode.__init__(self, [name, schema, source])
        ops.HasSchema.__init__(self, schema, name=name)


class AlchemyClient(SQLClient):

    def _sqla_table_to_expr(self, table):
        node = AlchemyTable(table, self)
        return self._table_expr_klass(node)


class AlchemyExprTranslator(ddl.ExprTranslator):

    _registry = _operation_registry
    _rewrites = _expr_rewrites

    def name(self, translated, name, force=True):
        return translated.label(name)


class AlchemySelect(ddl.Select):

    def compile(self):
        # Can't tell if this is a hack or not. Revisit later
        self.context.set_query(self)

        frag = self._compile_table_set()
        steps = [self._add_select,
                 self._add_groupby,
                 self._add_where,
                 self._add_order_by,
                 self._add_limit]

        for step in steps:
            frag = step(frag)

        return frag

    def _compile_table_set(self):
        helper = _AlchemyTableSet(self, self.table_set)
        return helper.get_result()

    def _add_select(self, table_set):
        to_select = []
        for expr in self.select_set:
            if isinstance(expr, ir.ValueExpr):
                arg = self._translate(expr, named=True)
            elif isinstance(expr, ir.TableExpr):
                if expr.equals(self.table_set):
                    # the select * case
                    arg = table_set
                else:
                    arg = self.context.get_ref(expr)
                    if arg is None:
                        raise ValueError(expr)

            to_select.append(arg)

        return sa.select(to_select).select_from(table_set)

    def _add_groupby(self, fragment):
        # GROUP BY and HAVING
        if not len(self.group_by):
            return fragment

        group_keys = [self._translate(arg) for arg in self.group_by]
        fragment = fragment.group_by(*group_keys)

        if len(self.having) > 0:
            having_args = [self._translate(arg) for arg in self.having]
            having_clause = _and_all(having_args)
            fragment = fragment.having(having_clause)

        return fragment

    def _add_where(self, fragment):
        if not len(self.where):
            return fragment

        args = [self._translate(pred, permit_subquery=True)
                for pred in self.where]
        clause = _and_all(args)
        return fragment.where(clause)

    def _add_order_by(self, fragment):
        if not len(self.order_by):
            return fragment

    def _add_limit(self, fragment):
        if self.limit is None:
            return fragment

    def _translate(self, expr, context=None, named=False,
                   permit_subquery=False):
        translator = AlchemyExprTranslator(expr, context=context, named=named,
                                           permit_subquery=permit_subquery)
        return translator.get_result()


class _AlchemyTableSet(ddl._TableSetFormatter):

    def get_result(self):
        # Got to unravel the join stack; the nesting order could be
        # arbitrary, so we do a depth first search and push the join tokens
        # and predicates onto a flat list, then format them
        op = self.expr.op()

        if isinstance(op, ops.Join):
            self._walk_join_tree(op)
        else:
            self.join_tables.append(self._format_table(self.expr))

        result = self.join_tables[0]
        for jtype, table, preds in zip(self.join_types,
                                       self.join_tables[1:],
                                       self.join_predicates):
            if len(preds):
                sqla_preds = [self._translate(pred) for pred in preds]
                onclause = _and_all(sqla_preds)
            else:
                onclause = None

            if jtype in (ops.InnerJoin, ops.CrossJoin):
                result = result.join(table, onclause)
            elif jtype is ops.LeftJoin:
                result = result.join(table, onclause, isouter=True)
            elif jtype is ops.RightJoin:
                result = table.join(result, onclause, isouter=True)
            elif jtype is ops.OuterJoin:
                result = result.outerjoin(table, onclause)
            else:
                raise NotImplementedError(jtype)

        return result

    def _get_join_type(self, op):
        return type(op)

    def _format_table(self, expr):
        return _get_sqla_table(self.context, expr)


def _and_all(clauses):
    result = clauses[0]
    for clause in clauses[1:]:
        result = sql.and_(result, clause)
    return result


def _get_sqla_table(ctx, expr):
    ref_expr = expr
    op = ref_op = expr.op()
    if isinstance(op, ops.SelfReference):
        ref_expr = op.table
        ref_op = ref_expr.op()

    if isinstance(ref_op, AlchemyTable):
        return ref_op.sqla_table
    else:
        # A subquery
        if ctx.is_extracted(ref_expr):
            # Was put elsewhere, e.g. WITH block, we just need to grab its
            # alias
            return ctx.get_ref(expr)

        return ctx.get_compiled_expr(expr)


class AlchemyUnion(ddl.Union):

    def compile(self):
        context = self.context

        if self.distinct:
            union_keyword = 'UNION'
        else:
            union_keyword = 'UNION ALL'

        left_set = context.get_compiled_expr(self.left)
        right_set = context.get_compiled_expr(self.right)

        query = '{0}\n{1}\n{2}'.format(left_set, union_keyword, right_set)
        return query
