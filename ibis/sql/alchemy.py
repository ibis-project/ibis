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

import numbers
import operator
import functools
import contextlib
from collections import OrderedDict

import six

import sqlalchemy as sa
import sqlalchemy.sql as sql

from sqlalchemy.sql.elements import Over as _Over
from sqlalchemy.ext.compiler import compiles as sa_compiles

import numpy as np
import pandas as pd

from ibis.client import SQLClient, AsyncQuery, Query, Database
from ibis.sql.compiler import Select, Union, TableSetFormatter

from ibis import compat

import ibis.common as com
import ibis.expr.schema as sch
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.sql.compiler as comp
import ibis.sql.transforms as transforms
import ibis.util as util
import ibis


_ibis_type_to_sqla = OrderedDict([
    (dt.Int8, sa.SmallInteger),
    (dt.Int16, sa.SmallInteger),
    (dt.Int32, sa.Integer),
    (dt.Int64, sa.BigInteger),

    # Mantissa-based
    (dt.Float, sa.Float(precision=24)),
    (dt.Double, sa.Float(precision=53)),

    (dt.Boolean, sa.Boolean),

    (dt.String, sa.Text),

    (dt.Date, sa.Date),
    (dt.Time, sa.Time),

    (dt.Decimal, sa.NUMERIC),
    (dt.Null, sa.types.NullType),
    (dt.Binary, sa.Binary)
])

_sqla_type_mapping = OrderedDict([
    (sa.SmallInteger, dt.Int16),
    (sa.SMALLINT, dt.Int16),
    (sa.Integer, dt.Int32),
    (sa.INTEGER, dt.Int32),
    (sa.BigInteger, dt.Int64),
    (sa.BIGINT, dt.Int64),
    (sa.Boolean, dt.Boolean),
    (sa.BOOLEAN, dt.Boolean),
    (sa.Float, dt.Double),
    (sa.FLOAT, dt.Double),
    (sa.REAL, dt.Float),
    (sa.String, dt.String),
    (sa.VARCHAR, dt.String),
    (sa.CHAR, dt.String),
    (sa.Text, dt.String),
    (sa.TEXT, dt.String),
    (sa.BINARY, dt.Binary),
    (sa.Binary, dt.Binary),
    (sa.DATE, dt.Date),
    (sa.Time, dt.Time),
    (sa.types.NullType, dt.Null)
])

_sqla_type_to_ibis = OrderedDict(
    [(v, k) for k, v in _ibis_type_to_sqla.items()] +
    list(_sqla_type_mapping.items())
)


def sqlalchemy_type_to_ibis_type(
    column_type, nullable=True, default_timezone=None
):
    type_class = type(column_type)
    if isinstance(column_type, sa.types.NUMERIC):
        return dt.Decimal(
            column_type.precision, column_type.scale, nullable=nullable
        )

    if type_class in _sqla_type_to_ibis:
        ibis_class = _sqla_type_to_ibis[type_class]
    elif isinstance(column_type, sa.DateTime):
        return dt.Timestamp(
            timezone=default_timezone if column_type.timezone else None,
            nullable=nullable
        )
    elif isinstance(column_type, sa.ARRAY):
        dimensions = column_type.dimensions
        if dimensions is not None and dimensions != 1:
            raise NotImplementedError(
                'Nested array types not yet supported'
            )
        value_type = sqlalchemy_type_to_ibis_type(
            column_type.item_type,
            default_timezone=default_timezone,
        )

        def make_array_type(nullable, value_type=value_type):
            return dt.Array(value_type, nullable=nullable)

        ibis_class = make_array_type
    else:
        try:
            ibis_class = next(
                v for k, v in reversed(_sqla_type_mapping.items())
                if isinstance(column_type, k)
            )
        except StopIteration:
            raise NotImplementedError(
                'Unable to convert SQLAlchemy type {} to ibis type'.format(
                    column_type
                )
            )

    return ibis_class(nullable)


def schema_from_table(table):
    """Retrieve an ibis schema from a SQLAlchemy ``Table``.

    Parameters
    ----------
    table : sa.Table

    Returns
    -------
    schema : ibis.expr.datatypes.Schema
        An ibis schema corresponding to the types of the columns in `table`.
    """
    # Convert SQLA table to Ibis schema
    types = [
        sqlalchemy_type_to_ibis_type(
            column.type,
            nullable=column.nullable,
            default_timezone='UTC',
        )
        for column in table.columns.values()
    ]
    return sch.Schema(table.columns.keys(), types)


def table_from_schema(name, meta, schema):
    # Convert Ibis schema to SQLA table
    sqla_cols = []

    for cname, itype in zip(schema.names, schema.types):
        ctype = _to_sqla_type(itype)

        col = sa.Column(cname, ctype, nullable=itype.nullable)
        sqla_cols.append(col)

    return sa.Table(name, meta, *sqla_cols)


def _to_sqla_type(itype, type_map=None):
    if type_map is None:
        type_map = _ibis_type_to_sqla
    if isinstance(itype, dt.Decimal):
        return sa.types.NUMERIC(itype.precision, itype.scale)
    elif isinstance(itype, dt.Date):
        return sa.Date()
    elif isinstance(itype, dt.Timestamp):
        # SQLAlchemy DateTimes do not store the timezone, just whether the db
        # supports timezones.
        return sa.TIMESTAMP(bool(itype.timezone))
    elif isinstance(itype, dt.Array):
        ibis_type = itype.value_type
        if not isinstance(ibis_type, (dt.Primitive, dt.String)):
            raise TypeError(
                'Type {} is not a primitive type or string type'.format(
                    ibis_type
                )
            )
        return sa.ARRAY(_to_sqla_type(ibis_type, type_map=type_map))
    else:
        return type_map[type(itype)]


def _variance_reduction(func_name):
    suffix = {
        'sample': 'samp',
        'pop': 'pop'
    }

    def variance_compiler(t, expr):
        arg, how, where = expr.op().args

        if arg.type().equals(dt.boolean):
            arg = arg.cast('int32')

        func = getattr(
            sa.func,
            '{}_{}'.format(func_name, suffix.get(how, 'samp'))
        )

        if where is not None:
            arg = where.ifelse(arg, None)
        return func(t.translate(arg))

    return variance_compiler


def infix_op(infix_sym):
    def formatter(t, expr):
        op = expr.op()
        left, right = op.args

        left_arg = t.translate(left)
        right_arg = t.translate(right)
        return left_arg.op(infix_sym)(right_arg)

    return formatter


def fixed_arity(sa_func, arity):
    if isinstance(sa_func, six.string_types):
        sa_func = getattr(sa.func, sa_func)

    def formatter(t, expr):
        if arity != len(expr.op().args):
            raise com.IbisError('incorrect number of args')

        return _varargs_call(sa_func, t, expr)

    return formatter


def varargs(sa_func):
    def formatter(t, expr):
        op = expr.op()
        trans_args = [t.translate(arg) for arg in op.args]
        return sa_func(*trans_args)
    return formatter


def _varargs_call(sa_func, t, expr):
    op = expr.op()
    trans_args = [t.translate(arg) for arg in op.args]
    return sa_func(*trans_args)


def _table_column(t, expr):
    op = expr.op()
    ctx = t.context
    table = op.table

    sa_table = _get_sqla_table(ctx, table)
    out_expr = getattr(sa_table.c, op.name)

    # If the column does not originate from the table set in the current SELECT
    # context, we should format as a subquery
    if t.permit_subquery and ctx.is_foreign_expr(table):
        return sa.select([out_expr])

    return out_expr


def _get_sqla_table(ctx, table):
    if ctx.has_ref(table):
        ctx_level = ctx
        sa_table = ctx_level.get_table(table)
        while sa_table is None and ctx_level.parent is not ctx_level:
            ctx_level = ctx_level.parent
            sa_table = ctx_level.get_table(table)
    else:
        op = table.op()
        if isinstance(op, AlchemyTable):
            sa_table = op.sqla_table
        else:
            sa_table = ctx.get_compiled_expr(table)

    return sa_table


def _table_array_view(t, expr):
    ctx = t.context
    table = ctx.get_compiled_expr(expr.op().table)
    return table


def _exists_subquery(t, expr):
    op = expr.op()
    ctx = t.context

    filtered = (op.foreign_table.filter(op.predicates)
                .projection([ir.literal(1).name(ir.unnamed)]))

    sub_ctx = ctx.subcontext()
    clause = to_sqlalchemy(
        filtered, context=sub_ctx, exists=True, params=t.params
    )

    if isinstance(op, transforms.NotExistsSubquery):
        clause = sa.not_(clause)

    return clause


def _cast(t, expr):
    op = expr.op()
    arg, target_type = op.args
    sa_arg = t.translate(arg)
    sa_type = t.get_sqla_type(target_type)

    if isinstance(arg, ir.CategoryValue) and target_type == 'int32':
        return sa_arg
    else:
        return sa.cast(sa_arg, sa_type)


def _contains(t, expr):
    op = expr.op()

    left, right = [t.translate(arg) for arg in op.args]
    return left.in_(right)


def _not_contains(t, expr):
    return sa.not_(_contains(t, expr))


def _reduction(sa_func):
    def formatter(t, expr):
        op = expr.op()

        # HACK: support trailing arguments
        arg, where = op.args[:2]

        return _reduction_format(t, sa_func, arg, where)
    return formatter


def _reduction_format(t, sa_func, arg, where):
    if where is not None:
        case = where.ifelse(arg, ibis.NA)
        arg = t.translate(case)
    else:
        arg = t.translate(arg)

    return sa_func(arg)


def _literal(t, expr):
    return sa.literal(expr.op().value)


def _value_list(t, expr):
    return [t.translate(x) for x in expr.op().values]


def _is_null(t, expr):
    arg = t.translate(expr.op().args[0])
    return arg.is_(sa.null())


def _not_null(t, expr):
    arg = t.translate(expr.op().args[0])
    return arg.isnot(sa.null())


def _round(t, expr):
    op = expr.op()
    arg, digits = op.args
    sa_arg = t.translate(arg)

    f = sa.func.round

    if digits is not None:
        sa_digits = t.translate(digits)
        return f(sa_arg, sa_digits)
    else:
        return f(sa_arg)


def _count_distinct(t, expr):
    arg, where = expr.op().args

    if where is not None:
        sa_arg = t.translate(where.ifelse(arg, None))
    else:
        sa_arg = t.translate(arg)

    return sa.func.count(sa_arg.distinct())


def _simple_case(t, expr):
    op = expr.op()

    cases = [op.base == case for case in op.cases]
    return _translate_case(t, cases, op.results, op.default)


def _searched_case(t, expr):
    op = expr.op()
    return _translate_case(t, op.cases, op.results, op.default)


def _translate_case(t, cases, results, default):
    case_args = [t.translate(arg) for arg in cases]
    result_args = [t.translate(arg) for arg in results]

    whens = zip(case_args, result_args)
    default = t.translate(default)

    return sa.case(whens, else_=default)


def _negate(t, expr):
    op = expr.op()
    arg, = map(t.translate, op.args)
    return sa.not_(arg) if isinstance(expr, ir.BooleanValue) else -arg


def unary(sa_func):
    return fixed_arity(sa_func, 1)


def _string_like(t, expr):
    arg, pattern, escape = expr.op().args
    result = t.translate(arg).like(t.translate(pattern), escape=escape)
    return result


_operation_registry = {
    ops.And: fixed_arity(sql.and_, 2),
    ops.Or: fixed_arity(sql.or_, 2),
    ops.Not: unary(sa.not_),

    ops.Abs: unary(sa.func.abs),

    ops.Cast: _cast,

    ops.Coalesce: varargs(sa.func.coalesce),

    ops.NullIf: fixed_arity(sa.func.nullif, 2),

    ops.Contains: _contains,
    ops.NotContains: _not_contains,

    ops.Count: _reduction(sa.func.count),
    ops.Sum: _reduction(sa.func.sum),
    ops.Mean: _reduction(sa.func.avg),
    ops.Min: _reduction(sa.func.min),
    ops.Max: _reduction(sa.func.max),

    ops.CountDistinct: _count_distinct,

    ops.GroupConcat: fixed_arity(sa.func.group_concat, 2),

    ops.Between: fixed_arity(sa.between, 3),

    ops.IsNull: _is_null,
    ops.NotNull: _not_null,

    ops.Negate: _negate,
    ops.Round: _round,

    ops.TypeOf: unary(sa.func.typeof),

    ir.Literal: _literal,
    ir.ValueList: _value_list,
    ir.NullLiteral: lambda *args: sa.null(),

    ops.SimpleCase: _simple_case,
    ops.SearchedCase: _searched_case,

    ops.TableColumn: _table_column,
    ops.TableArrayView: _table_array_view,

    transforms.ExistsSubquery: _exists_subquery,
    transforms.NotExistsSubquery: _exists_subquery,

    ops.StringSQLLike: _string_like,
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
    ops.IdenticalTo: lambda x, y: x.op('IS NOT DISTINCT FROM')(y),

    # Boolean comparisons

    # TODO
}

for _k, _v in _binary_ops.items():
    _operation_registry[_k] = fixed_arity(_v, 2)


class AlchemySelectBuilder(comp.SelectBuilder):

    @property
    def _select_class(self):
        return AlchemySelect

    def _convert_group_by(self, exprs):
        return exprs


class AlchemyContext(comp.QueryContext):

    def __init__(self, *args, **kwargs):
        self._table_objects = {}
        self.dialect = kwargs.pop('dialect', AlchemyDialect)
        super(AlchemyContext, self).__init__(*args, **kwargs)

    def subcontext(self):
        return type(self)(dialect=self.dialect, parent=self)

    def _to_sql(self, expr, ctx, params=None):
        return to_sqlalchemy(expr, context=ctx, params=params)

    def _compile_subquery(self, expr, params=None):
        sub_ctx = self.subcontext()
        return self._to_sql(expr, sub_ctx, params=params)

    def has_table(self, expr, parent_contexts=False):
        key = self._get_table_key(expr)
        return self._key_in(key, '_table_objects',
                            parent_contexts=parent_contexts)

    def set_table(self, expr, obj):
        key = self._get_table_key(expr)
        self._table_objects[key] = obj

    def get_table(self, expr):
        """
        Get the memoized SQLAlchemy expression object
        """
        return self._get_table_item('_table_objects', expr)


class AlchemyQueryBuilder(comp.QueryBuilder):

    select_builder = AlchemySelectBuilder

    def __init__(self, expr, context=None, dialect=None, params=None):
        self.dialect = dialect if dialect is not None else AlchemyDialect
        super(AlchemyQueryBuilder, self).__init__(
            expr, context=context, params=params
        )

    def _make_context(self):
        return AlchemyContext(dialect=self.dialect)

    @property
    def _union_class(self):
        return AlchemyUnion


def to_sqlalchemy(expr, context=None, exists=False, dialect=None, params=None):
    if context is not None:
        dialect = dialect or context.dialect

    ast = build_ast(expr, context=context, dialect=dialect, params=params)
    query = ast.queries[0]

    if exists:
        query.exists = exists

    return query.compile()


def build_ast(expr, context=None, dialect=None, params=None):
    builder = AlchemyQueryBuilder(
        expr, context=context, dialect=dialect, params=params
    )
    return builder.get_result()


class AlchemyDatabaseSchema(Database):

    def __init__(self, name, database):
        """

        Parameters
        ----------
        name : str
        database : AlchemyDatabase
        """
        self.name = name
        self.database = database
        self.client = database.client

    def __repr__(self):
        return "Schema({!r})".format(self.name)

    def drop(self, force=False):
        """
        Drop the schema

        Parameters
        ----------
        force : boolean, default False
          Drop any objects if they exist, and do not fail if the schema does
          not exist.
        """
        raise NotImplementedError(
            "Drop is not implemented yet for sqlalchemy schemas")

    def table(self, name):
        """
        Return a table expression referencing a table in this schema

        Returns
        -------
        table : TableExpr
        """
        qualified_name = self._qualify(name)
        return self.database.table(qualified_name, self.name)

    def list_tables(self, like=None):
        return self.database.list_tables(
            schema=self.name,
            like=self._qualify_like(like))


class AlchemyDatabase(Database):
    """

    Attributes
    ----------
    client : AlchemyClient

    """
    schema_class = AlchemyDatabaseSchema

    def table(self, name, schema=None):
        return self.client.table(name, schema=schema)

    def list_tables(self, like=None, schema=None):
        return self.client.list_tables(
            schema=schema,
            like=self._qualify_like(like),
            database=self.name)

    def schema(self, name):
        return self.schema_class(name, self)


class AlchemyTable(ops.DatabaseTable):

    def __init__(self, table, source, schema=None):
        super(AlchemyTable, self).__init__(
            table.name,
            schema or schema_from_table(table),
            source,
        )
        self.sqla_table = table


class AlchemyExprTranslator(comp.ExprTranslator):

    _registry = _operation_registry
    _rewrites = comp.ExprTranslator._rewrites.copy()
    _type_map = _ibis_type_to_sqla

    def name(self, translated, name, force=True):
        return translated.label(name)

    @property
    def _context_class(self):
        return AlchemyContext

    def get_sqla_type(self, data_type):
        return _to_sqla_type(data_type, type_map=self._type_map)


rewrites = AlchemyExprTranslator.rewrites
compiles = AlchemyExprTranslator.compiles


class AlchemyQuery(Query):

    def _fetch(self, cursor):
        df = pd.DataFrame.from_records(
            cursor.proxy.fetchall(),
            columns=cursor.proxy.keys(),
            coerce_float=True
        )
        dtypes = df.dtypes
        for column in df.columns:
            existing_dtype = dtypes[column]
            try:
                db_type = _to_sqla_type(self.schema[column])
            except KeyError:
                new_dtype = existing_dtype
            else:
                try:
                    new_dtype = self._db_type_to_dtype(db_type, column)
                except TypeError:
                    new_dtype = existing_dtype
                else:
                    # to support datetime64[D]
                    if isinstance(new_dtype, np.dtype):
                        df[column] = pd.Series(
                            df[column].values.astype(new_dtype),
                            name=df[column].name
                        )
                    else:
                        df[column] = df[column].astype(new_dtype)

        return df

    def _db_type_to_dtype(self, db_type, column):
        if isinstance(db_type, sa.Date):
            return np.dtype('datetime64[D]')
        elif isinstance(db_type, sa.DateTime):
            if not db_type.timezone:
                return np.dtype('datetime64[ns]')
            else:
                return compat.DatetimeTZDtype(
                    'ns', self.schema[column].timezone
                )
        else:
            raise TypeError(repr(db_type))

    @property
    def schema(self):
        expr = self.expr
        try:
            return expr.schema()
        except AttributeError:
            return ibis.schema([(expr.get_name(), expr.type())])


class AlchemyAsyncQuery(AsyncQuery):
    pass


class AlchemyDialect(object):

    translator = AlchemyExprTranslator


def invalidates_reflection_cache(f):
    """Invalidate the SQLAlchemy reflection cache if `f` performs an operation
    that mutates database or table metadata such as ``CREATE TABLE``,
    ``DROP TABLE``, etc.

    Parameters
    ----------
    f : callable
        A method on :class:`ibis.sql.alchemy.AlchemyClient`
    """
    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        result = f(self, *args, **kwargs)

        # only invalidate the cache after we've succesfully called the wrapped
        # function
        self._reflection_cache_is_dirty = True
        return result
    return wrapped


class AlchemyClient(SQLClient):

    dialect = AlchemyDialect
    sync_query = AlchemyQuery

    def __init__(self, con):
        super(AlchemyClient, self).__init__()
        self.con = con
        self.meta = sa.MetaData(bind=con)
        self._inspector = sa.inspect(con)
        self._reflection_cache_is_dirty = False
        self._schemas = {}

    @property
    def async_query(self):
        raise NotImplementedError(
            'async_query not implemented in {}'.format(type(self).__name__)
        )

    @property
    def inspector(self):
        if self._reflection_cache_is_dirty:
            self._inspector.info_cache.clear()
        return self._inspector

    @contextlib.contextmanager
    def begin(self):
        with self.con.begin() as bind:
            yield bind

    @invalidates_reflection_cache
    def create_table(self, name, expr=None, schema=None, database=None):
        if database is not None and database != self.engine.url.database:
            raise NotImplementedError(
                'Creating tables from a different database is not yet '
                'implemented'
            )

        if expr is None and schema is None:
            raise ValueError('You must pass either an expression or a schema')

        if expr is not None and schema is not None:
            if not expr.schema().equals(ibis.schema(schema)):
                raise TypeError(
                    'Expression schema is not equal to passed schema. '
                    'Try passing the expression without the schema'
                )
        if schema is None:
            schema = expr.schema()

        self._schemas[self._fully_qualified_name(name, database)] = schema
        t = table_from_schema(name, self.meta, schema)

        with self.begin() as bind:
            t.create(bind=bind)
            if expr is not None:
                bind.execute(
                    t.insert().from_select(list(expr.columns), expr.compile())
                )

    @invalidates_reflection_cache
    def drop_table(self, table_name, database=None, force=False):
        if database is not None and database != self.engine.url.database:
            raise NotImplementedError(
                'Dropping tables from a different database is not yet '
                'implemented'
            )

        t = sa.Table(table_name, self.meta)
        t.drop(checkfirst=force)

        assert not t.exists(), \
            'Something went wrong during DROP of table {!r}'.format(t.name)

        self.meta.remove(t)

        qualified_name = self._fully_qualified_name(table_name, database)

        try:
            del self._schemas[qualified_name]
        except KeyError:  # schemas won't be cached if created with raw_sql
            pass

    def truncate_table(self, table_name, database=None):
        self.meta.tables[table_name].delete().execute()

    def list_tables(self, like=None, database=None, schema=None):
        """
        List tables/views in the current (or indicated) database.

        Parameters
        ----------
        like : string, default None
          Checks for this string contained in name
        database : string, default None
          If not passed, uses the current/default database

        Returns
        -------
        tables : list of strings
        """
        inspector = self.inspector
        names = inspector.get_table_names(schema=schema)
        names.extend(inspector.get_view_names(schema=schema))
        if like is not None:
            names = [x for x in names if like in x]
        return sorted(names)

    def _execute(self, query, results=True):
        with self.begin() as con:
            return AlchemyProxy(con.execute(query))

    @invalidates_reflection_cache
    def raw_sql(self, query, results=False):
        return super(AlchemyClient, self).raw_sql(query, results=results)

    def _build_ast(self, expr, params=None):
        return build_ast(expr, dialect=self.dialect, params=params)

    def _get_sqla_table(self, name, schema=None):
        return sa.Table(name, self.meta, schema=schema, autoload=True)

    def _sqla_table_to_expr(self, table):
        node = AlchemyTable(table, self)
        return self._table_expr_klass(node)


class AlchemySelect(Select):

    def __init__(self, *args, **kwargs):
        self.exists = kwargs.pop('exists', False)
        super(AlchemySelect, self).__init__(*args, **kwargs)

    def compile(self):
        # Can't tell if this is a hack or not. Revisit later
        self.context.set_query(self)

        self._compile_subqueries()

        frag = self._compile_table_set()
        steps = [self._add_select,
                 self._add_groupby,
                 self._add_where,
                 self._add_order_by,
                 self._add_limit]

        for step in steps:
            frag = step(frag)

        return frag

    def _compile_subqueries(self):
        if not self.subqueries:
            return

        for expr in self.subqueries:
            result = self.context.get_compiled_expr(expr)
            alias = self.context.get_ref(expr)
            result = result.cte(alias)
            self.context.set_table(expr, result)

    def _compile_table_set(self):
        if self.table_set is not None:
            helper = _AlchemyTableSet(self, self.table_set)
            return helper.get_result()
        else:
            return None

    def _add_select(self, table_set):
        to_select = []

        has_select_star = False
        for expr in self.select_set:
            if isinstance(expr, ir.ValueExpr):
                arg = self._translate(expr, named=True)
            elif isinstance(expr, ir.TableExpr):
                if expr.equals(self.table_set):
                    cached_table = self.context.get_table(expr)
                    if cached_table is None:
                        # the select * case from materialized join
                        has_select_star = True
                        continue
                    else:
                        arg = table_set
                else:
                    arg = self.context.get_table(expr)
                    if arg is None:
                        raise ValueError(expr)

            to_select.append(arg)

        if has_select_star:
            if table_set is None:
                raise ValueError('table_set cannot be None here')

            clauses = [table_set] + to_select
        else:
            clauses = to_select

        if self.exists:
            result = sa.exists(clauses)
        else:
            result = sa.select(clauses)

        if self.distinct:
            result = result.distinct()

        if not has_select_star:
            if table_set is not None:
                return result.select_from(table_set)
            else:
                return result
        else:
            return result

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

        clauses = []
        for expr in self.order_by:
            key = expr.op()
            sort_expr = key.expr

            # here we have to determine if key.expr is in the select set (as it
            # will be in the case of order_by fused with an aggregation
            if _can_lower_sort_column(self.table_set, sort_expr):
                arg = sort_expr.get_name()
            else:
                arg = self._translate(sort_expr)

            if not key.ascending:
                arg = sa.desc(arg)

            clauses.append(arg)

        return fragment.order_by(*clauses)

    def _among_select_set(self, expr):
        for other in self.select_set:
            if expr.equals(other):
                return True
        return False

    def _add_limit(self, fragment):
        if self.limit is None:
            return fragment

        n, offset = self.limit['n'], self.limit['offset']
        fragment = fragment.limit(n)
        if offset is not None and offset != 0:
            fragment = fragment.offset(offset)

        return fragment

    @property
    def translator(self):
        return self.dialect.translator

    @property
    def dialect(self):
        return self.context.dialect


class _AlchemyTableSet(TableSetFormatter):

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
            elif jtype is ops.LeftSemiJoin:
                result = (sa.select([result])
                          .where(sa.exists(sa.select([1])
                                           .where(onclause))))
            elif jtype is ops.LeftAntiJoin:
                result = (sa.select([result])
                          .where(~sa.exists(sa.select([1])
                                            .where(onclause))))
            else:
                raise NotImplementedError(jtype)

        return result

    def _get_join_type(self, op):
        return type(op)

    def _format_table(self, expr):
        ctx = self.context
        ref_expr = expr
        op = ref_op = expr.op()

        if isinstance(op, ops.SelfReference):
            ref_expr = op.table
            ref_op = ref_expr.op()

        alias = ctx.get_ref(expr)

        if isinstance(ref_op, AlchemyTable):
            result = ref_op.sqla_table
        elif isinstance(ref_op, ops.UnboundTable):
            # use SQLAlchemy's TableClause and ColumnClause for unbound tables
            schema = ref_op.schema
            result = sa.table(
                ref_op.name if ref_op.name is not None else ctx.get_ref(expr),
                *(
                    sa.column(n, _to_sqla_type(t))
                    for n, t in zip(schema.names, schema.types)
                )
            )
        else:
            # A subquery
            if ctx.is_extracted(ref_expr):
                # Was put elsewhere, e.g. WITH block, we just need to grab
                # its alias
                alias = ctx.get_ref(expr)

                # hack
                if isinstance(op, ops.SelfReference):
                    table = ctx.get_table(ref_expr)
                    self_ref = table.alias(alias)
                    ctx.set_table(expr, self_ref)
                    return self_ref
                else:
                    return ctx.get_table(expr)

            result = ctx.get_compiled_expr(expr)
            alias = ctx.get_ref(expr)

        result = result.alias(alias)
        ctx.set_table(expr, result)
        return result


def _can_lower_sort_column(table_set, expr):
    # TODO(wesm): This code is pending removal through cleaner internal
    # semantics

    # we can currently sort by just-appeared aggregate metrics, but the way
    # these are references in the expression DSL is as a SortBy (blocking
    # table operation) on an aggregation. There's a hack in _collect_SortBy
    # in the generic SQL compiler that "fuses" the sort with the
    # aggregation so they appear in same query. It's generally for
    # cosmetics and doesn't really affect query semantics.
    bases = ops.find_all_base_tables(expr)
    if len(bases) > 1:
        return False

    base = list(bases.values())[0]
    base_op = base.op()

    if isinstance(base_op, ops.Aggregation):
        return base_op.table.equals(table_set)
    elif isinstance(base_op, ops.Selection):
        return base.equals(table_set)
    else:
        return False


def _and_all(clauses):
    result = clauses[0]
    for clause in clauses[1:]:
        result = sql.and_(result, clause)
    return result


class AlchemyUnion(Union):

    def compile(self):
        context = self.context
        sa_func = sa.union if self.distinct else sa.union_all

        left_set = context.get_compiled_expr(self.left)
        left_cte = left_set.cte()
        left_select = left_cte.select()

        right_set = context.get_compiled_expr(self.right)
        right_cte = right_set.cte()
        right_select = right_cte.select()

        return sa_func(left_select, right_select)


class AlchemyProxy(object):
    """
    Wraps a SQLAlchemy ResultProxy and ensures that .close() is called on
    garbage collection
    """
    def __init__(self, proxy):
        self.proxy = proxy

    def __del__(self):
        self._close_cursor()

    def _close_cursor(self):
        self.proxy.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self._close_cursor()

    def fetchall(self):
        return self.proxy.fetchall()


@rewrites(ops.NullIfZero)
def _nullifzero(expr):
    arg = expr.op().args[0]
    return (arg == 0).ifelse(ibis.NA, arg)


@compiles(ops.Divide)
def _true_divide(t, expr):
    op = expr.op()
    left, right = op.args

    if util.all_of(op.args, ir.IntegerValue):
        new_expr = left.div(right.cast('double'))
        return t.translate(new_expr)

    return fixed_arity(lambda x, y: x / y, 2)(t, expr)


@compiles(ops.FloorDivide)
def _floor_divide(t, expr):
    op = expr.op()
    left, right = op.args

    if util.any_of(op.args, ir.FloatingValue):
        new_expr = expr.floor()
        return t.translate(new_expr)

    return fixed_arity(lambda x, y: x / y, 2)(t, expr)


@compiles(ops.SortKey)
def _sort_key(t, expr):
    # We need to define this for window functions that have an order by
    by, ascending = expr.op().args
    sort_direction = sa.asc if ascending else sa.desc
    return sort_direction(t.translate(by))


_valid_frame_types = numbers.Integral, str, type(None)


class Over(_Over):
    def __init__(
        self,
        element,
        order_by=None,
        partition_by=None,
        preceding=None,
        following=None,
    ):
        super(Over, self).__init__(
            element, order_by=order_by, partition_by=partition_by
        )
        if not isinstance(preceding, _valid_frame_types):
            raise TypeError(
                'preceding must be a string, integer or None, got %r' % (
                    type(preceding).__name__
                )
            )
        if not isinstance(following, _valid_frame_types):
            raise TypeError(
                'following must be a string, integer or None, got %r' % (
                    type(following).__name__
                )
            )
        self.preceding = preceding if preceding is not None else 'UNBOUNDED'
        self.following = following if following is not None else 'UNBOUNDED'


@sa_compiles(Over)
def compile_over_with_frame(element, compiler, **kw):
    clauses = ' '.join(
        '%s BY %s' % (word, compiler.process(clause, **kw))
        for word, clause in (
            ('PARTITION', element.partition_by),
            ('ORDER', element.order_by),
        )
        if clause is not None and len(clause)
    )
    return '%s OVER (%s%sROWS BETWEEN %s PRECEDING AND %s FOLLOWING)' % (
        compiler.process(getattr(element, 'element', element.func), **kw),
        clauses,
        ' ' if clauses else '',  # only add a space if we order by or group by
        str(element.preceding).upper(),
        str(element.following).upper(),
    )
