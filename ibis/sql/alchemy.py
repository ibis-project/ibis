import contextlib
import numbers
import operator

import six

import sqlalchemy as sa
import sqlalchemy.sql as sql

from sqlalchemy.sql.elements import Over as _Over
from sqlalchemy.ext.compiler import compiles as sa_compiles
from sqlalchemy.engine.interfaces import Dialect as SQLAlchemyDialect

from sqlalchemy.dialects.sqlite.base import SQLiteDialect
from sqlalchemy.dialects.postgresql.base import PGDialect as PostgreSQLDialect
from sqlalchemy.dialects.mysql.base import MySQLDialect

import pandas as pd

from ibis.compat import parse_version, functools
from ibis.client import SQLClient, Query, Database
from ibis.sql.compiler import Select, Union, TableSetFormatter, Dialect

import ibis
import ibis.util as util
import ibis.common as com
import ibis.expr.types as ir
import ibis.expr.schema as sch
import ibis.expr.window as W
import ibis.expr.analysis as L
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.sql.compiler as comp
import ibis.sql.transforms as transforms


# TODO(cleanup)
_ibis_type_to_sqla = {
    dt.Null: sa.types.NullType,
    dt.Date: sa.Date,
    dt.Time: sa.Time,
    dt.Boolean: sa.Boolean,
    dt.Binary: sa.Binary,
    dt.String: sa.Text,
    dt.Decimal: sa.NUMERIC,

    # Mantissa-based
    dt.Float: sa.Float(precision=24),
    dt.Double: sa.Float(precision=53),

    dt.Int8: sa.SmallInteger,
    dt.Int16: sa.SmallInteger,
    dt.Int32: sa.Integer,
    dt.Int64: sa.BigInteger
}


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


@dt.dtype.register(SQLAlchemyDialect, sa.types.NullType)
def sa_null(_, satype, nullable=True):
    return dt.null


@dt.dtype.register(SQLAlchemyDialect, sa.types.Boolean)
def sa_boolean(_, satype, nullable=True):
    return dt.Boolean(nullable=nullable)


@dt.dtype.register(SQLAlchemyDialect, sa.types.Numeric)
def sa_numeric(_, satype, nullable=True):
    return dt.Decimal(satype.precision, satype.scale, nullable=nullable)


@dt.dtype.register(SQLAlchemyDialect, sa.types.SmallInteger)
def sa_smallint(_, satype, nullable=True):
    return dt.Int16(nullable=nullable)


@dt.dtype.register(SQLAlchemyDialect, sa.types.Integer)
def sa_integer(_, satype, nullable=True):
    return dt.Int32(nullable=nullable)


@dt.dtype.register(SQLAlchemyDialect, sa.dialects.mysql.TINYINT)
def sa_mysql_tinyint(_, satype, nullable=True):
    return dt.Int8(nullable=nullable)


@dt.dtype.register(SQLAlchemyDialect, sa.types.BigInteger)
def sa_bigint(_, satype, nullable=True):
    return dt.Int64(nullable=nullable)


@dt.dtype.register(SQLAlchemyDialect, sa.types.Float)
def sa_float(_, satype, nullable=True):
    return dt.Float(nullable=nullable)


@dt.dtype.register(SQLiteDialect, sa.types.Float)
@dt.dtype.register(PostgreSQLDialect, sa.dialects.postgresql.DOUBLE_PRECISION)
def sa_double(_, satype, nullable=True):
    return dt.Double(nullable=nullable)


@dt.dtype.register(MySQLDialect, sa.dialects.mysql.DOUBLE)
def sa_mysql_double(_, satype, nullable=True):
    # TODO: handle asdecimal=True
    return dt.Double(nullable=nullable)


@dt.dtype.register(SQLAlchemyDialect, sa.types.String)
def sa_string(_, satype, nullable=True):
    return dt.String(nullable=nullable)


@dt.dtype.register(SQLAlchemyDialect, sa.types.Binary)
def sa_binary(_, satype, nullable=True):
    return dt.Binary(nullable=nullable)


@dt.dtype.register(SQLAlchemyDialect, sa.Time)
def sa_time(_, satype, nullable=True):
    return dt.Time(nullable=nullable)


@dt.dtype.register(SQLAlchemyDialect, sa.Date)
def sa_date(_, satype, nullable=True):
    return dt.Date(nullable=nullable)


@dt.dtype.register(SQLAlchemyDialect, sa.DateTime)
def sa_datetime(_, satype, nullable=True, default_timezone='UTC'):
    timezone = default_timezone if satype.timezone else None
    return dt.Timestamp(timezone=timezone, nullable=nullable)


@dt.dtype.register(SQLAlchemyDialect, sa.ARRAY)
def sa_array(dialect, satype, nullable=True):
    dimensions = satype.dimensions
    if dimensions is not None and dimensions != 1:
        raise NotImplementedError('Nested array types not yet supported')

    value_dtype = dt.dtype(dialect, satype.item_type)
    return dt.Array(value_dtype, nullable=nullable)


@sch.infer.register(sa.Table)
def schema_from_table(table, schema=None):
    """Retrieve an ibis schema from a SQLAlchemy ``Table``.

    Parameters
    ----------
    table : sa.Table

    Returns
    -------
    schema : ibis.expr.datatypes.Schema
        An ibis schema corresponding to the types of the columns in `table`.
    """
    schema = schema if schema is not None else {}
    pairs = []
    for name, column in table.columns.items():
        if name in schema:
            dtype = dt.dtype(schema[name])
        else:
            dtype = dt.dtype(
                getattr(table.bind, 'dialect', SQLAlchemyDialect()),
                column.type,
                nullable=column.nullable
            )
        pairs.append((name, dtype))
    return sch.schema(pairs)


def table_from_schema(name, meta, schema):
    # Convert Ibis schema to SQLA table
    columns = []

    for colname, dtype in zip(schema.names, schema.types):
        satype = _to_sqla_type(dtype)
        column = sa.Column(colname, satype, nullable=dtype.nullable)
        columns.append(column)

    return sa.Table(name, meta, *columns)


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
        trans_args = [t.translate(arg) for arg in op.arg]
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
    clause = to_sqlalchemy(filtered, sub_ctx, exists=True)

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
    dtype = expr.type()
    value = expr.op().value

    if isinstance(dtype, dt.Set):
        return list(map(sa.literal, value))

    return sa.literal(value)


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


def _floor_divide(t, expr):
    left, right = map(t.translate, expr.op().args)
    return sa.func.floor(left / right)


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


_cumulative_to_reduction = {
    ops.CumulativeSum: ops.Sum,
    ops.CumulativeMin: ops.Min,
    ops.CumulativeMax: ops.Max,
    ops.CumulativeMean: ops.Mean,
    ops.CumulativeAny: ops.Any,
    ops.CumulativeAll: ops.All,
}


def _cumulative_to_window(translator, expr, window):
    win = W.cumulative_window()
    win = win.group_by(window._group_by).order_by(window._order_by)

    op = expr.op()

    klass = _cumulative_to_reduction[type(op)]
    new_op = klass(*op.args)
    new_expr = expr._factory(new_op, name=expr._name)

    if type(new_op) in translator._rewrites:
        new_expr = translator._rewrites[type(new_op)](new_expr)

    return L.windowize_function(new_expr, win)


def _window(t, expr):
    op = expr.op()

    arg, window = op.args
    reduction = t.translate(arg)

    window_op = arg.op()

    _require_order_by = (
        ops.Lag,
        ops.Lead,
        ops.DenseRank,
        ops.MinRank,
        ops.NTile,
        ops.PercentRank,
        ops.FirstValue,
        ops.LastValue,
    )

    if isinstance(window_op, ops.CumulativeOp):
        arg = _cumulative_to_window(t, arg, window)
        return t.translate(arg)

    # Some analytic functions need to have the expression of interest in
    # the ORDER BY part of the window clause
    if isinstance(window_op, _require_order_by) and not window._order_by:
        order_by = t.translate(window_op.args[0])
    else:
        order_by = list(map(t.translate, window._order_by))

    partition_by = list(map(t.translate, window._group_by))

    result = Over(
        reduction,
        partition_by=partition_by,
        order_by=order_by,
        preceding=window.preceding,
        following=window.following,
    )

    if isinstance(
        window_op, (ops.RowNumber, ops.DenseRank, ops.MinRank, ops.NTile)
    ):
        return result - 1
    else:
        return result


def _lag(t, expr):
    arg, offset, default = expr.op().args
    if default is not None:
        raise NotImplementedError()

    sa_arg = t.translate(arg)
    sa_offset = t.translate(offset) if offset is not None else 1
    return sa.func.lag(sa_arg, sa_offset)


def _lead(t, expr):
    arg, offset, default = expr.op().args
    if default is not None:
        raise NotImplementedError()
    sa_arg = t.translate(arg)
    sa_offset = t.translate(offset) if offset is not None else 1
    return sa.func.lead(sa_arg, sa_offset)


def _ntile(t, expr):
    op = expr.op()
    args = op.args
    arg, buckets = map(t.translate, args)
    return sa.func.ntile(buckets)


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

    ops.Literal: _literal,
    ops.ValueList: _value_list,
    ops.NullLiteral: lambda *args: sa.null(),

    ops.SimpleCase: _simple_case,
    ops.SearchedCase: _searched_case,

    ops.TableColumn: _table_column,
    ops.TableArrayView: _table_array_view,

    transforms.ExistsSubquery: _exists_subquery,
    transforms.NotExistsSubquery: _exists_subquery,

    # miscellaneous varargs
    ops.Least: varargs(sa.func.least),
    ops.Greatest: varargs(sa.func.greatest),

    # string
    ops.LPad: fixed_arity(sa.func.lpad, 3),
    ops.RPad: fixed_arity(sa.func.rpad, 3),
    ops.Strip: unary(sa.func.trim),
    ops.LStrip: unary(sa.func.ltrim),
    ops.RStrip: unary(sa.func.rtrim),
    ops.Repeat: fixed_arity(sa.func.repeat, 2),
    ops.Reverse: unary(sa.func.reverse),
    ops.StrRight: fixed_arity(sa.func.right, 2),
    ops.Lowercase: unary(sa.func.lower),
    ops.Uppercase: unary(sa.func.upper),
    ops.StringAscii: unary(sa.func.ascii),
    ops.StringLength: unary(sa.func.length),
    ops.StringReplace: fixed_arity(sa.func.replace, 3),
    ops.StringSQLLike: _string_like,

    # math
    ops.Ln: unary(sa.func.ln),
    ops.Exp: unary(sa.func.exp),
    ops.Sign: unary(sa.func.sign),
    ops.Sqrt: unary(sa.func.sqrt),
    ops.Ceil: unary(sa.func.ceil),
    ops.Floor: unary(sa.func.floor),
    ops.Power: fixed_arity(sa.func.pow, 2),
    ops.FloorDivide: _floor_divide,
}


# TODO: unit tests for each of these
_binary_ops = {
    # Binary arithmetic
    ops.Add: operator.add,
    ops.Subtract: operator.sub,
    ops.Multiply: operator.mul,
    ops.Divide: operator.truediv,
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


_window_functions = {
    ops.Lag: _lag,
    ops.Lead: _lead,
    ops.NTile: _ntile,
    ops.FirstValue: unary(sa.func.first_value),
    ops.LastValue: unary(sa.func.last_value),
    ops.RowNumber: fixed_arity(lambda: sa.func.row_number(), 0),
    ops.DenseRank: unary(lambda arg: sa.func.dense_rank()),
    ops.MinRank: unary(lambda arg: sa.func.rank()),
    ops.PercentRank: unary(lambda arg: sa.func.percent_rank()),
    ops.WindowOp: _window,
    ops.CumulativeOp: _window,
    ops.CumulativeMax: unary(sa.func.max),
    ops.CumulativeMin: unary(sa.func.min),
    ops.CumulativeSum: unary(sa.func.sum),
    ops.CumulativeMean: unary(sa.func.avg),
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
        super(AlchemyContext, self).__init__(*args, **kwargs)
        self._table_objects = {}

    def collapse(self, queries):
        if isinstance(queries, six.string_types):
            return queries

        if len(queries) > 1:
            raise NotImplementedError(
                'Only a single query is supported for SQLAlchemy backends'
            )
        return queries[0]

    def subcontext(self):
        return type(self)(
            dialect=self.dialect, parent=self, params=self.params
        )

    def _to_sql(self, expr, ctx):
        return to_sqlalchemy(expr, ctx)

    def _compile_subquery(self, expr):
        sub_ctx = self.subcontext()
        return self._to_sql(expr, sub_ctx)

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


class AlchemyUnion(Union):

    def compile(self):
        def reduce_union(left, right, distincts=iter(self.distincts)):
            distinct = next(distincts)
            sa_func = sa.union if distinct else sa.union_all
            return sa_func(left, right)

        context = self.context
        selects = []

        for table in self.tables:
            table_set = context.get_compiled_expr(table)
            selects.append(table_set.cte().select())

        return functools.reduce(reduce_union, selects)


class AlchemyQueryBuilder(comp.QueryBuilder):

    select_builder = AlchemySelectBuilder
    union_class = AlchemyUnion


def to_sqlalchemy(expr, context, exists=False):
    ast = build_ast(expr, context)
    query = ast.queries[0]

    if exists:
        query.exists = exists

    return query.compile()


def build_ast(expr, context):
    builder = AlchemyQueryBuilder(expr, context)
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
        schema = sch.infer(table, schema=schema)
        super(AlchemyTable, self).__init__(table.name, schema, source)
        self.sqla_table = table


class AlchemyExprTranslator(comp.ExprTranslator):

    _registry = _operation_registry
    _rewrites = comp.ExprTranslator._rewrites.copy()
    _type_map = _ibis_type_to_sqla

    context_class = AlchemyContext

    def name(self, translated, name, force=True):
        return translated.label(name)

    def get_sqla_type(self, data_type):
        return _to_sqla_type(data_type, type_map=self._type_map)


rewrites = AlchemyExprTranslator.rewrites
compiles = AlchemyExprTranslator.compiles


class AlchemyQuery(Query):

    def _fetch(self, cursor):
        df = pd.DataFrame.from_records(cursor.proxy.fetchall(),
                                       columns=cursor.proxy.keys(),
                                       coerce_float=True)
        return self.schema().apply_to(df)


class AlchemyDialect(Dialect):

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
    query_class = AlchemyQuery

    def __init__(self, con):
        super(AlchemyClient, self).__init__()
        self.con = con
        self.meta = sa.MetaData(bind=con)
        self._inspector = sa.inspect(con)
        self._reflection_cache_is_dirty = False
        self._schemas = {}

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

    def _build_ast(self, expr, context):
        return build_ast(expr, context)

    def _get_sqla_table(self, name, schema=None):
        return sa.Table(name, self.meta, schema=schema, autoload=True)

    def _sqla_table_to_expr(self, table):
        node = self.table_class(table, self)
        return self.table_expr_class(node)

    @property
    def version(self):
        vstring = '.'.join(map(str, self.con.dialect.server_version_info))
        return parse_version(vstring)


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
