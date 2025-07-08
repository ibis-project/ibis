from __future__ import annotations

import abc
import calendar
import itertools
import math
import operator
import string
from functools import partial, reduce
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal

import sqlglot as sg
import sqlglot.expressions as sge
from public import public

import ibis.common.exceptions as com
import ibis.common.patterns as pats
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.rewrites import (
    FirstValue,
    LastValue,
    add_one_to_nth_value_input,
    add_order_by_to_empty_ranking_window_functions,
    empty_in_values_right_side,
    lower_bucket,
    lower_capitalize,
    lower_sample,
    one_to_zero_index,
    sqlize,
)
from ibis.config import options
from ibis.expr.operations.udf import InputType
from ibis.expr.rewrites import lower_stringslice
from ibis.util import get_subclasses

try:
    from sqlglot.expressions import Alter
except ImportError:
    from sqlglot.expressions import AlterTable
else:

    def AlterTable(*args, kind="TABLE", **kwargs):
        return Alter(*args, kind=kind, **kwargs)


try:
    from sqlglot.expressions import AlterRename as RenameTable
except ImportError:
    from sqlglot.expressions import RenameTable  # noqa: F401


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import ibis.expr.schema as sch
    import ibis.expr.types as ir
    from ibis.backends.sql.datatypes import SqlglotType


ALL_OPERATIONS = frozenset(get_subclasses(ops.Node))


class AggGen:
    """A descriptor for compiling aggregate functions.

    Common cases can be handled by setting configuration flags,
    special cases should override the `aggregate` method directly.

    Parameters
    ----------
    supports_filter
        Whether the backend supports a FILTER clause in the aggregate.
        Defaults to False.
    supports_order_by
        Whether the backend supports an ORDER BY clause in (relevant)
        aggregates. Defaults to False.
    """

    class _Accessor:
        """An internal type to handle getattr/getitem access."""

        __slots__ = ("compiler", "handler")

        def __init__(self, handler: Callable, compiler: SQLGlotCompiler):
            self.handler = handler
            self.compiler = compiler

        def __getattr__(self, name: str) -> Callable:
            return partial(self.handler, self.compiler, name)

        __getitem__ = __getattr__

    __slots__ = ("supports_filter", "supports_order_by")

    def __init__(
        self, *, supports_filter: bool = False, supports_order_by: bool = False
    ):
        self.supports_filter = supports_filter
        self.supports_order_by = supports_order_by

    def __get__(self, instance, owner=None):
        if instance is None:
            return self

        return AggGen._Accessor(self.aggregate, instance)

    def aggregate(
        self,
        compiler: SQLGlotCompiler,
        name: str,
        *args: Any,
        where: Any = None,
        order_by: tuple = (),
    ):
        """Compile the specified aggregate.

        Parameters
        ----------
        compiler
            The backend's compiler.
        name
            The aggregate name (e.g. `"sum"`).
        args
            Any arguments to pass to the aggregate.
        where
            An optional column filter to apply before performing the aggregate.
        order_by
            Optional ordering keys to use to order the rows before performing
            the aggregate.
        """
        func = compiler.f[name]

        if order_by and not self.supports_order_by:
            raise com.UnsupportedOperationError(
                "ordering of order-sensitive aggregations via `order_by` is "
                f"not supported for the {compiler.dialect} backend"
            )

        if where is not None and not self.supports_filter:
            args = tuple(compiler.if_(where, arg, NULL) for arg in args)

        if order_by and self.supports_order_by:
            *rest, last = args
            out = func(*rest, sge.Order(this=last, expressions=order_by))
        else:
            out = func(*args)

        if where is not None and self.supports_filter:
            out = sge.Filter(this=out, expression=sge.Where(this=where))

        return out


class VarGen:
    __slots__ = ()

    def __getattr__(self, name: str) -> sge.Var:
        return sge.Var(this=name)

    def __getitem__(self, key: str) -> sge.Var:
        return sge.Var(this=key)


class AnonymousFuncGen:
    __slots__ = ()

    def __getattr__(self, name: str) -> Callable[..., sge.Anonymous]:
        return lambda *args: sge.Anonymous(
            this=name, expressions=list(map(sge.convert, args))
        )

    def __getitem__(self, key: str) -> Callable[..., sge.Anonymous]:
        return getattr(self, key)


class FuncGen:
    __slots__ = ("anon", "copy", "dialect", "namespace")

    def __init__(
        self, *, dialect: sg.Dialect, namespace: str | None = None, copy: bool = False
    ) -> None:
        self.dialect = dialect
        self.namespace = namespace
        self.anon = AnonymousFuncGen()
        self.copy = copy

    def __getattr__(self, name: str) -> Callable[..., sge.Func]:
        name = ".".join(filter(None, (self.namespace, name)))
        return lambda *args, **kwargs: sg.func(
            name,
            *map(sge.convert, args),
            **kwargs,
            copy=self.copy,
            dialect=self.dialect,
        )

    def __getitem__(self, key: str) -> Callable[..., sge.Func]:
        return getattr(self, key)

    def array(self, *args: Any) -> sge.Array:
        if not args:
            return sge.Array(expressions=[])

        first, *rest = args

        if isinstance(first, sge.Select):
            assert not rest, (
                "only one argument allowed when `first` is a select statement"
            )

        return sge.Array(expressions=list(map(sge.convert, (first, *rest))))

    def tuple(self, *args: Any) -> sge.Anonymous:
        return self.anon.tuple(*args)

    def exists(self, query: sge.Expression) -> sge.Exists:
        return sge.Exists(this=query)

    def concat(self, *args: Any) -> sge.Concat:
        return sge.Concat(expressions=list(map(sge.convert, args)))

    def map(self, keys: Iterable, values: Iterable) -> sge.Map:
        return sge.Map(keys=keys, values=values)


class ColGen:
    __slots__ = ("table",)

    def __init__(self, table: str | None = None) -> None:
        self.table = table

    def __getattr__(self, name: str) -> sge.Column:
        return sg.column(name, table=self.table, copy=False)

    def __getitem__(self, key: str) -> sge.Column:
        return sg.column(key, table=self.table, copy=False)


C = ColGen()
NULL = sge.Null()
FALSE = sge.false()
TRUE = sge.true()
STAR = sge.Star()


@public
class SQLGlotCompiler(abc.ABC):
    __slots__ = "f", "v"

    agg = AggGen()
    """A generator for handling aggregate functions"""

    rewrites: tuple[type[pats.Replace], ...] = (
        empty_in_values_right_side,
        add_order_by_to_empty_ranking_window_functions,
        one_to_zero_index,
        add_one_to_nth_value_input,
    )
    """A sequence of rewrites to apply to the expression tree before SQL-specific transforms."""

    post_rewrites: tuple[type[pats.Replace], ...] = ()
    """A sequence of rewrites to apply to the expression tree after SQL-specific transforms."""

    no_limit_value: sge.Null | None = None
    """The value to use to indicate no limit."""

    quoted: bool = True
    """Whether to always quote identifiers."""

    copy_func_args: bool = False
    """Whether to copy function arguments when generating SQL."""

    supports_qualify: bool = False
    """Whether the backend supports the QUALIFY clause."""

    NAN: ClassVar[sge.Expression] = sge.Cast(
        this=sge.convert("NaN"), to=sge.DataType(this=sge.DataType.Type.DOUBLE)
    )
    """Backend's NaN literal."""

    POS_INF: ClassVar[sge.Expression] = sge.Cast(
        this=sge.convert("Inf"), to=sge.DataType(this=sge.DataType.Type.DOUBLE)
    )
    """Backend's positive infinity literal."""

    NEG_INF: ClassVar[sge.Expression] = sge.Cast(
        this=sge.convert("-Inf"), to=sge.DataType(this=sge.DataType.Type.DOUBLE)
    )
    """Backend's negative infinity literal."""

    EXTRA_SUPPORTED_OPS: tuple[type[ops.Node], ...] = (
        ops.Project,
        ops.Filter,
        ops.Sort,
        ops.WindowFunction,
    )
    """A tuple of ops classes that are supported, but don't have explicit
    `visit_*` methods (usually due to being handled by rewrite rules). Used by
    `has_operation`"""

    UNSUPPORTED_OPS: tuple[type[ops.Node], ...] = ()
    """Tuple of operations the backend doesn't support."""

    LOWERED_OPS: dict[type[ops.Node], pats.Replace | None] = {
        ops.Bucket: lower_bucket,
        ops.Capitalize: lower_capitalize,
        ops.Sample: lower_sample(supported_methods=()),
        ops.StringSlice: lower_stringslice,
    }
    """A mapping from an operation class to either a rewrite rule for rewriting that
    operation to one composed of lower-level operations ("lowering"), or `None` to
    remove an existing rewrite rule for that operation added in a base class"""

    SIMPLE_OPS = {
        ops.Abs: "abs",
        ops.Acos: "acos",
        ops.All: "bool_and",
        ops.Any: "bool_or",
        ops.ApproxCountDistinct: "approx_distinct",
        ops.ArrayContains: "array_contains",
        ops.ArrayFlatten: "flatten",
        ops.ArrayLength: "array_size",
        ops.ArraySort: "array_sort",
        ops.ArrayStringJoin: "array_to_string",
        ops.ArgMax: "max_by",
        ops.ArgMin: "min_by",
        ops.Asin: "asin",
        ops.Atan2: "atan2",
        ops.Atan: "atan",
        ops.Cos: "cos",
        ops.Cot: "cot",
        ops.Count: "count",
        ops.CumeDist: "cume_dist",
        ops.Date: "date",
        ops.DateFromYMD: "datefromparts",
        ops.Degrees: "degrees",
        ops.DenseRank: "dense_rank",
        ops.Exp: "exp",
        FirstValue: "first_value",
        ops.GroupConcat: "group_concat",
        ops.IfElse: "if",
        ops.IsInf: "isinf",
        ops.IsNan: "isnan",
        ops.JSONGetItem: "json_extract",
        LastValue: "last_value",
        ops.Levenshtein: "levenshtein",
        ops.Ln: "ln",
        ops.Log10: "log",
        ops.Log2: "log2",
        ops.Lowercase: "lower",
        ops.Map: "map",
        ops.Median: "median",
        ops.MinRank: "rank",
        ops.NTile: "ntile",
        ops.NthValue: "nth_value",
        ops.NullIf: "nullif",
        ops.PercentRank: "percent_rank",
        ops.Pi: "pi",
        ops.Power: "pow",
        ops.Radians: "radians",
        ops.RegexSearch: "regexp_like",
        ops.RegexSplit: "regexp_split",
        ops.RegexExtract: "regexp_extract",
        ops.Repeat: "repeat",
        ops.Reverse: "reverse",
        ops.RowNumber: "row_number",
        ops.Sign: "sign",
        ops.Sin: "sin",
        ops.Sqrt: "sqrt",
        ops.StartsWith: "starts_with",
        ops.StrRight: "right",
        ops.StringAscii: "ascii",
        ops.StringContains: "contains",
        ops.StringLength: "length",
        ops.StringReplace: "replace",
        ops.StringSplit: "split",
        ops.StringToDate: "str_to_date",
        ops.StringToTimestamp: "str_to_time",
        ops.Tan: "tan",
        ops.Translate: "translate",
        ops.Unnest: "explode",
        ops.Uppercase: "upper",
        ops.RandomUUID: "uuid",
        ops.RandomScalar: "rand",
    }

    BINARY_INFIX_OPS = {
        # Numeric
        ops.Add: sge.Add,
        ops.Subtract: sge.Sub,
        ops.Multiply: sge.Mul,
        ops.Divide: sge.Div,
        ops.Modulus: sge.Mod,
        ops.Power: sge.Pow,
        # Comparisons
        ops.GreaterEqual: sge.GTE,
        ops.Greater: sge.GT,
        ops.LessEqual: sge.LTE,
        ops.Less: sge.LT,
        ops.Equals: sge.EQ,
        ops.NotEquals: sge.NEQ,
        # Logical
        ops.And: sge.And,
        ops.Or: sge.Or,
        ops.Xor: sge.Xor,
        # Bitwise
        ops.BitwiseLeftShift: sge.BitwiseLeftShift,
        ops.BitwiseRightShift: sge.BitwiseRightShift,
        ops.BitwiseAnd: sge.BitwiseAnd,
        ops.BitwiseOr: sge.BitwiseOr,
        ops.BitwiseXor: sge.BitwiseXor,
        # Date
        ops.DateAdd: sge.Add,
        ops.DateSub: sge.Sub,
        ops.DateDiff: sge.Sub,
        # Time
        ops.TimeAdd: sge.Add,
        ops.TimeSub: sge.Sub,
        ops.TimeDiff: sge.Sub,
        # Timestamp
        ops.TimestampAdd: sge.Add,
        ops.TimestampSub: sge.Sub,
        ops.TimestampDiff: sge.Sub,
        # Interval
        ops.IntervalAdd: sge.Add,
        ops.IntervalMultiply: sge.Mul,
        ops.IntervalSubtract: sge.Sub,
    }

    # A set of SQLGlot classes that may need to be parenthesized
    SQLGLOT_NEEDS_PARENS = set(BINARY_INFIX_OPS.values()).union((sge.Is,))

    # A set of SQLGlot classes that are associative operations
    SQLGLOT_ASSOCIATIVE_OPS = {
        sge.Add,
        sge.Mul,
        sge.And,
        sge.Or,
        sge.Xor,
        sge.BitwiseAnd,
        sge.BitwiseOr,
        sge.BitwiseXor,
    }

    # Constructed dynamically in `__init_subclass__` from their respective
    # UPPERCASE values to handle inheritance, do not modify directly here.
    extra_supported_ops: ClassVar[frozenset[type[ops.Node]]] = frozenset()
    lowered_ops: ClassVar[dict[type[ops.Node], pats.Replace]] = {}

    def __init__(self) -> None:
        self.f = FuncGen(
            dialect=self.__class__.dialect, copy=self.__class__.copy_func_args
        )
        self.v = VarGen()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def methodname(op: type) -> str:
            assert isinstance(type(op), type), type(op)
            return f"visit_{op.__name__}"

        def make_impl(op, target_name):
            assert isinstance(type(op), type), type(op)

            if issubclass(op, ops.Reduction):

                def impl(
                    self, _, *, _name: str = target_name, where, order_by=(), **kw
                ):
                    return self.agg[_name](*kw.values(), where=where, order_by=order_by)

            else:

                def impl(self, _, *, _name: str = target_name, **kw):
                    return self.f[_name](*kw.values())

            return impl

        for op, target_name in cls.SIMPLE_OPS.items():
            setattr(cls, methodname(op), make_impl(op, target_name))

        # Define binary op methods, only if BINARY_INFIX_OPS is set on the
        # compiler class.
        if binops := cls.__dict__.get("BINARY_INFIX_OPS", {}):

            def make_binop(sge_cls):
                def impl(self, op, *, left, right):
                    return self.binop(sge_cls, left, right)

                return impl

            for op, sge_cls in binops.items():
                setattr(cls, methodname(op), make_binop(sge_cls))

        # unconditionally raise an exception for unsupported operations
        #
        # these *must* be defined after SIMPLE_OPS to handle compilers that
        # subclass other compilers
        for op in cls.UNSUPPORTED_OPS:
            # change to visit_Unsupported in a follow up
            # TODO: handle geoespatial ops as a separate case?
            setattr(cls, methodname(op), cls.visit_Undefined)

        # raise on any remaining unsupported operations
        for op in ALL_OPERATIONS:
            name = methodname(op)
            if not hasattr(cls, name):
                setattr(cls, name, cls.visit_Undefined)

        # Amend `lowered_ops` and `extra_supported_ops` using their
        # respective UPPERCASE classvar values.
        extra_supported_ops = set(cls.extra_supported_ops)
        lowered_ops = dict(cls.lowered_ops)
        extra_supported_ops.update(cls.EXTRA_SUPPORTED_OPS)
        for op_cls, rewrite in cls.LOWERED_OPS.items():
            if rewrite is not None:
                lowered_ops[op_cls] = rewrite
                extra_supported_ops.add(op_cls)
            else:
                lowered_ops.pop(op_cls, None)
                extra_supported_ops.discard(op_cls)
        cls.lowered_ops = lowered_ops
        cls.extra_supported_ops = frozenset(extra_supported_ops)

    @property
    @abc.abstractmethod
    def dialect(self) -> type[sg.Dialect]:
        """Backend dialect."""

    @property
    @abc.abstractmethod
    def type_mapper(self) -> type[SqlglotType]:
        """The type mapper for the backend."""

    def _compile_builtin_udf(self, udf_node: ops.ScalarUDF) -> None:  # noqa: B027
        """No-op."""

    def _compile_python_udf(self, udf_node: ops.ScalarUDF) -> None:
        raise NotImplementedError(
            f"Python UDFs are not supported in the {self.dialect} backend"
        )

    def _compile_pyarrow_udf(self, udf_node: ops.ScalarUDF) -> None:
        raise NotImplementedError(
            f"PyArrow UDFs are not supported in the {self.dialect} backend"
        )

    def _compile_pandas_udf(self, udf_node: ops.ScalarUDF) -> str:
        raise NotImplementedError(
            f"pandas UDFs are not supported in the {self.dialect} backend"
        )

    # Concrete API

    def if_(self, condition, true, false: sge.Expression | None = None) -> sge.If:
        return sge.If(
            this=sge.convert(condition),
            true=sge.convert(true),
            false=None if false is None else sge.convert(false),
        )

    def cast(self, arg, to: dt.DataType) -> sge.Cast:
        return sge.Cast(
            this=sge.convert(arg), to=self.type_mapper.from_ibis(to), copy=False
        )

    def _prepare_params(self, params):
        result = {}
        for param, value in params.items():
            node = param.op()
            if isinstance(node, ops.Alias):
                node = node.arg
            result[node] = value
        return result

    def to_sqlglot(
        self,
        expr: ir.Expr,
        *,
        limit: str | None = None,
        params: Mapping[ir.Expr, Any] | None = None,
    ):
        import ibis

        table_expr = expr.as_table()

        if limit == "default":
            limit = ibis.options.sql.default_limit
        if limit is not None:
            table_expr = table_expr.limit(limit)

        if params is None:
            params = {}

        sql = self.translate(table_expr.op(), params=params)
        assert not isinstance(sql, sge.Subquery)

        if isinstance(sql, sge.Table):
            sql = sg.select(STAR, copy=False).from_(sql, copy=False)

        assert not isinstance(sql, sge.Subquery)
        return sql

    def translate(self, op, *, params: Mapping[ir.Value, Any]) -> sge.Expression:
        """Translate an ibis operation to a sqlglot expression.

        Parameters
        ----------
        op
            An ibis operation
        params
            A mapping of expressions to concrete values
        compiler
            An instance of SQLGlotCompiler
        translate_rel
            Relation node translator
        translate_val
            Value node translator

        Returns
        -------
        sqlglot.expressions.Expression
            A sqlglot expression

        """
        # substitute parameters immediately to avoid having to define a
        # ScalarParameter translation rule
        params = self._prepare_params(params)
        if self.lowered_ops:
            op = op.replace(reduce(operator.or_, self.lowered_ops.values()))
        op, ctes = sqlize(
            op,
            params=params,
            rewrites=self.rewrites,
            post_rewrites=self.post_rewrites,
            fuse_selects=options.sql.fuse_selects,
        )

        aliases = {}
        counter = itertools.count()

        def fn(node, __unused__, **kwargs):
            result = self.visit_node(node, **kwargs)

            # if it's not a relation then we don't need to do anything special
            if node is op or not isinstance(node, ops.Relation):
                return result

            # alias ops.Views to their explicitly assigned name otherwise generate
            alias = node.name if isinstance(node, ops.View) else f"t{next(counter)}"
            aliases[node] = alias

            alias = sg.to_identifier(alias, quoted=self.quoted)
            if isinstance(result, sge.Subquery):
                return result.as_(alias, quoted=self.quoted)
            else:
                try:
                    return result.subquery(alias, copy=False)
                except AttributeError:
                    return result.as_(alias, quoted=self.quoted)

        # apply translate rules in topological order
        results = op.map(fn)

        # get the root node as a sqlglot select statement
        out = results[op]
        if isinstance(out, sge.Table):
            out = sg.select(STAR, copy=False).from_(out, copy=False)
        elif isinstance(out, sge.Subquery):
            out = out.this

        merged_ctes = []
        for cte in ctes:
            this = results[cte]
            if "alias" in this.args:
                this = this.this
            modified_cte = sge.CTE(
                alias=sg.to_identifier(aliases[cte], quoted=self.quoted), this=this
            )
            merged_ctes.append(modified_cte)
        merged_ctes.extend(out.ctes)
        out.args.pop("with", None)

        out = reduce(
            lambda parsed, cte: parsed.with_(
                cte.args["alias"],
                as_=cte.args["this"],
                dialect=self.dialect,
                copy=False,
            ),
            merged_ctes,
            out,
        )

        return out

    def visit_node(self, op: ops.Node, **kwargs):
        if isinstance(op, ops.ScalarUDF):
            return self.visit_ScalarUDF(op, **kwargs)
        elif isinstance(op, ops.AggUDF):
            return self.visit_AggUDF(op, **kwargs)
        else:
            method = getattr(self, f"visit_{type(op).__name__}", None)
            if method is not None:
                return method(op, **kwargs)
            else:
                raise com.OperationNotDefinedError(
                    f"No translation rule for {type(op).__name__}"
                )

    def visit_Field(self, op, *, rel, name):
        return sg.column(
            self._gen_valid_name(name), table=rel.alias_or_name, quoted=self.quoted
        )

    def visit_Cast(self, op, *, arg, to):
        from_ = op.arg.dtype

        if from_.is_integer() and to.is_interval():
            return self._make_interval(arg, to.unit)

        return self.cast(arg, to)

    def visit_ScalarSubquery(self, op, *, rel):
        return rel.this.subquery(copy=False)

    def visit_Literal(self, op, *, value, dtype):
        """Compile a literal value.

        This is the default implementation for compiling literal values.

        Most backends should not need to override this method unless they want
        to handle NULL literals as well as every other type of non-null literal
        including integers, floating point numbers, decimals, strings, etc.

        The logic here is:

        1. If the value is None and the type is nullable, return NULL
        1. If the value is None and the type is not nullable, raise an error
        1. Call `visit_NonNullLiteral` method.
        1. If the previous returns `None`, call `visit_DefaultLiteral` method
           else return the result of the previous step.
        """
        if value is None:
            if dtype.nullable:
                return NULL if dtype.is_null() else self.cast(NULL, dtype)
            raise com.UnsupportedOperationError(
                f"Unsupported NULL for non-nullable type: {dtype!r}"
            )
        else:
            result = self.visit_NonNullLiteral(op, value=value, dtype=dtype)
            if result is None:
                return self.visit_DefaultLiteral(op, value=value, dtype=dtype)
            return result

    def visit_NonNullLiteral(self, op, *, value, dtype):
        """Compile a non-null literal differently than the default implementation.

        Most backends should implement this, but only when they need to handle
        some non-null literal differently than the default implementation
        (`visit_DefaultLiteral`).

        Return `None` from an override of this method to fall back to
        `visit_DefaultLiteral`.
        """
        return self.visit_DefaultLiteral(op, value=value, dtype=dtype)

    def visit_DefaultLiteral(self, op, *, value, dtype):
        """Compile a literal with a non-null value.

        This is the default implementation for compiling non-null literals.

        Most backends should not need to override this method unless they want
        to handle compiling every kind of non-null literal value.
        """
        if dtype.is_integer():
            return sge.convert(value)
        elif dtype.is_floating():
            if math.isnan(value):
                return self.NAN
            elif math.isinf(value):
                return self.POS_INF if value > 0 else self.NEG_INF
            return sge.convert(value)
        elif dtype.is_decimal():
            return self.cast(str(value), dtype)
        elif dtype.is_interval():
            return sge.Interval(
                this=sge.convert(str(value)),
                unit=sge.Var(this=dtype.resolution.upper()),
            )
        elif dtype.is_boolean():
            return sge.Boolean(this=bool(value))
        elif dtype.is_string():
            return sge.convert(value)
        elif dtype.is_inet() or dtype.is_macaddr():
            return sge.convert(str(value))
        elif dtype.is_timestamp() or dtype.is_time():
            return self.cast(value.isoformat(), dtype)
        elif dtype.is_date():
            return self.f.datefromparts(value.year, value.month, value.day)
        elif dtype.is_array():
            value_type = dtype.value_type
            return self.f.array(
                *(
                    self.visit_Literal(
                        ops.Literal(v, value_type), value=v, dtype=value_type
                    )
                    for v in value
                )
            )
        elif dtype.is_map():
            key_type = dtype.key_type
            keys = self.f.array(
                *(
                    self.visit_Literal(
                        ops.Literal(k, key_type), value=k, dtype=key_type
                    )
                    for k in value.keys()
                )
            )

            value_type = dtype.value_type
            values = self.f.array(
                *(
                    self.visit_Literal(
                        ops.Literal(v, value_type), value=v, dtype=value_type
                    )
                    for v in value.values()
                )
            )

            return self.f.map(keys, values)
        elif dtype.is_struct():
            items = [
                self.visit_Literal(
                    ops.Literal(v, field_dtype), value=v, dtype=field_dtype
                ).as_(k, quoted=self.quoted)
                for field_dtype, (k, v) in zip(dtype.types, value.items())
            ]
            return sge.Struct.from_arg_list(items)
        elif dtype.is_uuid():
            return self.cast(str(value), dtype)
        elif dtype.is_geospatial():
            args = [value.wkt]
            if (srid := dtype.srid) is not None:
                args.append(srid)
            return self.f.st_geomfromtext(*args)

        raise NotImplementedError(f"Unsupported type: {dtype!r}")

    def visit_BitwiseNot(self, op, *, arg):
        return sge.BitwiseNot(this=arg)

    ### Mathematical Calisthenics

    def visit_E(self, op):
        return self.f.exp(1)

    def visit_Log(self, op, *, arg, base):
        if base is None:
            return self.f.ln(arg)
        elif str(base) in ("2", "10"):
            return self.f[f"log{base}"](arg)
        else:
            return self.f.ln(arg) / self.f.ln(base)

    def visit_Clip(self, op, *, arg, lower, upper):
        if upper is not None:
            arg = self.if_(arg.is_(NULL), arg, self.f.least(upper, arg))

        if lower is not None:
            arg = self.if_(arg.is_(NULL), arg, self.f.greatest(lower, arg))

        return arg

    def visit_FloorDivide(self, op, *, left, right):
        return self.cast(self.f.floor(sge.paren(left) / sge.paren(right)), op.dtype)

    def visit_Ceil(self, op, *, arg):
        return self.cast(self.f.ceil(arg), op.dtype)

    def visit_Floor(self, op, *, arg):
        return self.cast(self.f.floor(arg), op.dtype)

    def visit_Round(self, op, *, arg, digits):
        return self.cast(self.f.round(arg, digits), op.dtype)

    ### Dtype Dysmorphia

    def visit_TryCast(self, op, *, arg, to):
        return sge.TryCast(this=arg, to=self.type_mapper.from_ibis(to))

    ### Comparator Conundrums

    def visit_Between(self, op, *, arg, lower_bound, upper_bound):
        return sge.Between(this=arg, low=lower_bound, high=upper_bound)

    def visit_Negate(self, op, *, arg):
        return -sge.paren(arg, copy=False)

    def visit_Not(self, op, *, arg):
        if isinstance(arg, sge.Filter):
            return sge.Filter(
                this=sg.not_(arg.this, copy=False), expression=arg.expression
            )
        return sg.not_(sge.paren(arg, copy=False))

    ### Timey McTimeFace

    def visit_Time(self, op, *, arg):
        return self.cast(arg, to=dt.time)

    def visit_TimestampNow(self, op):
        return sge.CurrentTimestamp()

    def visit_DateNow(self, op):
        return sge.CurrentDate()

    def visit_Strftime(self, op, *, arg, format_str):
        return sge.TimeToStr(this=arg, format=format_str)

    def visit_ExtractEpochSeconds(self, op, *, arg):
        return self.f.epoch(self.cast(arg, dt.timestamp))

    def visit_ExtractYear(self, op, *, arg):
        return self.f.extract(self.v.year, arg)

    def visit_ExtractMonth(self, op, *, arg):
        return self.f.extract(self.v.month, arg)

    def visit_ExtractDay(self, op, *, arg):
        return self.f.extract(self.v.day, arg)

    def visit_ExtractDayOfYear(self, op, *, arg):
        return self.f.extract(self.v.dayofyear, arg)

    def visit_ExtractQuarter(self, op, *, arg):
        return self.f.extract(self.v.quarter, arg)

    def visit_ExtractWeekOfYear(self, op, *, arg):
        return self.f.extract(self.v.week, arg)

    def visit_ExtractHour(self, op, *, arg):
        return self.f.extract(self.v.hour, arg)

    def visit_ExtractMinute(self, op, *, arg):
        return self.f.extract(self.v.minute, arg)

    def visit_ExtractSecond(self, op, *, arg):
        return self.f.extract(self.v.second, arg)

    def visit_TimestampTruncate(self, op, *, arg, unit):
        unit_mapping = {
            "Y": "year",
            "Q": "quarter",
            "M": "month",
            "W": "week",
            "D": "day",
            "h": "hour",
            "m": "minute",
            "s": "second",
            "ms": "ms",
            "us": "us",
        }

        if (raw_unit := unit_mapping.get(unit.short)) is None:
            raise com.UnsupportedOperationError(
                f"Unsupported truncate unit {unit.short!r}"
            )

        return self.f.date_trunc(raw_unit, arg)

    def visit_DateTruncate(self, op, *, arg, unit):
        return self.visit_TimestampTruncate(op, arg=arg, unit=unit)

    def visit_TimeTruncate(self, op, *, arg, unit):
        return self.visit_TimestampTruncate(op, arg=arg, unit=unit)

    def visit_DayOfWeekIndex(self, op, *, arg):
        return (self.f.dayofweek(arg) + 6) % 7

    def visit_DayOfWeekName(self, op, *, arg):
        # day of week number is 0-indexed
        # Sunday == 0
        # Saturday == 6
        return sge.Case(
            this=(self.f.dayofweek(arg) + 6) % 7,
            ifs=list(itertools.starmap(self.if_, enumerate(calendar.day_name))),
        )

    def _make_interval(self, arg, unit):
        return sge.Interval(this=arg, unit=self.v[unit.singular])

    def visit_IntervalFromInteger(self, op, *, arg, unit):
        return self._make_interval(arg, unit)

    ### String Instruments
    def visit_Strip(self, op, *, arg):
        return self.f.trim(arg, string.whitespace)

    def visit_RStrip(self, op, *, arg):
        return self.f.rtrim(arg, string.whitespace)

    def visit_LStrip(self, op, *, arg):
        return self.f.ltrim(arg, string.whitespace)

    def visit_LPad(self, op, *, arg, length, pad):
        return self.f.lpad(arg, self.f.greatest(self.f.length(arg), length), pad)

    def visit_RPad(self, op, *, arg, length, pad):
        return self.f.rpad(arg, self.f.greatest(self.f.length(arg), length), pad)

    def visit_Substring(self, op, *, arg, start, length):
        if isinstance(op.length, ops.Literal) and (value := op.length.value) < 0:
            raise com.IbisInputError(
                f"Length parameter must be a non-negative value; got {value}"
            )
        start += 1
        start = self.if_(start >= 1, start, start + self.f.length(arg))
        if length is None:
            return self.f.substring(arg, start)
        return self.f.substring(arg, start, length)

    def visit_StringFind(self, op, *, arg, substr, start, end):
        if end is not None:
            raise com.UnsupportedOperationError(
                "String find doesn't support `end` argument"
            )

        if start is not None:
            arg = self.f.substr(arg, start + 1)
            pos = self.f.strpos(arg, substr)
            return self.if_(pos > 0, pos + start, 0)

        return self.f.strpos(arg, substr)

    def visit_RegexReplace(self, op, *, arg, pattern, replacement):
        return self.f.regexp_replace(arg, pattern, replacement, "g")

    def visit_StringConcat(self, op, *, arg):
        return self.f.concat(*arg)

    def visit_StringJoin(self, op, *, sep, arg):
        return self.f.concat_ws(sep, *arg)

    def visit_StringSQLLike(self, op, *, arg, pattern, escape):
        return arg.like(pattern)

    def visit_StringSQLILike(self, op, *, arg, pattern, escape):
        return arg.ilike(pattern)

    ### NULL PLAYER CHARACTER
    def visit_IsNull(self, op, *, arg):
        return arg.is_(NULL)

    def visit_NotNull(self, op, *, arg):
        return arg.is_(sg.not_(NULL, copy=False))

    def visit_InValues(self, op, *, value, options):
        return value.isin(*options)

    def visit_StringToTime(self, op, *, arg, format_str):
        return self.f.time(self.f.str_to_time(arg, format_str))

    ### Counting

    def visit_CountDistinct(self, op, *, arg, where):
        return self.agg.count(sge.Distinct(expressions=[arg]), where=where)

    def visit_CountDistinctStar(self, op, *, arg, where):
        return self.agg.count(sge.Distinct(expressions=[STAR]), where=where)

    def visit_CountStar(self, op, *, arg, where):
        return self.agg.count(STAR, where=where)

    def visit_Kurtosis(self, op, *, arg, where, how: Literal["sample", "pop"]):
        if op.arg.dtype.is_boolean():
            arg = self.cast(arg, dt.int32)

        if how == "sample":
            return self.agg.kurtosis(arg, where=where)
        else:
            return self.agg.kurtosis_pop(arg, where=where)

    def visit_Sum(self, op, *, arg, where):
        if op.arg.dtype.is_boolean():
            arg = self.cast(arg, dt.int32)
        return self.agg.sum(arg, where=where)

    def visit_Mean(self, op, *, arg, where):
        if op.arg.dtype.is_boolean():
            arg = self.cast(arg, dt.int32)
        return self.agg.avg(arg, where=where)

    def visit_Min(self, op, *, arg, where):
        if op.arg.dtype.is_boolean():
            return self.agg.bool_and(arg, where=where)
        return self.agg.min(arg, where=where)

    def visit_Max(self, op, *, arg, where):
        if op.arg.dtype.is_boolean():
            return self.agg.bool_or(arg, where=where)
        return self.agg.max(arg, where=where)

    ### Stats

    def visit_VarianceStandardDevCovariance(self, op, *, how, where, **kw):
        hows = {"sample": "samp", "pop": "pop"}
        funcs = {
            ops.Variance: "var",
            ops.StandardDev: "stddev",
            ops.Covariance: "covar",
        }

        args = []

        for oparg, arg in zip(op.args, kw.values()):
            if (arg_dtype := oparg.dtype).is_boolean():
                arg = self.cast(arg, dt.Int32(nullable=arg_dtype.nullable))
            args.append(arg)

        funcname = f"{funcs[type(op)]}_{hows[how]}"
        return self.agg[funcname](*args, where=where)

    visit_Variance = visit_StandardDev = visit_Covariance = (
        visit_VarianceStandardDevCovariance
    )

    def visit_SimpleCase(self, op, *, base=None, cases, results, default):
        return sge.Case(
            this=base, ifs=list(map(self.if_, cases, results)), default=default
        )

    visit_SearchedCase = visit_SimpleCase

    def visit_ExistsSubquery(self, op, *, rel):
        select = rel.this.select(1, append=False)
        return self.f.exists(select)

    def visit_InSubquery(self, op, *, rel, needle):
        query = rel.this
        if not isinstance(query, sge.Select):
            query = sg.select(STAR).from_(query)
        return needle.isin(query=query)

    def visit_Array(self, op, *, exprs):
        return self.f.array(*exprs)

    def visit_StructColumn(self, op, *, names, values):
        return sge.Struct.from_arg_list(
            [value.as_(name, quoted=self.quoted) for name, value in zip(names, values)]
        )

    def visit_StructField(self, op, *, arg, field):
        return sge.Dot(this=arg, expression=sg.to_identifier(field, quoted=self.quoted))

    def visit_IdenticalTo(self, op, *, left, right):
        return sge.NullSafeEQ(this=left, expression=right)

    def visit_Greatest(self, op, *, arg):
        return self.f.greatest(*arg)

    def visit_Least(self, op, *, arg):
        return self.f.least(*arg)

    def visit_Coalesce(self, op, *, arg):
        return self.f.coalesce(*arg)

    ### Ordering and window functions

    def visit_SortKey(self, op, *, expr, ascending: bool, nulls_first: bool):
        return sge.Ordered(this=expr, desc=not ascending, nulls_first=nulls_first)

    def visit_ApproxMedian(self, op, *, arg, where):
        return self.agg.approx_quantile(arg, 0.5, where=where)

    def visit_WindowBoundary(self, op, *, value, preceding):
        # TODO: bit of a hack to return a dict, but there's no sqlglot expression
        # that corresponds to _only_ this information
        return {"value": value, "side": "preceding" if preceding else "following"}

    def visit_WindowFunction(self, op, *, how, func, start, end, group_by, order_by):
        if start is None:
            start = {}
        if end is None:
            end = {}

        start_value = start.get("value", "UNBOUNDED")
        start_side = start.get("side", "PRECEDING")
        end_value = end.get("value", "UNBOUNDED")
        end_side = end.get("side", "FOLLOWING")

        if getattr(start_value, "this", None) == "0":
            start_value = "CURRENT ROW"
            start_side = None

        if getattr(end_value, "this", None) == "0":
            end_value = "CURRENT ROW"
            end_side = None

        spec = sge.WindowSpec(
            kind=how.upper(),
            start=start_value,
            start_side=start_side,
            end=end_value,
            end_side=end_side,
            over="OVER",
        )
        order = sge.Order(expressions=order_by) if order_by else None

        spec = self._minimize_spec(op, spec)

        return sge.Window(this=func, partition_by=group_by, order=order, spec=spec)

    @staticmethod
    def _minimize_spec(op, spec):
        return spec

    def visit_LagLead(self, op, *, arg, offset, default):
        args = [arg]

        if default is not None:
            if offset is None:
                offset = 1

            args.append(offset)
            args.append(default)
        elif offset is not None:
            args.append(offset)

        return self.f[type(op).__name__.lower()](*args)

    visit_Lag = visit_Lead = visit_LagLead

    def visit_Argument(self, op, *, name: str, shape, dtype):
        return sg.to_identifier(op.param)

    def visit_RowID(self, op, *, table):
        return sg.column(
            op.name, table=table.alias_or_name, quoted=self.quoted, copy=False
        )

    # TODO(kszucs): this should be renamed to something UDF related
    def __sql_name__(self, op: ops.ScalarUDF | ops.AggUDF) -> str:
        # for builtin functions use the exact function name, otherwise use the
        # generated name to handle the case of redefinition
        funcname = (
            op.__func_name__
            if op.__input_type__ == InputType.BUILTIN
            else type(op).__name__
        )

        # not actually a table, but easier to quote individual namespace
        # components this way
        namespace = op.__udf_namespace__
        return sg.table(funcname, db=namespace.database, catalog=namespace.catalog).sql(
            self.dialect
        )

    def visit_ScalarUDF(self, op, **kw):
        return self.f[self.__sql_name__(op)](*kw.values())

    def visit_AggUDF(self, op, *, where, **kw):
        return self.agg[self.__sql_name__(op)](*kw.values(), where=where)

    def visit_TimestampDelta(self, op, *, part, left, right):
        # dialect is necessary due to sqlglot's default behavior
        # of `part` coming last
        return sge.DateDiff(
            this=left, expression=right, unit=part, dialect=self.dialect
        )

    visit_TimeDelta = visit_DateDelta = visit_TimestampDelta

    def visit_TimestampBucket(self, op, *, arg, interval, offset):
        origin = self.f.cast("epoch", self.type_mapper.from_ibis(dt.timestamp))
        if offset is not None:
            origin += offset
        return self.f.time_bucket(interval, arg, origin)

    def visit_ArrayConcat(self, op, *, arg):
        return sge.ArrayConcat(this=arg[0], expressions=list(arg[1:]))

    ## relations

    @staticmethod
    def _gen_valid_name(name: str) -> str:
        """Generate a valid name for a value expression.

        Override this method if the dialect has restrictions on valid
        identifiers even when quoted.

        See the BigQuery backend's implementation for an example.
        """
        return name

    def _cleanup_names(self, exprs: Mapping[str, sge.Expression]):
        """Compose `_gen_valid_name` and `_dedup_name` to clean up names in projections."""

        for name, value in exprs.items():
            name = self._gen_valid_name(name)
            if isinstance(value, sge.Column) and name == value.name:
                # don't alias columns that are already named the same as their alias
                yield value
            else:
                yield value.as_(name, quoted=self.quoted, copy=False)

    def visit_Select(
        self, op, *, parent, selections, predicates, qualified, sort_keys, distinct
    ):
        # if we've constructed a useless projection return the parent relation
        if not (selections or predicates or qualified or sort_keys or distinct):
            return parent

        result = parent

        if selections:
            # if there are `qualify` predicates then sqlglot adds a hidden
            # column to implement the functionality if the dialect doesn't
            # support it
            #
            # using STAR in that case would lead to an extra column, so in that
            # case we have to spell out the columns
            if op.is_star_selection() and (not qualified or self.supports_qualify):
                fields = [STAR]
            else:
                fields = self._cleanup_names(selections)
            result = sg.select(*fields, copy=False).from_(result, copy=False)

        if predicates:
            result = result.where(*predicates, copy=False)

        if qualified:
            result = result.qualify(*qualified, copy=False)

        if sort_keys:
            result = result.order_by(*sort_keys, copy=False)

        if distinct:
            result = result.distinct()

        return result

    def visit_DummyTable(self, op, *, values):
        return sg.select(*self._cleanup_names(values), copy=False)

    def visit_UnboundTable(
        self, op, *, name: str, schema: sch.Schema, namespace: ops.Namespace
    ) -> sg.Table:
        return sg.table(
            name, db=namespace.database, catalog=namespace.catalog, quoted=self.quoted
        )

    def visit_InMemoryTable(
        self, op, *, name: str, schema: sch.Schema, data
    ) -> sg.Table:
        return sg.table(name, quoted=self.quoted)

    def visit_DatabaseTable(
        self,
        op,
        *,
        name: str,
        schema: sch.Schema,
        source: Any,
        namespace: ops.Namespace,
    ) -> sg.Table:
        return sg.table(
            name, db=namespace.database, catalog=namespace.catalog, quoted=self.quoted
        )

    def visit_SelfReference(self, op, *, parent, identifier):
        return parent

    visit_JoinReference = visit_SelfReference

    def visit_JoinChain(self, op, *, first, rest, values):
        result = sg.select(*self._cleanup_names(values), copy=False).from_(
            first, copy=False
        )

        for link in rest:
            if isinstance(link, sge.Alias):
                link = link.this
            result = result.join(link, copy=False)
        return result

    def visit_JoinLink(self, op, *, how, table, predicates):
        sides = {
            "inner": None,
            "left": "left",
            "right": "right",
            "semi": "left",
            "anti": "left",
            "cross": None,
            "outer": "full",
            "asof": "asof",
            "any_left": "left",
            "any_inner": None,
            "positional": None,
        }
        kinds = {
            "any_left": "any",
            "any_inner": "any",
            "asof": "left",
            "inner": "inner",
            "left": "outer",
            "right": "outer",
            "semi": "semi",
            "anti": "anti",
            "cross": "cross",
            "outer": "outer",
            "positional": "positional",
        }
        assert predicates or how in {
            "cross",
            "positional",
        }, "expected non-empty predicates when not a cross join"
        on = sg.and_(*predicates) if predicates else None
        return sge.Join(this=table, side=sides[how], kind=kinds[how], on=on)

    @staticmethod
    def _generate_groups(groups):
        return map(sge.convert, range(1, len(groups) + 1))

    def visit_Aggregate(self, op, *, parent, groups, metrics):
        sel = sg.select(
            *self._cleanup_names(groups), *self._cleanup_names(metrics), copy=False
        ).from_(parent, copy=False)

        if groups:
            sel = sel.group_by(*self._generate_groups(groups.values()), copy=False)

        return sel

    @classmethod
    def _add_parens(cls, sg_expr):
        if type(sg_expr) in cls.SQLGLOT_NEEDS_PARENS:
            return sge.paren(sg_expr, copy=False)
        return sg_expr

    def visit_Union(self, op, *, left, right, distinct):
        if isinstance(left, (sge.Table, sge.Subquery)):
            left = sg.select(STAR, copy=False).from_(left, copy=False)

        if isinstance(right, (sge.Table, sge.Subquery)):
            right = sg.select(STAR, copy=False).from_(right, copy=False)

        return sg.union(
            left.args.get("this", left),
            right.args.get("this", right),
            distinct=distinct,
            copy=False,
        )

    def visit_Intersection(self, op, *, left, right, distinct):
        if isinstance(left, (sge.Table, sge.Subquery)):
            left = sg.select(STAR, copy=False).from_(left, copy=False)

        if isinstance(right, (sge.Table, sge.Subquery)):
            right = sg.select(STAR, copy=False).from_(right, copy=False)

        return sg.intersect(
            left.args.get("this", left),
            right.args.get("this", right),
            distinct=distinct,
            copy=False,
        )

    def visit_Difference(self, op, *, left, right, distinct):
        if isinstance(left, (sge.Table, sge.Subquery)):
            left = sg.select(STAR, copy=False).from_(left, copy=False)

        if isinstance(right, (sge.Table, sge.Subquery)):
            right = sg.select(STAR, copy=False).from_(right, copy=False)

        return sg.except_(
            left.args.get("this", left),
            right.args.get("this", right),
            distinct=distinct,
            copy=False,
        )

    def visit_Sample(
        self, op, *, parent, fraction: float, method: str, seed: int | None, **_
    ):
        sample = sge.TableSample(
            method="bernoulli" if method == "row" else "system",
            percent=sge.convert(fraction * 100.0),
            seed=None if seed is None else sge.convert(seed),
        )
        # sample was changed to be owned by the table being sampled in 25.17.0
        #
        # this is a small workaround for backwards compatibility
        if "this" in sample.__class__.arg_types:
            sample.args["this"] = parent
        else:
            parent.args["sample"] = sample
        return sg.select(STAR).from_(parent)

    def visit_Limit(self, op, *, parent, n, offset):
        # push limit/offset into subqueries
        if isinstance(parent, sge.Subquery) and parent.this.args.get("limit") is None:
            result = parent.this.copy()
            alias = parent.alias
        else:
            result = sg.select(STAR, copy=False).from_(parent, copy=False)
            alias = None

        if isinstance(n, int):
            result = result.limit(n, copy=False)
        elif n is not None:
            result = result.limit(
                sg.select(n, copy=False).from_(parent, copy=False).subquery(copy=False),
                copy=False,
            )
        else:
            assert n is None, n
            if self.no_limit_value is not None:
                result = result.limit(self.no_limit_value, copy=False)

        assert offset is not None, "offset is None"

        if not isinstance(offset, int):
            skip = offset
            skip = (
                sg.select(skip, copy=False)
                .from_(parent, copy=False)
                .subquery(copy=False)
            )
        elif not offset:
            if alias is not None:
                return result.subquery(alias, copy=False)
            return result
        else:
            skip = offset

        result = result.offset(skip, copy=False)
        if alias is not None:
            return result.subquery(alias, copy=False)
        return result

    def visit_CTE(self, op, *, parent):
        return sg.table(parent.alias_or_name, quoted=self.quoted)

    def visit_View(self, op, *, child, name: str):
        if isinstance(child, sge.Table):
            child = sg.select(STAR, copy=False).from_(child, copy=False)
        else:
            child = child.copy()

        if isinstance(child, sge.Subquery):
            return child.as_(name, quoted=self.quoted)
        else:
            try:
                return child.subquery(name, copy=False)
            except AttributeError:
                return child.as_(name, quoted=self.quoted)

    def visit_SQLStringView(self, op, *, query: str, child, schema):
        return sg.parse_one(query, read=self.dialect)

    def visit_SQLQueryResult(self, op, *, query, schema, source):
        return sg.parse_one(query, dialect=self.dialect).subquery(copy=False)

    def binop(self, sg_cls, left, right):
        # If the op is associative we can skip parenthesizing ops of the same
        # type if they're on the left, since they would evaluate the same.
        # SQLGlot has an optimizer for generating long sql chains of the same
        # op of this form without recursion, by avoiding parenthesis in this
        # common case we can make use of this optimization to handle large
        # operator chains.
        if not (sg_cls in self.SQLGLOT_ASSOCIATIVE_OPS and type(left) is sg_cls):
            left = self._add_parens(left)
        return sg_cls(this=left, expression=self._add_parens(right))

    def visit_Undefined(self, op, **_):
        raise com.OperationNotDefinedError(
            f"Compilation rule for {type(op).__name__!r} operation is not defined"
        )

    def visit_Unsupported(self, op, **_):
        raise com.UnsupportedOperationError(
            f"{type(op).__name__!r} operation is not supported in the {self.dialect} backend"
        )

    def visit_DropColumns(self, op, *, parent, columns_to_drop):
        # the generated query will be huge for wide tables
        #
        # TODO: figure out a way to produce an IR that only contains exactly
        # what is used
        parent_alias = parent.alias_or_name
        quoted = self.quoted
        columns_to_keep = (
            sg.column(column, table=parent_alias, quoted=quoted)
            for column in op.schema.names
        )
        return sg.select(*columns_to_keep).from_(parent)

    def add_query_to_expr(self, *, name: str, table: ir.Table, query: str) -> str:
        dialect = self.dialect

        compiled_ibis_expr = self.to_sqlglot(table)
        compiled_query = sg.parse_one(query, read=dialect)

        ctes = [
            *compiled_ibis_expr.ctes,
            sge.CTE(
                alias=sg.to_identifier(name, quoted=self.quoted),
                this=compiled_ibis_expr,
            ),
            *compiled_query.ctes,
        ]
        compiled_ibis_expr.args.pop("with", None)
        compiled_query.args.pop("with", None)

        # pull existing CTEs from the compiled Ibis expression and combine them
        # with the new query
        parsed = reduce(
            lambda parsed, cte: parsed.with_(cte.args["alias"], as_=cte.args["this"]),
            ctes,
            compiled_query,
        )

        # generate the SQL string
        return parsed.sql(dialect)

    def _make_sample_backwards_compatible(self, *, sample, parent):
        # sample was changed to be owned by the table being sampled in 25.17.0
        #
        # this is a small workaround for backwards compatibility
        if "this" in sample.__class__.arg_types:
            sample.args["this"] = parent
        else:
            parent.args["sample"] = sample
        return sg.select(STAR).from_(parent)


# `__init_subclass__` is uncalled for subclasses - we manually call it here to
# autogenerate the base class implementations as well.
SQLGlotCompiler.__init_subclass__()
