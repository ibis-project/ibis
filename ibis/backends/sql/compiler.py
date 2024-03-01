from __future__ import annotations

import abc
import calendar
import itertools
import math
import string
from collections.abc import Iterator, Mapping
from functools import partial, reduce
from itertools import starmap
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import sqlglot as sg
import sqlglot.expressions as sge
import toolz
from public import public

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.rewrites import (
    add_one_to_nth_value_input,
    add_order_by_to_empty_ranking_window_functions,
    empty_in_values_right_side,
    one_to_zero_index,
    rewrite_capitalize,
    sqlize,
)
from ibis.expr.operations.udf import InputType
from ibis.expr.rewrites import replace_bucket

if TYPE_CHECKING:
    from collections.abc import Iterable

    import ibis.expr.schema as sch
    import ibis.expr.types as ir
    from ibis.backends.sql.datatypes import SqlglotType


def get_leaf_classes(op):
    for child_class in op.__subclasses__():
        if not child_class.__subclasses__():
            yield child_class
        else:
            yield from get_leaf_classes(child_class)


ALL_OPERATIONS = frozenset(get_leaf_classes(ops.Node))


class AggGen:
    __slots__ = ("aggfunc",)

    def __init__(self, *, aggfunc: Callable) -> None:
        self.aggfunc = aggfunc

    def __getattr__(self, name: str) -> partial:
        return partial(self.aggfunc, name)

    def __getitem__(self, key: str) -> partial:
        return getattr(self, key)


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
    __slots__ = ("namespace", "anon")

    def __init__(self, namespace: str | None = None) -> None:
        self.namespace = namespace
        self.anon = AnonymousFuncGen()

    def __getattr__(self, name: str) -> Callable[..., sge.Func]:
        name = ".".join(filter(None, (self.namespace, name)))
        return lambda *args, **kwargs: sg.func(name, *map(sge.convert, args), **kwargs)

    def __getitem__(self, key: str) -> Callable[..., sge.Func]:
        return getattr(self, key)

    def array(self, *args: Any) -> sge.Array:
        if not args:
            return sge.Array(expressions=[])

        first, *rest = args

        if isinstance(first, sge.Select):
            assert (
                not rest
            ), "only one argument allowed when `first` is a select statement"

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
        return sg.column(name, table=self.table)

    def __getitem__(self, key: str) -> sge.Column:
        return sg.column(key, table=self.table)


def paren(expr):
    """Wrap a sqlglot expression in parentheses."""
    return sge.Paren(this=sge.convert(expr))


def parenthesize(op, arg):
    if isinstance(op, (ops.Binary, ops.Unary)):
        return paren(arg)
    # function calls don't need parens
    return arg


C = ColGen()
F = FuncGen()
NULL = sge.Null()
FALSE = sge.false()
TRUE = sge.true()
STAR = sge.Star()


def parenthesize_inputs(f):
    """Decorate a translation rule to parenthesize inputs."""

    def wrapper(self, op, *, left, right):
        return f(
            self,
            op,
            left=self._add_parens(op.left, left),
            right=self._add_parens(op.right, right),
        )

    return wrapper


@public
class SQLGlotCompiler(abc.ABC):
    __slots__ = "agg", "f", "v"

    rewrites: tuple = (
        empty_in_values_right_side,
        add_order_by_to_empty_ranking_window_functions,
        one_to_zero_index,
        add_one_to_nth_value_input,
        replace_bucket,
        rewrite_capitalize,
    )
    """A sequence of rewrites to apply to the expression tree before compilation."""

    no_limit_value: sge.Null | None = None
    """The value to use to indicate no limit."""

    quoted: bool = True
    """Whether to always quote identifiers."""

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

    UNSUPPORTED_OPERATIONS: frozenset[type[ops.Node]] = frozenset()
    """Set of operations the backend doesn't support."""

    SIMPLE_OPS = {
        ops.Abs: "abs",
        ops.Acos: "acos",
        ops.All: "bool_and",
        ops.Any: "bool_or",
        ops.ApproxCountDistinct: "approx_distinct",
        ops.ArgMax: "max_by",
        ops.ArgMin: "min_by",
        ops.ArrayCollect: "array_agg",
        ops.ArrayContains: "array_contains",
        ops.ArrayFlatten: "flatten",
        ops.ArrayLength: "array_size",
        ops.ArraySort: "array_sort",
        ops.ArrayStringJoin: "array_to_string",
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
        ops.First: "first",
        ops.FirstValue: "first_value",
        ops.GroupConcat: "group_concat",
        ops.IfElse: "if",
        ops.IsInf: "isinf",
        ops.IsNan: "isnan",
        ops.JSONGetItem: "json_extract",
        ops.LPad: "lpad",
        ops.Last: "last",
        ops.LastValue: "last_value",
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
        ops.RPad: "rpad",
        ops.Radians: "radians",
        ops.RandomScalar: "random",
        ops.RegexSearch: "regexp_like",
        ops.RegexSplit: "regexp_split",
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
        ops.StringToTimestamp: "str_to_time",
        ops.Tan: "tan",
        ops.Translate: "translate",
        ops.Unnest: "explode",
        ops.Uppercase: "upper",
    }

    BINARY_INFIX_OPS = (
        # Binary operations
        ops.Add,
        ops.Subtract,
        ops.Multiply,
        ops.Divide,
        ops.Modulus,
        ops.Power,
        # Comparisons
        ops.GreaterEqual,
        ops.Greater,
        ops.LessEqual,
        ops.Less,
        ops.Equals,
        ops.NotEquals,
        # Boolean comparisons
        ops.And,
        ops.Or,
        ops.Xor,
        # Bitwise business
        ops.BitwiseLeftShift,
        ops.BitwiseRightShift,
        ops.BitwiseAnd,
        ops.BitwiseOr,
        ops.BitwiseXor,
        # Time arithmetic
        ops.DateAdd,
        ops.DateSub,
        ops.DateDiff,
        ops.TimestampAdd,
        ops.TimestampSub,
        ops.TimestampDiff,
        # Interval Marginalia
        ops.IntervalAdd,
        ops.IntervalMultiply,
        ops.IntervalSubtract,
    )

    NEEDS_PARENS = BINARY_INFIX_OPS + (ops.IsNull,)

    def __init__(self) -> None:
        self.agg = AggGen(aggfunc=self._aggregate)
        self.f = FuncGen()
        self.v = VarGen()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def methodname(op: type) -> str:
            assert isinstance(type(op), type), type(op)
            return f"visit_{op.__name__}"

        def make_impl(op, target_name):
            assert isinstance(type(op), type), type(op)

            if issubclass(op, ops.Reduction):

                def impl(self, _, *, _name: str = target_name, where, **kw):
                    return self.agg[_name](*kw.values(), where=where)

            else:

                def impl(self, _, *, _name: str = target_name, **kw):
                    return self.f[_name](*kw.values())

            return impl

        # unconditionally raise an exception for unsupported operations
        for op in cls.UNSUPPORTED_OPERATIONS:
            # change to visit_Unsupported in a follow up
            # TODO: handle geoespatial ops as a separate case?
            setattr(cls, methodname(op), cls.visit_Undefined)

        # override existing base class implementations
        for op, target_name in cls.SIMPLE_OPS.items():
            setattr(cls, methodname(op), make_impl(op, target_name))

        # add simple ops that are not already implemented
        for op, target_name in SQLGlotCompiler.SIMPLE_OPS.items():
            name = methodname(op)
            if not hasattr(cls, name):
                setattr(cls, name, make_impl(op, target_name))

        # raise on any remaining unsupported operations
        for op in ALL_OPERATIONS:
            name = methodname(op)
            if not hasattr(cls, name):
                setattr(cls, name, cls.visit_Undefined)

    @property
    @abc.abstractmethod
    def dialect(self) -> str:
        """Backend dialect."""

    @property
    @abc.abstractmethod
    def type_mapper(self) -> type[SqlglotType]:
        """The type mapper for the backend."""

    @abc.abstractmethod
    def _aggregate(self, funcname, *args, where):
        """Translate an aggregate function.

        Three flavors of filtering aggregate function inputs:

        1. supports filter (duckdb, postgres, others)
           e.g.: sum(x) filter (where predicate)
        2. use null to filter out
           e.g.: sum(if(predicate, x, NULL))
        3. clickhouse's ${func}If implementation, e.g.:
           sumIf(predicate, x)
        """

    # Concrete API

    def if_(self, condition, true, false: sge.Expression | None = None) -> sge.If:
        return sge.If(
            this=sge.convert(condition),
            true=sge.convert(true),
            false=false if false is None else sge.convert(false),
        )

    def cast(self, arg, to: dt.DataType) -> sge.Cast:
        return sg.cast(sge.convert(arg), to=self.type_mapper.from_ibis(to))

    def _prepare_params(self, params):
        result = {}
        for param, value in params.items():
            node = param.op()
            if isinstance(node, ops.Alias):
                node = node.arg
            result[node] = value
        return result

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
        op, ctes = sqlize(op, params=params, rewrites=self.rewrites)

        aliases = {}
        counter = itertools.count()

        def fn(node, _, **kwargs):
            result = self.visit_node(node, **kwargs)

            # if it's not a relation then we don't need to do anything special
            if node is op or not isinstance(node, ops.Relation):
                return result

            # alias ops.Views to their explicitly assigned name otherwise generate
            alias = node.name if isinstance(node, ops.View) else f"t{next(counter)}"
            aliases[node] = alias

            alias = sg.to_identifier(alias, quoted=self.quoted)
            try:
                return result.subquery(alias)
            except AttributeError:
                return result.as_(alias, quoted=self.quoted)

        # apply translate rules in topological order
        results = op.map(fn)

        # get the root node as a sqlglot select statement
        out = results[op]
        if isinstance(out, sge.Table):
            out = sg.select(STAR).from_(out)
        elif isinstance(out, sge.Subquery):
            out = out.this

        # add cte definitions to the select statement
        for cte in ctes:
            alias = sg.to_identifier(aliases[cte], quoted=self.quoted)
            out = out.with_(alias, as_=results[cte].this, dialect=self.dialect)

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
        return self.cast(arg, to)

    def visit_ScalarSubquery(self, op, *, rel):
        return rel.this.subquery()

    def visit_Alias(self, op, *, arg, name):
        return arg

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
                this=sge.convert(str(value)), unit=dtype.resolution.upper()
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
        return self.cast(self.f.floor(left / right), op.dtype)

    def visit_Ceil(self, op, *, arg):
        return self.cast(self.f.ceil(arg), op.dtype)

    def visit_Floor(self, op, *, arg):
        return self.cast(self.f.floor(arg), op.dtype)

    def visit_Round(self, op, *, arg, digits):
        if digits is not None:
            return sge.Round(this=arg, decimals=digits)
        return sge.Round(this=arg)

    ### Dtype Dysmorphia

    def visit_TryCast(self, op, *, arg, to):
        return sge.TryCast(this=arg, to=self.type_mapper.from_ibis(to))

    ### Comparator Conundrums

    def visit_Between(self, op, *, arg, lower_bound, upper_bound):
        return sge.Between(this=arg, low=lower_bound, high=upper_bound)

    def visit_Negate(self, op, *, arg):
        return -paren(arg)

    def visit_Not(self, op, *, arg):
        if isinstance(arg, sge.Filter):
            return sge.Filter(this=sg.not_(arg.this), expression=arg.expression)
        return sg.not_(paren(arg))

    ### Timey McTimeFace

    def visit_Time(self, op, *, arg):
        return self.cast(arg, to=dt.time)

    def visit_TimestampNow(self, op):
        return sge.CurrentTimestamp()

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
            "M": "month",
            "W": "week",
            "D": "day",
            "h": "hour",
            "m": "minute",
            "s": "second",
            "ms": "ms",
            "us": "us",
        }

        if (unit := unit_mapping.get(unit.short)) is None:
            raise com.UnsupportedOperationError(f"Unsupported truncate unit {unit}")

        return self.f.date_trunc(unit, arg)

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

    def visit_IntervalFromInteger(self, op, *, arg, unit):
        return sge.Interval(this=sge.convert(arg), unit=unit.singular.upper())

    ### String Instruments

    def visit_Strip(self, op, *, arg):
        return self.f.trim(arg, string.whitespace)

    def visit_RStrip(self, op, *, arg):
        return self.f.rtrim(arg, string.whitespace)

    def visit_LStrip(self, op, *, arg):
        return self.f.ltrim(arg, string.whitespace)

    def visit_Substring(self, op, *, arg, start, length):
        start += 1
        arg_length = self.f.length(arg)

        if length is None:
            return self.if_(
                start >= 1,
                self.f.substring(arg, start),
                self.f.substring(arg, start + arg_length),
            )
        return self.if_(
            start >= 1,
            self.f.substring(arg, start, length),
            self.f.substring(arg, start + arg_length, length),
        )

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
        return arg.is_(sg.not_(NULL))

    def visit_InValues(self, op, *, value, options):
        return value.isin(*options)

    ### Counting

    def visit_CountDistinct(self, op, *, arg, where):
        return self.agg.count(sge.Distinct(expressions=[arg]), where=where)

    def visit_CountDistinctStar(self, op, *, arg, where):
        return self.agg.count(sge.Distinct(expressions=[STAR]), where=where)

    def visit_CountStar(self, op, *, arg, where):
        return self.agg.count(STAR, where=where)

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

    def visit_Quantile(self, op, *, arg, quantile, where):
        suffix = "cont" if op.arg.dtype.is_numeric() else "disc"
        funcname = f"percentile_{suffix}"
        expr = sge.WithinGroup(
            this=self.f[funcname](quantile),
            expression=sge.Order(expressions=[sge.Ordered(this=arg)]),
        )
        if where is not None:
            expr = sge.Filter(this=expr, expression=sge.Where(this=where))
        return expr

    visit_MultiQuantile = visit_Quantile

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

    def visit_Arbitrary(self, op, *, arg, how, where):
        if how == "heavy":
            raise com.UnsupportedOperationError(
                f"how='heavy' not supported in the {self.dialect} backend"
            )
        return self.agg[how](arg, where=where)

    def visit_SimpleCase(self, op, *, base=None, cases, results, default):
        return sge.Case(
            this=base, ifs=list(map(self.if_, cases, results)), default=default
        )

    visit_SearchedCase = visit_SimpleCase

    def visit_ExistsSubquery(self, op, *, rel):
        select = rel.this.select(1, append=False)
        return self.f.exists(select)

    def visit_InSubquery(self, op, *, rel, needle):
        return needle.isin(rel.this)

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

    def visit_SortKey(self, op, *, expr, ascending: bool):
        return sge.Ordered(this=expr, desc=not ascending)

    def visit_ApproxMedian(self, op, *, arg, where):
        return self.agg.approx_quantile(arg, 0.5, where=where)

    def visit_WindowBoundary(self, op, *, value, preceding):
        # TODO: bit of a hack to return a dict, but there's no sqlglot expression
        # that corresponds to _only_ this information
        return {"value": value, "side": "preceding" if preceding else "following"}

    def visit_Window(self, op, *, how, func, start, end, group_by, order_by):
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

        spec = self._minimize_spec(op.start, op.end, spec)

        return sge.Window(this=func, partition_by=group_by, order=order, spec=spec)

    @staticmethod
    def _minimize_spec(start, end, spec):
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
        return sg.column(op.name, table=table.alias_or_name, quoted=self.quoted)

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
        return sg.table(funcname, db=namespace.schema, catalog=namespace.database).sql(
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

    def _dedup_name(
        self, key: str, value: sge.Expression
    ) -> Iterator[sge.Alias | sge.Column]:
        """Don't alias columns that are already named the same as their alias."""
        return (
            value
            if isinstance(value, sge.Column) and key == value.name
            else value.as_(key, quoted=self.quoted)
        )

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
        return starmap(
            self._dedup_name, toolz.keymap(self._gen_valid_name, exprs).items()
        )

    def visit_Select(self, op, *, parent, selections, predicates, sort_keys):
        # if we've constructed a useless projection return the parent relation
        if not selections and not predicates and not sort_keys:
            return parent

        result = parent

        if selections:
            result = sg.select(*self._cleanup_names(selections)).from_(result)

        if predicates:
            result = result.where(*predicates)

        if sort_keys:
            result = result.order_by(*sort_keys)

        return result

    def visit_DummyTable(self, op, *, values):
        return sg.select(*self._cleanup_names(values))

    def visit_UnboundTable(
        self, op, *, name: str, schema: sch.Schema, namespace: ops.Namespace
    ) -> sg.Table:
        return sg.table(
            name, db=namespace.schema, catalog=namespace.database, quoted=self.quoted
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
            name, db=namespace.schema, catalog=namespace.database, quoted=self.quoted
        )

    def visit_SelfReference(self, op, *, parent, identifier):
        return parent

    def visit_JoinChain(self, op, *, first, rest, values):
        result = sg.select(*self._cleanup_names(values)).from_(first)

        for link in rest:
            if isinstance(link, sge.Alias):
                link = link.this
            result = result.join(link)
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
        }
        assert (
            predicates or how == "cross"
        ), "expected non-empty predicates when not a cross join"

        on = sg.and_(*predicates) if predicates else None
        return sge.Join(this=table, side=sides[how], kind=kinds[how], on=on)

    @staticmethod
    def _generate_groups(groups):
        return map(sge.convert, range(1, len(groups) + 1))

    def visit_Aggregate(self, op, *, parent, groups, metrics):
        sel = sg.select(
            *self._cleanup_names(groups), *self._cleanup_names(metrics)
        ).from_(parent)

        if groups:
            sel = sel.group_by(*self._generate_groups(groups.values()))

        return sel

    @classmethod
    def _add_parens(cls, op, sg_expr):
        if isinstance(op, cls.NEEDS_PARENS):
            return paren(sg_expr)
        return sg_expr

    def visit_Filter(self, op, *, parent, predicates):
        predicates = (
            self._add_parens(raw_predicate, predicate)
            for raw_predicate, predicate in zip(op.predicates, predicates)
        )
        try:
            return parent.where(*predicates)
        except AttributeError:
            return sg.select(STAR).from_(parent).where(*predicates)

    def visit_Sort(self, op, *, parent, keys):
        try:
            return parent.order_by(*keys)
        except AttributeError:
            return sg.select(STAR).from_(parent).order_by(*keys)

    def visit_Union(self, op, *, left, right, distinct):
        if isinstance(left, sge.Table):
            left = sg.select(STAR).from_(left)

        if isinstance(right, sge.Table):
            right = sg.select(STAR).from_(right)

        return sg.union(
            left.args.get("this", left),
            right.args.get("this", right),
            distinct=distinct,
        )

    def visit_Intersection(self, op, *, left, right, distinct):
        if isinstance(left, sge.Table):
            left = sg.select(STAR).from_(left)

        if isinstance(right, sge.Table):
            right = sg.select(STAR).from_(right)

        return sg.intersect(
            left.args.get("this", left),
            right.args.get("this", right),
            distinct=distinct,
        )

    def visit_Difference(self, op, *, left, right, distinct):
        if isinstance(left, sge.Table):
            left = sg.select(STAR).from_(left)

        if isinstance(right, sge.Table):
            right = sg.select(STAR).from_(right)

        return sg.except_(
            left.args.get("this", left),
            right.args.get("this", right),
            distinct=distinct,
        )

    def visit_Limit(self, op, *, parent, n, offset):
        # push limit/offset into subqueries
        if isinstance(parent, sge.Subquery) and parent.this.args.get("limit") is None:
            result = parent.this
            alias = parent.alias
        else:
            result = sg.select(STAR).from_(parent)
            alias = None

        if isinstance(n, int):
            result = result.limit(n)
        elif n is not None:
            result = result.limit(sg.select(n).from_(parent).subquery())
        else:
            assert n is None, n
            if self.no_limit_value is not None:
                result = result.limit(self.no_limit_value)

        assert offset is not None, "offset is None"

        if not isinstance(offset, int):
            skip = offset
            skip = sg.select(skip).from_(parent).subquery()
        elif not offset:
            if alias is not None:
                return result.subquery(alias)
            return result
        else:
            skip = offset

        result = result.offset(skip)
        if alias is not None:
            return result.subquery(alias)
        return result

    def visit_Distinct(self, op, *, parent):
        return sg.select(STAR).distinct().from_(parent)

    def visit_DropNa(self, op, *, parent, how, subset):
        if subset is None:
            subset = [
                sg.column(name, table=parent.alias_or_name, quoted=self.quoted)
                for name in op.schema.names
            ]

        if subset:
            predicate = reduce(
                sg.and_ if how == "any" else sg.or_,
                (sg.not_(col.is_(NULL)) for col in subset),
            )
        elif how == "all":
            predicate = FALSE
        else:
            predicate = None

        if predicate is None:
            return parent

        try:
            return parent.where(predicate)
        except AttributeError:
            return sg.select(STAR).from_(parent).where(predicate)

    def visit_FillNa(self, op, *, parent, replacements):
        if isinstance(replacements, Mapping):
            mapping = replacements
        else:
            mapping = {
                name: replacements
                for name, dtype in op.schema.items()
                if dtype.nullable
            }
        exprs = {
            col: (
                self.f.coalesce(sg.column(col, quoted=self.quoted), sge.convert(alt))
                if (alt := mapping.get(col)) is not None
                else sg.column(col, quoted=self.quoted)
            )
            for col in op.schema.keys()
        }
        return sg.select(*self._cleanup_names(exprs)).from_(parent)

    def visit_CTE(self, op, *, parent):
        return sg.table(parent.alias_or_name, quoted=self.quoted)

    def visit_View(self, op, *, child, name: str):
        if isinstance(child, sge.Table):
            child = sg.select(STAR).from_(child)

        try:
            return child.subquery(name)
        except AttributeError:
            return child.as_(name)

    def visit_SQLStringView(self, op, *, query: str, child, schema):
        return sg.parse_one(query, read=self.dialect)

    def visit_SQLQueryResult(self, op, *, query, schema, source):
        return sg.parse_one(query, dialect=self.dialect).subquery()

    def visit_JoinTable(self, op, *, parent, index):
        return parent

    def visit_RegexExtract(self, op, *, arg, pattern, index):
        return self.f.regexp_extract(arg, pattern, index, dialect=self.dialect)

    @parenthesize_inputs
    def visit_Add(self, op, *, left, right):
        return sge.Add(this=left, expression=right)

    visit_DateAdd = visit_TimestampAdd = visit_IntervalAdd = visit_Add

    @parenthesize_inputs
    def visit_Subtract(self, op, *, left, right):
        return sge.Sub(this=left, expression=right)

    visit_DateSub = visit_DateDiff = visit_TimestampSub = visit_TimestampDiff = (
        visit_IntervalSubtract
    ) = visit_Subtract

    @parenthesize_inputs
    def visit_Multiply(self, op, *, left, right):
        return sge.Mul(this=left, expression=right)

    visit_IntervalMultiply = visit_Multiply

    @parenthesize_inputs
    def visit_Divide(self, op, *, left, right):
        return sge.Div(this=left, expression=right)

    @parenthesize_inputs
    def visit_Modulus(self, op, *, left, right):
        return sge.Mod(this=left, expression=right)

    @parenthesize_inputs
    def visit_Power(self, op, *, left, right):
        return sge.Pow(this=left, expression=right)

    @parenthesize_inputs
    def visit_GreaterEqual(self, op, *, left, right):
        return sge.GTE(this=left, expression=right)

    @parenthesize_inputs
    def visit_Greater(self, op, *, left, right):
        return sge.GT(this=left, expression=right)

    @parenthesize_inputs
    def visit_LessEqual(self, op, *, left, right):
        return sge.LTE(this=left, expression=right)

    @parenthesize_inputs
    def visit_Less(self, op, *, left, right):
        return sge.LT(this=left, expression=right)

    @parenthesize_inputs
    def visit_Equals(self, op, *, left, right):
        return sge.EQ(this=left, expression=right)

    @parenthesize_inputs
    def visit_NotEquals(self, op, *, left, right):
        return sge.NEQ(this=left, expression=right)

    @parenthesize_inputs
    def visit_And(self, op, *, left, right):
        return sge.And(this=left, expression=right)

    @parenthesize_inputs
    def visit_Or(self, op, *, left, right):
        return sge.Or(this=left, expression=right)

    @parenthesize_inputs
    def visit_Xor(self, op, *, left, right):
        return sge.Xor(this=left, expression=right)

    @parenthesize_inputs
    def visit_BitwiseLeftShift(self, op, *, left, right):
        return sge.BitwiseLeftShift(this=left, expression=right)

    @parenthesize_inputs
    def visit_BitwiseRightShift(self, op, *, left, right):
        return sge.BitwiseRightShift(this=left, expression=right)

    @parenthesize_inputs
    def visit_BitwiseAnd(self, op, *, left, right):
        return sge.BitwiseAnd(this=left, expression=right)

    @parenthesize_inputs
    def visit_BitwiseOr(self, op, *, left, right):
        return sge.BitwiseOr(this=left, expression=right)

    @parenthesize_inputs
    def visit_BitwiseXor(self, op, *, left, right):
        return sge.BitwiseXor(this=left, expression=right)

    def visit_Undefined(self, op, **_):
        raise com.OperationNotDefinedError(
            f"Compilation rule for {type(op).__name__!r} operation is not defined"
        )

    def visit_Unsupported(self, op, **_):
        raise com.UnsupportedOperationError(
            f"{type(op).__name__!r} operation is not supported in the {self.dialect} backend"
        )
