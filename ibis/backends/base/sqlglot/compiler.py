from __future__ import annotations

import abc
import calendar
import itertools
import math
import operator
import string
from collections.abc import Iterator, Mapping
from functools import partial, reduce, singledispatchmethod
from itertools import starmap
from typing import TYPE_CHECKING, Any, Callable

import sqlglot as sg
import sqlglot.expressions as sge
import toolz
from public import public

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.rewrites import Select, Window, sqlize
from ibis.expr.rewrites import (
    add_one_to_nth_value_input,
    add_order_by_to_empty_ranking_window_functions,
    empty_in_values_right_side,
    one_to_zero_index,
    replace_bucket,
    replace_scalar_parameter,
    unwrap_scalar_parameter,
)

if TYPE_CHECKING:
    import ibis.expr.schema as sch
    import ibis.expr.types as ir
    from ibis.backends.base.sqlglot.datatypes import SqlglotType


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


class FuncGen:
    __slots__ = ("namespace",)

    def __init__(self, namespace: str | None = None) -> None:
        self.namespace = namespace

    def __getattr__(self, name: str) -> partial:
        name = ".".join(filter(None, (self.namespace, name)))
        return lambda *args, **kwargs: sg.func(name, *map(sge.convert, args), **kwargs)

    def __getitem__(self, key: str) -> partial:
        return getattr(self, key)

    def array(self, *args):
        return sge.Array.from_arg_list(list(map(sge.convert, args)))

    def tuple(self, *args):
        return sg.func("tuple", *map(sge.convert, args))

    def exists(self, query):
        return sge.Exists(this=query)

    def concat(self, *args):
        return sge.Concat(expressions=list(map(sge.convert, args)))

    def map(self, keys, values):
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
    return sge.Paren(this=expr)


def parenthesize(op, arg):
    if isinstance(op, (ops.Binary, ops.Unary)):
        return paren(arg)
    # function calls don't need parens
    return arg


C = ColGen()
F = FuncGen()
NULL = sge.NULL
FALSE = sge.FALSE
TRUE = sge.TRUE
STAR = sge.Star()


@public
class SQLGlotCompiler(abc.ABC):
    __slots__ = "agg", "f", "v"

    rewrites: tuple = (
        empty_in_values_right_side,
        add_order_by_to_empty_ranking_window_functions,
        one_to_zero_index,
        add_one_to_nth_value_input,
        replace_bucket,
    )
    """A sequence of rewrites to apply to the expression tree before compilation."""

    no_limit_value: sge.Null | None = None
    """The value to use to indicate no limit."""

    quoted: bool | None = None
    """Whether to always quote identifiers."""

    NAN = sge.Literal.number("'NaN'::double")
    """Backend's NaN literal."""

    POS_INF = sge.Literal.number("'Inf'::double")
    """Backend's positive infinity literal."""

    NEG_INF = sge.Literal.number("'-Inf'::double")
    """Backend's negative infinity literal."""

    def __init__(self) -> None:
        self.agg = AggGen(aggfunc=self._aggregate)
        self.f = FuncGen()
        self.v = VarGen()

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

        gen_alias_index = itertools.count()
        quoted = self.quoted

        def fn(node, _, **kwargs):
            result = self.visit_node(node, **kwargs)

            # don't alias root nodes or value ops
            if node is op or isinstance(node, ops.Value):
                return result

            alias_index = next(gen_alias_index)
            alias = sg.to_identifier(f"t{alias_index:d}", quoted=quoted)

            try:
                return result.subquery(alias)
            except AttributeError:
                return result.as_(alias, quoted=quoted)

        # substitute parameters immediately to avoid having to define a
        # ScalarParameter translation rule
        #
        # this lets us avoid threading `params` through every `translate_val`
        # call only to be used in the one place it would be needed: the
        # ScalarParameter `translate_val` rule
        params = {
            # remove aliases from scalar parameters
            param.op().replace(unwrap_scalar_parameter): value
            for param, value in (params or {}).items()
        }

        op = op.replace(
            replace_scalar_parameter(params) | reduce(operator.or_, self.rewrites)
        )
        op = sqlize(op)
        # apply translate rules in topological order
        results = op.map(fn)
        node = results[op]
        return node.this if isinstance(node, sge.Subquery) else node

    @singledispatchmethod
    def visit_node(self, op: ops.Node, **_):
        raise com.OperationNotDefinedError(
            f"No translation rule for {type(op).__name__}"
        )

    @visit_node.register(ops.Field)
    def visit_Field(self, op, *, rel, name):
        return sg.column(
            self._gen_valid_name(name), table=rel.alias_or_name, quoted=self.quoted
        )

    @visit_node.register(ops.Cast)
    def visit_Cast(self, op, *, arg, to):
        return self.cast(arg, to)

    @visit_node.register(ops.ScalarSubquery)
    def visit_ScalarSubquery(self, op, *, rel):
        return rel.this.subquery()

    @visit_node.register(ops.Alias)
    def visit_Alias(self, op, *, arg, name):
        return arg

    @visit_node.register(ops.Literal)
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
                return self.POS_INF if value < 0 else self.NEG_INF
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

    @visit_node.register(ops.BitwiseNot)
    def visit_BitwiseNot(self, op, *, arg):
        return sge.BitwiseNot(this=arg)

    ### Mathematical Calisthenics

    @visit_node.register(ops.E)
    def visit_E(self, op):
        return self.f.exp(1)

    @visit_node.register(ops.Log)
    def visit_Log(self, op, *, arg, base):
        if base is None:
            return self.f.ln(arg)
        elif str(base) in ("2", "10"):
            return self.f[f"log{base}"](arg)
        else:
            return self.f.ln(arg) / self.f.ln(base)

    @visit_node.register(ops.Clip)
    def visit_Clip(self, op, *, arg, lower, upper):
        if upper is not None:
            arg = self.if_(arg.is_(NULL), arg, self.f.least(upper, arg))

        if lower is not None:
            arg = self.if_(arg.is_(NULL), arg, self.f.greatest(lower, arg))

        return arg

    @visit_node.register(ops.FloorDivide)
    def visit_FloorDivide(self, op, *, left, right):
        return self.cast(self.f.floor(left / right), op.dtype)

    @visit_node.register(ops.Ceil)
    @visit_node.register(ops.Floor)
    def visit_CeilFloor(self, op, *, arg):
        return self.cast(self.f[type(op).__name__.lower()](arg), op.dtype)

    @visit_node.register(ops.Round)
    def visit_Round(self, op, *, arg, digits):
        if digits is not None:
            return sge.Round(this=arg, decimals=digits)
        return sge.Round(this=arg)

    ### Dtype Dysmorphia

    @visit_node.register(ops.TryCast)
    def visit_TryCast(self, op, *, arg, to):
        return sge.TryCast(this=arg, to=self.type_mapper.from_ibis(to))

    ### Comparator Conundrums

    @visit_node.register(ops.Between)
    def visit_Between(self, op, *, arg, lower_bound, upper_bound):
        return sge.Between(this=arg, low=lower_bound, high=upper_bound)

    @visit_node.register(ops.Negate)
    def visit_Negate(self, op, *, arg):
        return -paren(arg)

    @visit_node.register(ops.Not)
    def visit_Not(self, op, *, arg):
        if isinstance(arg, sge.Filter):
            return sge.Filter(this=sg.not_(arg.this), expression=arg.expression)
        return sg.not_(paren(arg))

    ### Timey McTimeFace

    @visit_node.register(ops.Time)
    def visit_Time(self, op, *, arg):
        return self.cast(arg, to=dt.time)

    @visit_node.register(ops.TimestampNow)
    def visit_TimestampNow(self, op):
        return sge.CurrentTimestamp()

    @visit_node.register(ops.Strftime)
    def visit_Strftime(self, op, *, arg, format_str):
        return sge.TimeToStr(this=arg, format=format_str)

    @visit_node.register(ops.ExtractEpochSeconds)
    def visit_ExtractEpochSeconds(self, op, *, arg):
        return self.f.epoch(self.cast(arg, dt.timestamp))

    @visit_node.register(ops.ExtractYear)
    def visit_ExtractYear(self, op, *, arg):
        return self.f.extract(self.v.year, arg)

    @visit_node.register(ops.ExtractMonth)
    def visit_ExtractMonth(self, op, *, arg):
        return self.f.extract(self.v.month, arg)

    @visit_node.register(ops.ExtractDay)
    def visit_ExtractDay(self, op, *, arg):
        return self.f.extract(self.v.day, arg)

    @visit_node.register(ops.ExtractDayOfYear)
    def visit_ExtractDayOfYear(self, op, *, arg):
        return self.f.extract(self.v.dayofyear, arg)

    @visit_node.register(ops.ExtractQuarter)
    def visit_ExtractQuarter(self, op, *, arg):
        return self.f.extract(self.v.quarter, arg)

    @visit_node.register(ops.ExtractWeekOfYear)
    def visit_ExtractWeekOfYear(self, op, *, arg):
        return self.f.extract(self.v.week, arg)

    @visit_node.register(ops.ExtractHour)
    def visit_ExtractHour(self, op, *, arg):
        return self.f.extract(self.v.hour, arg)

    @visit_node.register(ops.ExtractMinute)
    def visit_ExtractMinute(self, op, *, arg):
        return self.f.extract(self.v.minute, arg)

    @visit_node.register(ops.ExtractSecond)
    def visit_ExtractSecond(self, op, *, arg):
        return self.f.extract(self.v.second, arg)

    @visit_node.register(ops.TimestampTruncate)
    @visit_node.register(ops.DateTruncate)
    @visit_node.register(ops.TimeTruncate)
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

    @visit_node.register(ops.DayOfWeekIndex)
    def visit_DayOfWeekIndex(self, op, *, arg):
        return (self.f.dayofweek(arg) + 6) % 7

    @visit_node.register(ops.DayOfWeekName)
    def visit_DayOfWeekName(self, op, *, arg):
        # day of week number is 0-indexed
        # Sunday == 0
        # Saturday == 6
        return sge.Case(
            this=(self.f.dayofweek(arg) + 6) % 7,
            ifs=list(itertools.starmap(self.if_, enumerate(calendar.day_name))),
        )

    @visit_node.register(ops.IntervalFromInteger)
    def visit_IntervalFromInteger(self, op, *, arg, unit):
        return sge.Interval(this=sge.convert(arg), unit=unit.singular.upper())

    ### String Instruments

    @visit_node.register(ops.Strip)
    def visit_Strip(self, op, *, arg):
        return self.f.trim(arg, string.whitespace)

    @visit_node.register(ops.RStrip)
    def visit_RStrip(self, op, *, arg):
        return self.f.rtrim(arg, string.whitespace)

    @visit_node.register(ops.LStrip)
    def visit_LStrip(self, op, *, arg):
        return self.f.ltrim(arg, string.whitespace)

    @visit_node.register(ops.Substring)
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

    @visit_node.register(ops.StringFind)
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

    @visit_node.register(ops.RegexReplace)
    def visit_RegexReplace(self, op, *, arg, pattern, replacement):
        return self.f.regexp_replace(arg, pattern, replacement, "g")

    @visit_node.register(ops.StringConcat)
    def visit_StringConcat(self, op, *, arg):
        return self.f.concat(*arg)

    @visit_node.register(ops.StringJoin)
    def visit_StringJoin(self, op, *, sep, arg):
        return self.f.concat_ws(sep, *arg)

    @visit_node.register(ops.StringSQLLike)
    def visit_StringSQLLike(self, op, *, arg, pattern, escape):
        return arg.like(pattern)

    @visit_node.register(ops.StringSQLILike)
    def visit_StringSQLILike(self, op, *, arg, pattern, escape):
        return arg.ilike(pattern)

    ### NULL PLAYER CHARACTER
    @visit_node.register(ops.IsNull)
    def visit_IsNull(self, op, *, arg):
        return arg.is_(NULL)

    @visit_node.register(ops.NotNull)
    def visit_NotNull(self, op, *, arg):
        return arg.is_(sg.not_(NULL))

    @visit_node.register(ops.InValues)
    def visit_InValues(self, op, *, value, options):
        return value.isin(*options)

    ### Counting

    @visit_node.register(ops.CountDistinct)
    def visit_CountDistinct(self, op, *, arg, where):
        return self.agg.count(sge.Distinct(expressions=[arg]), where=where)

    @visit_node.register(ops.CountDistinctStar)
    def visit_CountDistinctStar(self, op, *, arg, where):
        return self.agg.count(sge.Distinct(expressions=[STAR]), where=where)

    @visit_node.register(ops.CountStar)
    def visit_CountStar(self, op, *, arg, where):
        return self.agg.count(STAR, where=where)

    @visit_node.register(ops.Sum)
    def visit_Sum(self, op, *, arg, where):
        if op.arg.dtype.is_boolean():
            arg = self.cast(arg, dt.int32)
        return self.agg.sum(arg, where=where)

    @visit_node.register(ops.Mean)
    def visit_Mean(self, op, *, arg, where):
        if op.arg.dtype.is_boolean():
            arg = self.cast(arg, dt.int32)
        return self.agg.avg(arg, where=where)

    @visit_node.register(ops.Min)
    def visit_Min(self, op, *, arg, where):
        if op.arg.dtype.is_boolean():
            return self.agg.bool_and(arg, where=where)
        return self.agg.min(arg, where=where)

    @visit_node.register(ops.Max)
    def visit_Max(self, op, *, arg, where):
        if op.arg.dtype.is_boolean():
            return self.agg.bool_or(arg, where=where)
        return self.agg.max(arg, where=where)

    ### Stats

    @visit_node.register(ops.Quantile)
    @visit_node.register(ops.MultiQuantile)
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

    @visit_node.register(ops.Variance)
    @visit_node.register(ops.StandardDev)
    @visit_node.register(ops.Covariance)
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

    @visit_node.register(ops.Arbitrary)
    def visit_Arbitrary(self, op, *, arg, how, where):
        if how == "heavy":
            raise com.UnsupportedOperationError(
                f"how='heavy' not supported in the {self.dialect} backend"
            )
        return self.agg[how](arg, where=where)

    @visit_node.register(ops.SimpleCase)
    @visit_node.register(ops.SearchedCase)
    def visit_SimpleCase(self, op, *, base=None, cases, results, default):
        return sge.Case(
            this=base, ifs=list(map(self.if_, cases, results)), default=default
        )

    @visit_node.register(ops.ExistsSubquery)
    def visit_ExistsSubquery(self, op, *, rel):
        return self.f.exists(rel.this)

    @visit_node.register(ops.InSubquery)
    def visit_InSubquery(self, op, *, rel, needle):
        return needle.isin(rel.this)

    @visit_node.register(ops.Array)
    def visit_Array(self, op, *, exprs):
        return self.f.array(*exprs)

    @visit_node.register(ops.StructColumn)
    def visit_StructColumn(self, op, *, names, values):
        return sge.Struct.from_arg_list(
            [value.as_(name, quoted=self.quoted) for name, value in zip(names, values)]
        )

    @visit_node.register(ops.StructField)
    def visit_StructField(self, op, *, arg, field):
        return sge.Dot(this=arg, expression=sg.to_identifier(field, quoted=self.quoted))

    @visit_node.register(ops.IdenticalTo)
    def visit_IdenticalTo(self, op, *, left, right):
        return sge.NullSafeEQ(this=left, expression=right)

    @visit_node.register(ops.Greatest)
    def visit_Greatest(self, op, *, arg):
        return self.f.greatest(*arg)

    @visit_node.register(ops.Least)
    def visit_Least(self, op, *, arg):
        return self.f.least(*arg)

    @visit_node.register(ops.Coalesce)
    def visit_Coalesce(self, op, *, arg):
        return self.f.coalesce(*arg)

    ### Ordering and window functions

    @visit_node.register(ops.SortKey)
    def visit_SortKey(self, op, *, expr, ascending: bool):
        return sge.Ordered(this=expr, desc=not ascending)

    @visit_node.register(ops.ApproxMedian)
    def visit_ApproxMedian(self, op, *, arg, where):
        return self.agg.approx_quantile(arg, 0.5, where=where)

    @visit_node.register(ops.WindowBoundary)
    def visit_WindowBoundary(self, op, *, value, preceding):
        # TODO: bit of a hack to return a dict, but there's no sqlglot expression
        # that corresponds to _only_ this information
        return {"value": value, "side": "preceding" if preceding else "following"}

    @visit_node.register(Window)
    def visit_Window(self, op, *, how, func, start, end, group_by, order_by):
        if start is None:
            start = {}
        if end is None:
            end = {}

        start_value = start.get("value", "UNBOUNDED")
        start_side = start.get("side", "PRECEDING")
        end_value = end.get("value", "UNBOUNDED")
        end_side = end.get("side", "FOLLOWING")

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

    @visit_node.register(ops.Lag)
    @visit_node.register(ops.Lead)
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

    @visit_node.register(ops.Argument)
    def visit_Argument(self, op, *, name: str, shape, dtype):
        return sg.to_identifier(op.param)

    @visit_node.register(ops.RowID)
    def visit_RowID(self, op, *, table):
        return sg.column(op.name, table=table.alias_or_name, quoted=self.quoted)

    def __sql_name__(self, op: ops.ScalarUDF | ops.AggUDF) -> str:
        # not actually a table, but easier to quote individual namespace
        # components this way
        return sg.table(op.__func_name__, db=op.__udf_namespace__).sql(self.dialect)

    @visit_node.register(ops.ScalarUDF)
    def visit_ScalarUDF(self, op, **kw):
        return self.f[self.__sql_name__(op)](*kw.values())

    @visit_node.register(ops.AggUDF)
    def visit_AggUDF(self, op, *, where, **kw):
        return self.agg[self.__sql_name__(op)](*kw.values(), where=where)

    @visit_node.register(ops.TimeDelta)
    @visit_node.register(ops.DateDelta)
    @visit_node.register(ops.TimestampDelta)
    def visit_TimestampDelta(self, op, *, part, left, right):
        # dialect is necessary due to sqlglot's default behavior
        # of `part` coming last
        return sge.DateDiff(
            this=left, expression=right, unit=part, dialect=self.dialect
        )

    @visit_node.register(ops.TimestampBucket)
    def visit_TimestampBucket(self, op, *, arg, interval, offset):
        origin = self.f.cast("epoch", self.type_mapper.from_ibis(dt.timestamp))
        if offset is not None:
            origin += offset
        return self.f.time_bucket(interval, arg, origin)

    @visit_node.register(ops.ArrayConcat)
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

    @visit_node.register(Select)
    def visit_Select(self, op, *, parent, selections, predicates, sort_keys):
        # if we've constructed a useless projection return the parent relation
        if not selections and not predicates and not sort_keys:
            return parent

        result = parent

        if selections:
            result = sg.select(*starmap(self._dedup_name, selections.items())).from_(
                result
            )

        if predicates:
            result = result.where(*predicates)

        if sort_keys:
            result = result.order_by(*sort_keys)

        return result

    @visit_node.register(ops.DummyTable)
    def visit_DummyTable(self, op, *, values):
        return sg.select(*starmap(self._dedup_name, values.items()))

    @visit_node.register(ops.UnboundTable)
    def visit_UnboundTable(
        self, op, *, name: str, schema: sch.Schema, namespace: ops.Namespace
    ) -> sg.Table:
        return sg.table(
            name, db=namespace.schema, catalog=namespace.database, quoted=self.quoted
        )

    @visit_node.register(ops.InMemoryTable)
    def visit_InMemoryTable(
        self, op, *, name: str, schema: sch.Schema, data
    ) -> sg.Table:
        return sg.table(name, quoted=self.quoted)

    @visit_node.register(ops.DatabaseTable)
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

    @visit_node.register(ops.SelfReference)
    def visit_SelfReference(self, op, *, parent, identifier):
        return parent

    @visit_node.register(ops.JoinChain)
    def visit_JoinChain(self, op, *, first, rest, values):
        result = sg.select(*starmap(self._dedup_name, values.items())).from_(first)

        for link in rest:
            if isinstance(link, sge.Alias):
                link = link.this
            result = result.join(link)
        return result

    @visit_node.register(ops.JoinLink)
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
        assert predicates
        return sge.Join(
            this=table, side=sides[how], kind=kinds[how], on=sg.and_(*predicates)
        )

    @staticmethod
    def _gen_valid_name(name: str) -> str:
        return name

    @visit_node.register(ops.Project)
    def visit_Project(self, op, *, parent, values):
        # needs_alias should never be true here in explicitly, but it may get
        # passed via a (recursive) call to translate_val
        return sg.select(*starmap(self._dedup_name, values.items())).from_(parent)

    @staticmethod
    def _generate_groups(groups):
        return map(sge.convert, range(1, len(groups) + 1))

    @visit_node.register(ops.Aggregate)
    def visit_Aggregate(self, op, *, parent, groups, metrics):
        sel = sg.select(
            *starmap(
                self._dedup_name, toolz.keymap(self._gen_valid_name, groups).items()
            ),
            *starmap(
                self._dedup_name, toolz.keymap(self._gen_valid_name, metrics).items()
            ),
        ).from_(parent)

        if groups:
            sel = sel.group_by(*self._generate_groups(groups.values()))

        return sel

    def _add_parens(self, op, sg_expr):
        if type(op) in _BINARY_INFIX_OPS:
            return paren(sg_expr)
        return sg_expr

    @visit_node.register(ops.Filter)
    def visit_Filter(self, op, *, parent, predicates):
        predicates = (
            self._add_parens(raw_predicate, predicate)
            for raw_predicate, predicate in zip(op.predicates, predicates)
        )
        try:
            return parent.where(*predicates)
        except AttributeError:
            return sg.select(STAR).from_(parent).where(*predicates)

    @visit_node.register(ops.Sort)
    def visit_Sort(self, op, *, parent, keys):
        try:
            return parent.order_by(*keys)
        except AttributeError:
            return sg.select(STAR).from_(parent).order_by(*keys)

    @visit_node.register(ops.Union)
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

    @visit_node.register(ops.Intersection)
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

    @visit_node.register(ops.Difference)
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

    @visit_node.register(ops.Limit)
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

    @visit_node.register(ops.Distinct)
    def visit_Distinct(self, op, *, parent):
        return sg.select(STAR).distinct().from_(parent)

    @visit_node.register(ops.DropNa)
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

    @visit_node.register(ops.FillNa)
    def visit_FillNa(self, op, *, parent, replacements):
        if isinstance(replacements, Mapping):
            mapping = replacements
        else:
            mapping = {
                name: replacements
                for name, dtype in op.schema.items()
                if dtype.nullable
            }
        exprs = [
            (
                sg.alias(
                    sge.Coalesce(
                        this=sg.column(col, quoted=self.quoted),
                        expressions=[sge.convert(alt)],
                    ),
                    col,
                )
                if (alt := mapping.get(col)) is not None
                else sg.column(col, quoted=self.quoted)
            )
            for col in op.schema.keys()
        ]
        return sg.select(*exprs).from_(parent)

    @visit_node.register(ops.View)
    def visit_View(self, op, *, child, name: str):
        # TODO: find a way to do this without creating a temporary view
        backend = op.child.to_expr()._find_backend()
        backend._create_temp_view(table_name=name, source=sg.select(STAR).from_(child))
        return sg.table(name, quoted=self.quoted)

    @visit_node.register(ops.SQLStringView)
    def visit_SQLStringView(self, op, *, query: str, name: str, child):
        table = sg.table(name, quoted=self.quoted)
        return (
            sg.select(STAR).from_(table).with_(table, as_=query, dialect=self.dialect)
        )

    @visit_node.register(ops.SQLQueryResult)
    def visit_SQLQueryResult(self, op, *, query, schema, source):
        return sg.parse_one(query, read=self.dialect).subquery()

    @visit_node.register(ops.Unnest)
    def visit_Unnest(self, op, *, arg):
        return sge.Explode(this=arg)

    @visit_node.register(ops.JoinTable)
    def visit_JoinTable(self, op, *, parent, index):
        return parent

    @visit_node.register(ops.Value)
    def visit_Undefined(self, op, **_):
        raise com.OperationNotDefinedError(type(op).__name__)

    @visit_node.register(ops.RegexExtract)
    def visit_RegexExtract(self, op, *, arg, pattern, index):
        return self.f.regexp_extract(arg, pattern, index, dialect=self.dialect)


_SIMPLE_OPS = {
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
    ops.Capitalize: "initcap",
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

_BINARY_INFIX_OPS = {
    # Binary operations
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
    # Boolean comparisons
    ops.And: sge.And,
    ops.Or: sge.Or,
    ops.Xor: sge.Xor,
    # Bitwise business
    ops.BitwiseLeftShift: sge.BitwiseLeftShift,
    ops.BitwiseRightShift: sge.BitwiseRightShift,
    ops.BitwiseAnd: sge.BitwiseAnd,
    ops.BitwiseOr: sge.BitwiseOr,
    ops.BitwiseXor: sge.BitwiseXor,
    # Time arithmetic
    ops.DateAdd: sge.Add,
    ops.DateSub: sge.Sub,
    ops.DateDiff: sge.Sub,
    ops.TimestampAdd: sge.Add,
    ops.TimestampSub: sge.Sub,
    ops.TimestampDiff: sge.Sub,
    # Interval Marginalia
    ops.IntervalAdd: sge.Add,
    ops.IntervalMultiply: sge.Mul,
    ops.IntervalSubtract: sge.Sub,
}

for _op, _sym in _BINARY_INFIX_OPS.items():

    @SQLGlotCompiler.visit_node.register(_op)
    def _fmt(self, op, *, _sym: sge.Expression = _sym, left, right):
        return _sym(
            this=self._add_parens(op.left, left),
            expression=self._add_parens(op.right, right),
        )

    setattr(SQLGlotCompiler, f"visit_{_op.__name__}", _fmt)


del _op, _sym, _fmt


for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @SQLGlotCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        @SQLGlotCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(SQLGlotCompiler, f"visit_{_op.__name__}", _fmt)


del _op, _name, _fmt
