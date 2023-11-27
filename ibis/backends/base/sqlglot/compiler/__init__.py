from __future__ import annotations

import abc
import calendar
import functools
import itertools
import math
import operator
import string
from collections.abc import Mapping
from functools import partial, singledispatchmethod
from typing import TYPE_CHECKING, Any

import sqlglot as sg
from public import public

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot import FALSE, NULL, STAR, AggGen, FuncGen
from ibis.common.deferred import _
from ibis.common.patterns import replace
from ibis.expr.analysis import p, x

if TYPE_CHECKING:
    import ibis.expr.types as ir
    from ibis.backends.base.sqlglot.datatypes import SqlglotType


@replace(p.InValues(..., ()))
def empty_in_values_right_side(_):
    """Replace checks against an empty right side with `False`."""
    return ops.Literal(False, dtype=dt.bool)


@replace(
    p.WindowFunction(
        p.PercentRank(x) | p.RankBase(x) | p.CumeDist(x) | p.NTile(x),
        p.WindowFrame(..., order_by=()) >> _.copy(order_by=(x,)),
    )
)
def add_order_by_to_empty_ranking_window_functions(_):
    """Add an ORDER BY clause to rank window functions that don't have one."""
    return _


@replace(
    p.WindowFunction(p.RankBase | p.NTile)
    | p.StringFind
    | p.FindInSet
    | p.ArrayPosition
)
def one_to_zero_index(_, **__):
    """Subtract one from one-index functions."""
    return ops.Subtract(_, 1)


@replace(ops.NthValue)
def add_one_to_nth_value_input(_, **__):
    return _.copy(nth=ops.Add(_.nth, 1))


_INTERVAL_SUFFIXES = {
    "ms": "milliseconds",
    "us": "microseconds",
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "D": "days",
    "M": "months",
    "Y": "years",
}


@public
class SQLGlotCompiler(abc.ABC):
    __slots__ = "agg", "f"

    rewrites = (
        empty_in_values_right_side,
        add_order_by_to_empty_ranking_window_functions,
        one_to_zero_index,
        add_one_to_nth_value_input,
    )

    def __init__(self) -> None:
        self.agg = AggGen(aggfunc=self._aggregate)
        self.f = FuncGen()

    @property
    @abc.abstractmethod
    def dialect(self) -> str:
        """Backend dialect."""

    @property
    @abc.abstractmethod
    def type_mapper(self) -> SqlglotType:
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

    def if_(self, condition, true, false: sg.exp.Expression | None = None) -> sg.exp.If:
        return sg.exp.If(
            this=sg.exp.convert(condition),
            true=sg.exp.convert(true),
            false=false if false is None else sg.exp.convert(false),
        )

    def cast(self, arg, to: dt.DataType) -> sg.exp.Cast:
        return sg.cast(sg.exp.convert(arg), to=self.type_mapper.from_ibis(to))

    def translate(self, op, *, params: Mapping[ir.Value, Any]) -> sg.exp.Expression:
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

        def fn(node, _, **kwargs):
            result = self.visit_node(node, **kwargs)

            # don't alias root nodes or value ops
            if node is op or isinstance(node, ops.Value):
                return result

            alias_index = next(gen_alias_index)
            alias = f"t{alias_index:d}"

            try:
                return result.subquery(alias)
            except AttributeError:
                return sg.alias(result, alias)

        # substitute parameters immediately to avoid having to define a
        # ScalarParameter translation rule
        #
        # this lets us avoid threading `params` through every `translate_val` call
        # only to be used in the one place it would be needed: the ScalarParameter
        # `translate_val` rule
        params = {param.op(): value for param, value in params.items()}
        replace_literals = p.ScalarParameter >> (
            lambda _: ops.Literal(value=params[_], dtype=_.dtype)
        )

        op = op.replace(
            replace_literals | functools.reduce(operator.or_, self.rewrites)
        )
        # apply translate rules in topological order
        results = op.map(fn)
        node = results[op]
        return node.this if isinstance(node, sg.exp.Subquery) else node

    def _visit_node(self, op: ops.Node, **_):
        raise com.OperationNotDefinedError(
            f"No translation rule for {type(op).__name__}"
        )

    visit_node = singledispatchmethod(_visit_node)

    @visit_node.register(ops.Field)
    def visit_Field(self, op, *, rel, name, **_):
        return sg.column(name, table=rel.alias_or_name)

    @visit_node.register(ops.ScalarSubquery)
    def visit_ScalarSubquery(self, op, *, rel, **_):
        return rel.this.subquery()

    @visit_node.register(ops.Alias)
    def visit_Alias(self, op, *, arg, name, **_):
        return arg.as_(name)

    @visit_node.register(ops.Literal)
    def visit_Literal(self, op, *, value, dtype, **kw):
        if value is None:
            if dtype.nullable:
                return NULL if dtype.is_null() else self.cast(NULL, dtype)
            raise NotImplementedError(
                f"Unsupported NULL for non-nullable type: {dtype!r}"
            )
        elif dtype.is_interval():
            if dtype.unit.short == "ns":
                raise com.UnsupportedOperationError(
                    f"{self.dialect} doesn't support nanosecond interval resolutions"
                )

            return sg.exp.Interval(
                this=sg.exp.convert(str(value)), unit=dtype.resolution.upper()
            )
        elif dtype.is_boolean():
            return sg.exp.Boolean(this=value)
        elif dtype.is_string():
            return sg.exp.convert(value)
        elif dtype.is_inet() or dtype.is_macaddr():
            return sg.exp.convert(str(value))
        elif dtype.is_numeric():
            # cast non finite values to float because that's the behavior of
            # duckdb when a mixed decimal/float operation is performed
            #
            # float will be upcast to double if necessary by duckdb
            if not math.isfinite(value):
                return self.cast(
                    str(value), to=dt.float32 if dtype.is_decimal() else dtype
                )
            return self.cast(value, dtype)
        elif dtype.is_time():
            return self.f.make_time(
                value.hour, value.minute, value.second + value.microsecond / 1e6
            )
        elif dtype.is_timestamp():
            args = [
                value.year,
                value.month,
                value.day,
                value.hour,
                value.minute,
                value.second + value.microsecond / 1e6,
            ]

            if (tz := dtype.timezone) is not None:
                func = self.f.make_timestamptz
                args.append(tz)
            else:
                func = self.f.make_timestamp

            return func(*args)
        elif dtype.is_date():
            return sg.exp.DateFromParts(
                year=sg.exp.convert(value.year),
                month=sg.exp.convert(value.month),
                day=sg.exp.convert(value.day),
            )
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
                        ops.Literal(k, key_type), value=k, dtype=key_type, **kw
                    )
                    for k in value.keys()
                )
            )

            value_type = dtype.value_type
            values = self.f.array(
                *(
                    self.visit_Literal(
                        ops.Literal(v, value_type), value=v, dtype=value_type, **kw
                    )
                    for v in value.values()
                )
            )

            return sg.exp.Map(keys=keys, values=values)
        elif dtype.is_struct():
            items = [
                sg.exp.Slice(
                    this=sg.exp.convert(k),
                    expression=self.visit_Literal(
                        ops.Literal(v, field_dtype), value=v, dtype=field_dtype, **kw
                    ),
                )
                for field_dtype, (k, v) in zip(dtype.types, value.items())
            ]
            return sg.exp.Struct.from_arg_list(items)
        elif dtype.is_uuid():
            return self.cast(str(value), dtype)
        elif dtype.is_binary():
            return self.cast("".join(map("\\x{:02x}".format, value)), dtype)
        else:
            raise NotImplementedError(f"Unsupported type: {dtype!r}")

    @visit_node.register(ops.BitwiseNot)
    def visit_BitwiseNot(self, op, *, arg, **_):
        return sg.exp.BitwiseNot(this=arg)

    ### Mathematical Calisthenics

    @visit_node.register(ops.E)
    def visit_E(self, op, **_):
        return self.f.exp(1)

    @visit_node.register(ops.Log)
    def visit_Log(self, op, *, arg, base, **_):
        if base is None:
            return self.f.ln(arg)
        elif str(base) in ("2", "10"):
            return self.f[f"log{base}"](arg)
        else:
            return self.f.ln(arg) / self.f.ln(base)

    @visit_node.register(ops.Clip)
    def visit_Clip(self, op, *, arg, lower, upper, **_):
        if upper is not None:
            arg = self.if_(arg.is_(NULL), arg, self.f.least(upper, arg))

        if lower is not None:
            arg = self.if_(arg.is_(NULL), arg, self.f.greatest(lower, arg))

        return arg

    @visit_node.register(ops.FloorDivide)
    def visit_FloorDivide(self, op, *, left, right, **_):
        return self.cast(self.f.fdiv(left, right), op.dtype)

    @visit_node.register(ops.Round)
    def visit_Round(self, op, *, arg, digits, **_):
        if digits is not None:
            return sg.exp.Round(this=arg, decimals=digits)
        return sg.exp.Round(this=arg)

    ### Dtype Dysmorphia
    @visit_node.register(ops.Cast)
    def visit_Cast(self, op, *, arg, to, **_):
        if to.is_interval():
            return self.f[f"to_{_INTERVAL_SUFFIXES[to.unit.short]}"](
                sg.cast(arg, to=self.type_mapper.from_ibis(dt.int32))
            )
        elif to.is_timestamp() and op.arg.dtype.is_integer():
            return self.f.to_timestamp(arg)

        return self.cast(arg, to)

    @visit_node.register(ops.TryCast)
    def visit_TryCast(self, op, *, arg, to, **_):
        return sg.exp.TryCast(this=arg, to=self.type_mapper.from_ibis(to))

    ### Comparator Conundrums

    @visit_node.register(ops.Between)
    def visit_Between(self, op, *, arg, lower_bound, upper_bound, **_):
        return sg.exp.Between(this=arg, low=lower_bound, high=upper_bound)

    @visit_node.register(ops.Negate)
    def visit_Negate(self, op, *, arg, **_):
        return sg.exp.Neg(this=arg)

    @visit_node.register(ops.Not)
    def visit_Not(self, op, *, arg, **_):
        return sg.exp.Not(this=arg)

    ### Timey McTimeFace

    @visit_node.register(ops.Date)
    def visit_Date(self, op, *, arg, **_):
        return sg.exp.Date(this=arg)

    @visit_node.register(ops.DateFromYMD)
    def visit_DateFromYMD(self, op, *, year, month, day, **_):
        return sg.exp.DateFromParts(year=year, month=month, day=day)

    @visit_node.register(ops.Time)
    def visit_Time(self, op, *, arg, **_):
        return self.cast(arg, to=dt.time)

    @visit_node.register(ops.TimestampNow)
    def visit_TimestampNow(self, op, **_):
        """DuckDB current timestamp defaults to timestamp + tz."""
        return self.cast(sg.exp.CurrentTimestamp(), dt.timestamp)

    @visit_node.register(ops.TimestampFromUNIX)
    def visit_TimestampFromUNIX(self, op, *, arg, unit, **_):
        unit = unit.short
        if unit == "ms":
            return self.f.epoch_ms(arg)
        elif unit == "s":
            return sg.exp.UnixToTime(this=arg)
        else:
            raise com.UnsupportedOperationError(f"{unit!r} unit is not supported!")

    @visit_node.register(ops.TimestampFromYMDHMS)
    def visit_TimestampFromYMDHMS(
        self, op, *, year, month, day, hours, minutes, seconds, **_
    ):
        args = [year, month, day, hours, minutes, seconds]

        func = "make_timestamp"
        if (timezone := op.dtype.timezone) is not None:
            func += "tz"
            args.append(timezone)

        return self.f[func](*args)

    @visit_node.register(ops.Strftime)
    def visit_Strftime(self, op, *, arg, format_str, **_):
        if not isinstance(op.format_str, ops.Literal):
            raise com.UnsupportedOperationError(
                f"{self.dialect} `format_str` must be a literal `str`; got {type(op.format_str)}"
            )
        return self.f.strftime(arg, format_str)

    @visit_node.register(ops.ExtractEpochSeconds)
    def visit_ExtractEpochSeconds(self, op, *, arg, **_):
        return self.f.epoch(self.cast(arg, dt.timestamp))

    @visit_node.register(ops.ExtractYear)
    def visit_ExtractYear(self, op, *, arg, **_):
        return self.f.extract("year", arg)

    @visit_node.register(ops.ExtractMonth)
    def visit_ExtractMonth(self, op, *, arg, **_):
        return self.f.extract("month", arg)

    @visit_node.register(ops.ExtractDay)
    def visit_ExtractDay(self, op, *, arg, **_):
        return self.f.extract("day", arg)

    @visit_node.register(ops.ExtractDayOfYear)
    def visit_ExtractDayOfYear(self, op, *, arg, **_):
        return self.f.extract("dayofyear", arg)

    @visit_node.register(ops.ExtractQuarter)
    def visit_ExtractQuarter(self, op, *, arg, **_):
        return self.f.extract("quarter", arg)

    @visit_node.register(ops.ExtractWeekOfYear)
    def visit_ExtractWeekOfYear(self, op, *, arg, **_):
        return self.f.extract("week", arg)

    @visit_node.register(ops.ExtractHour)
    def visit_ExtractHour(self, op, *, arg, **_):
        return self.f.extract("hour", arg)

    @visit_node.register(ops.ExtractMinute)
    def visit_ExtractMinute(self, op, *, arg, **_):
        return self.f.extract("minute", arg)

    @visit_node.register(ops.ExtractSecond)
    def visit_ExtractSecond(self, op, *, arg, **_):
        return self.f.extract("second", arg)

    @visit_node.register(ops.ExtractMillisecond)
    def visit_ExtractMillisecond(self, op, *, arg, **_):
        return self.f.mod(self.f.extract("ms", arg), 1_000)

    # DuckDB extracts subminute microseconds and milliseconds
    # so we have to finesse it a little bit
    @visit_node.register(ops.ExtractMicrosecond)
    def visit_ExtractMicrosecond(self, op, *, arg, **_):
        return self.f.mod(self.f.extract("us", arg), 1_000_000)

    @visit_node.register(ops.TimestampTruncate)
    @visit_node.register(ops.DateTruncate)
    @visit_node.register(ops.TimeTruncate)
    def visit_TimestampTruncate(self, op, *, arg, unit, **_):
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

        unit = unit.short
        if (duckunit := unit_mapping.get(unit)) is None:
            raise com.UnsupportedOperationError(f"Unsupported truncate unit {unit}")

        return self.f.date_trunc(duckunit, arg)

    @visit_node.register(ops.DayOfWeekIndex)
    def visit_DayOfWeekIndex(self, op, *, arg, **_):
        return (self.f.dayofweek(arg) + 6) % 7

    @visit_node.register(ops.DayOfWeekName)
    def visit_DayOfWeekName(self, op, *, arg, **_):
        # day of week number is 0-indexed
        # Sunday == 0
        # Saturday == 6
        return sg.exp.Case(
            this=(self.f.dayofweek(arg) + 6) % 7,
            ifs=list(itertools.starmap(self.if_, enumerate(calendar.day_name))),
        )

    @visit_node.register(ops.IntervalFromInteger)
    def visit_IntervalFromInteger(self, op, *, arg, **_):
        dtype = op.dtype
        if dtype.unit.short == "ns":
            raise com.UnsupportedOperationError(
                f"{self.dialect} doesn't support nanosecond interval resolutions"
            )

        if op.dtype.resolution == "week":
            return self.f.to_days(arg * 7)
        return self.f[f"to_{op.dtype.resolution}s"](arg)

    ### String Instruments

    @visit_node.register(ops.Strip)
    def visit_Strip(self, op, *, arg, **_):
        return self.f.trim(arg, string.whitespace)

    @visit_node.register(ops.RStrip)
    def visit_RStrip(self, op, *, arg, **_):
        return self.f.rtrim(arg, string.whitespace)

    @visit_node.register(ops.LStrip)
    def visit_LStrip(self, op, *, arg, **_):
        return self.f.ltrim(arg, string.whitespace)

    @visit_node.register(ops.Substring)
    def visit_Substring(self, op, *, arg, start, length, **_):
        if_pos = sg.exp.Substring(this=arg, start=start + 1, length=length)
        if_neg = sg.exp.Substring(this=arg, start=start, length=length)

        return self.if_(start >= 0, if_pos, if_neg)

    @visit_node.register(ops.StringFind)
    def visit_StringFind(self, op, *, arg, substr, start, end, **_):
        if end is not None:
            raise com.UnsupportedOperationError(
                "String find doesn't support `end` argument"
            )

        if start is not None:
            arg = self.f.substr(arg, start + 1)
            pos = self.f.strpos(arg, substr)
            return self.if_(pos > 0, pos + start, 0)

        return self.f.strpos(arg, substr)

    @visit_node.register(ops.RegexSearch)
    def visit_RegexSearch(self, op, *, arg, pattern, **_):
        return self.f.regexp_matches(arg, pattern, "s")

    @visit_node.register(ops.RegexReplace)
    def visit_RegexReplace(self, op, *, arg, pattern, replacement, **_):
        return self.f.regexp_replace(arg, pattern, replacement, "g")

    @visit_node.register(ops.RegexExtract)
    def visit_RegexExtract(self, op, *, arg, pattern, index, **_):
        return self.f.regexp_extract(arg, pattern, index, dialect=self.dialect)

    @visit_node.register(ops.StringSplit)
    def visit_StringSplit(self, op, *, arg, delimiter, **_):
        return sg.exp.Split(this=arg, expression=delimiter)

    @visit_node.register(ops.StringJoin)
    def visit_StringJoin(self, op, *, arg, sep, **_):
        return self.f.list_aggr(self.f.array(*arg), "string_agg", sep)

    @visit_node.register(ops.StringConcat)
    def visit_StringConcat(self, op, *, arg, **_):
        return sg.exp.Concat.from_arg_list(list(arg))

    @visit_node.register(ops.StringSQLLike)
    def visit_StringSQLLike(self, op, *, arg, pattern, **_):
        return arg.like(pattern)

    @visit_node.register(ops.StringSQLILike)
    def visit_StringSQLILike(self, op, *, arg, pattern, **_):
        return arg.ilike(pattern)

    @visit_node.register(ops.Capitalize)
    def visit_Capitalize(self, op, *, arg, **_):
        return sg.exp.Concat(
            expressions=[
                self.f.upper(self.f.substr(arg, 1, 1)),
                self.f.lower(self.f.substr(arg, 2)),
            ]
        )

    ### NULL PLAYER CHARACTER
    @visit_node.register(ops.IsNull)
    def visit_IsNull(self, op, *, arg, **_):
        return arg.is_(NULL)

    @visit_node.register(ops.NotNull)
    def visit_NotNull(self, op, *, arg, **_):
        return arg.is_(sg.not_(NULL))

    ### Definitely Not Tensors

    @visit_node.register(ops.ArrayDistinct)
    def visit_ArrayDistinct(self, op, *, arg, **_):
        return self.if_(
            arg.is_(NULL),
            NULL,
            self.f.list_distinct(arg)
            + self.if_(
                self.f.list_count(arg) < self.f.len(arg),
                self.f.array(NULL),
                self.f.array(),
            ),
        )

    @visit_node.register(ops.ArrayIndex)
    def visit_ArrayIndex(self, op, *, arg, index, **_):
        return self.f.list_extract(arg, index + self.cast(index >= 0, op.index.dtype))

    @visit_node.register(ops.InValues)
    def visit_InValues(self, op, *, value, options, **_):
        return value.isin(*options)

    @visit_node.register(ops.ArrayConcat)
    def visit_ArrayConcat(self, op, *, arg, **_):
        return functools.reduce(self.f.list_concat, arg)

    @visit_node.register(ops.ArrayRepeat)
    def visit_ArrayRepeat(self, op, *, arg, times, **_):
        func = sg.exp.Lambda(this=arg, expressions=[sg.to_identifier("_")])
        return self.f.flatten(self.f.list_apply(self.f.range(times), func))

    def _neg_idx_to_pos(self, array, idx):
        arg_length = self.f.len(array)
        return self.if_(
            idx >= 0,
            idx,
            # Need to have the greatest here to handle the case where
            # abs(neg_index) > arg_length
            # e.g. where the magnitude of the negative index is greater than the
            # length of the array
            # You cannot index a[:-3] if a = [1, 2]
            arg_length + self.f.greatest(idx, -arg_length),
        )

    @visit_node.register(ops.ArraySlice)
    def visit_ArraySlice(self, op, *, arg, start, stop, **_):
        arg_length = self.f.len(arg)

        if start is None:
            start = 0
        else:
            start = self.f.least(arg_length, self._neg_idx_to_pos(arg, start))

        if stop is None:
            stop = arg_length
        else:
            stop = self._neg_idx_to_pos(arg, stop)

        return self.f.list_slice(arg, start + 1, stop)

    @visit_node.register(ops.ArrayStringJoin)
    def visit_ArrayStringJoin(self, op, *, sep, arg, **_):
        return self.f.array_to_string(arg, sep)

    @visit_node.register(ops.ArrayMap)
    def visit_ArrayMap(self, op, *, arg, body, param, **_):
        lamduh = sg.exp.Lambda(this=body, expressions=[sg.to_identifier(param)])
        return self.f.list_apply(arg, lamduh)

    @visit_node.register(ops.ArrayFilter)
    def visit_ArrayFilter(self, op, *, arg, body, param, **_):
        lamduh = sg.exp.Lambda(this=body, expressions=[sg.to_identifier(param)])
        return self.f.list_filter(arg, lamduh)

    @visit_node.register(ops.ArrayIntersect)
    def visit_ArrayIntersect(self, op, *, left, right, **_):
        param = sg.to_identifier("x")
        body = self.f.list_contains(right, param)
        lamduh = sg.exp.Lambda(this=body, expressions=[param])
        return self.f.list_filter(left, lamduh)

    @visit_node.register(ops.ArrayRemove)
    def visit_ArrayRemove(self, op, *, arg, other, **_):
        param = sg.to_identifier("x")
        body = param.neq(other)
        lamduh = sg.exp.Lambda(this=body, expressions=[param])
        return self.f.list_filter(arg, lamduh)

    @visit_node.register(ops.ArrayUnion)
    def visit_ArrayUnion(self, op, *, left, right, **_):
        arg = self.f.list_concat(left, right)
        return self.if_(
            arg.is_(NULL),
            NULL,
            self.f.list_distinct(arg)
            + self.if_(
                self.f.list_count(arg) < self.f.len(arg),
                self.f.array(NULL),
                self.f.array(),
            ),
        )

    @visit_node.register(ops.ArrayZip)
    def visit_ArrayZip(self, op, *, arg, **_):
        i = sg.to_identifier("i")
        body = sg.exp.Struct.from_arg_list(
            [
                sg.exp.Slice(this=k, expression=v[i])
                for k, v in zip(map(sg.exp.convert, op.dtype.value_type.names), arg)
            ]
        )
        func = sg.exp.Lambda(this=body, expressions=[i])
        return self.f.list_apply(
            self.f.range(
                1,
                # DuckDB Range excludes upper bound
                self.f.greatest(*map(self.f.len, arg)) + 1,
            ),
            func,
        )

    ### Counting

    @visit_node.register(ops.CountDistinct)
    def visit_CountDistinct(self, op, *, arg, where, **_):
        return self.agg.count(sg.exp.Distinct(expressions=[arg]), where=where)

    @visit_node.register(ops.CountDistinctStar)
    def visit_CountDistinctStar(self, op, *, where, **_):
        # use a tuple because duckdb doesn't accept COUNT(DISTINCT a, b, c, ...)
        #
        # this turns the expression into COUNT(DISTINCT (a, b, c, ...))
        row = sg.exp.Tuple(expressions=list(map(sg.column, op.arg.schema.keys())))
        return self.agg.count(sg.exp.Distinct(expressions=[row]), where=where)

    @visit_node.register(ops.CountStar)
    def visit_CountStar(self, op, *, where, **_):
        return self.agg.count(STAR, where=where)

    @visit_node.register(ops.Sum)
    def visit_Sum(self, op, *, arg, where, **_):
        arg = self.cast(arg, op.dtype) if op.arg.dtype.is_boolean() else arg
        return self.agg.sum(arg, where=where)

    @visit_node.register(ops.NthValue)
    def visit_NthValue(self, op, *, arg, nth, **_):
        return self.f.nth_value(arg, nth)

    ### Stats

    @visit_node.register(ops.Quantile)
    @visit_node.register(ops.MultiQuantile)
    def visit_Quantile(self, op, *, arg, quantile, where, **_):
        return self.agg.quantile_cont(arg, quantile, where=where)

    @visit_node.register(ops.Correlation)
    def visit_Correlation(self, op, *, left, right, how, where, **_):
        if how == "sample":
            raise com.UnsupportedOperationError(
                f"{self.dialect} only implements `pop` correlation coefficient"
            )

        # TODO: rewrite rule?
        if (left_type := op.left.dtype).is_boolean():
            left = self.cast(left, dt.Int32(nullable=left_type.nullable))

        if (right_type := op.right.dtype).is_boolean():
            right = self.cast(right, dt.Int32(nullable=right_type.nullable))

        return self.agg.corr(left, right, where=where)

    @visit_node.register(ops.Covariance)
    def visit_Covariance(self, op, *, left, right, how, where, **_):
        hows = {"sample": "samp", "pop": "pop"}

        # TODO: rewrite rule?
        if (left_type := op.left.dtype).is_boolean():
            left = self.cast(left, dt.Int32(nullable=left_type.nullable))

        if (right_type := op.right.dtype).is_boolean():
            right = self.cast(right, dt.Int32(nullable=right_type.nullable))

        funcname = f"covar_{hows[how]}"
        return self.agg[funcname](left, right, where=where)

    @visit_node.register(ops.Variance)
    @visit_node.register(ops.StandardDev)
    def visit_Variance(self, op, *, arg, how, where, **_):
        hows = {"sample": "samp", "pop": "pop"}
        funcs = {ops.Variance: "var", ops.StandardDev: "stddev"}

        if (arg_dtype := op.arg.dtype).is_boolean():
            arg = self.cast(arg, dt.Int32(nullable=arg_dtype.nullable))

        funcname = f"{funcs[type(op)]}_{hows[how]}"
        return self.agg[funcname](arg, where=where)

    @visit_node.register(ops.Arbitrary)
    def visit_Arbitrary(self, op, *, arg, how, where, **_):
        if how == "heavy":
            raise com.UnsupportedOperationError(
                "how='heavy' not supported in the backend"
            )
        return self.agg[how](arg, where=where)

    @visit_node.register(ops.FindInSet)
    def visit_FindInSet(self, op, *, needle, values, **_):
        return self.f.list_indexof(self.f.array(*values), needle)

    @visit_node.register(ops.SimpleCase)
    @visit_node.register(ops.SearchedCase)
    def visit_SimpleCase(self, op, *, base=None, cases, results, default, **_):
        return sg.exp.Case(
            this=base, ifs=list(map(self.if_, cases, results)), default=default
        )

    @visit_node.register(ops.TableArrayView)
    def visit_TableArrayView(self, op, *, table, **_):
        return table.args["this"].subquery()

    @visit_node.register(ops.ExistsSubquery)
    def visit_ExistsSubquery(self, op, *, rel, **_):
        return self.f.exists(rel.this.subquery())

    @visit_node.register(ops.InSubquery)
    def visit_InSubquery(self, op, *, rel, needle, **_):
        return needle.isin(rel.this.subquery())

    @visit_node.register(ops.ArrayColumn)
    def visit_ArrayColumn(self, op, *, cols, **_):
        return self.f.array(*cols)

    @visit_node.register(ops.StructColumn)
    def visit_StructColumn(self, op, *, names, values, **_):
        return sg.exp.Struct.from_arg_list(
            [
                sg.exp.Slice(this=sg.exp.convert(name), expression=value)
                for name, value in zip(names, values)
            ]
        )

    @visit_node.register(ops.StructField)
    def visit_StructField(self, op, *, arg, field, **_):
        val = arg.this if isinstance(op.arg, ops.Alias) else arg
        return val[sg.exp.convert(field)]

    @visit_node.register(ops.IdenticalTo)
    def visit_IdenticalTo(self, op, *, left, right, **_):
        return sg.exp.NullSafeEQ(this=left, expression=right)

    @visit_node.register(ops.Greatest)
    def visit_Greatest(self, op, *, arg, **_):
        return self.f.greatest(*arg)

    @visit_node.register(ops.Least)
    def visit_Least(self, op, *, arg, **_):
        return self.f.least(*arg)

    @visit_node.register(ops.Coalesce)
    def visit_Coalesce(self, op, *, arg, **_):
        return self.f.coalesce(*arg)

    @visit_node.register(ops.MapGet)
    def visit_MapGet(self, op, *, arg, key, default, **_):
        return self.f.ifnull(
            self.f.list_extract(self.f.element_at(arg, key), 1), default
        )

    @visit_node.register(ops.MapContains)
    def visit_MapContains(self, op, *, arg, key, **_):
        return self.f.len(self.f.element_at(arg, key)).neq(0)

    ### Ordering and window functions

    @visit_node.register(ops.RowNumber)
    def visit_RowNumber(self, op, **_):
        return sg.exp.RowNumber()

    @visit_node.register(ops.DenseRank)
    def visit_DenseRank(self, op, **_):
        return self.f.dense_rank()

    @visit_node.register(ops.MinRank)
    def visit_MinRank(self, op, **_):
        return self.f.rank()

    @visit_node.register(ops.PercentRank)
    def visit_PercentRank(self, op, **_):
        return self.f.percent_rank()

    @visit_node.register(ops.CumeDist)
    def visit_CumeDist(self, op, **_):
        return self.f.cume_dist()

    @visit_node.register(ops.SortKey)
    def visit_SortKey(self, op, *, expr, ascending: bool, **_):
        return sg.exp.Ordered(this=expr, desc=not ascending)

    @visit_node.register(ops.ApproxMedian)
    def visit_ApproxMedian(self, op, *, arg, where, **_):
        return self.agg.approx_quantile(arg, 0.5, where=where)

    @visit_node.register(ops.WindowBoundary)
    def visit_WindowBoundary(self, op, *, value, preceding, **_):
        # TODO: bit of a hack to return a dict, but there's no sqlglot expression
        # that corresponds to _only_ this information
        return {"value": value, "side": "preceding" if preceding else "following"}

    @visit_node.register(ops.WindowFrame)
    def visit_WindowFrame(self, op, *, group_by, order_by, start, end, **_):
        if start is None:
            start = {}

        start_value = start.get("value", "UNBOUNDED")
        start_side = start.get("side", "PRECEDING")

        if end is None:
            end = {}

        end_value = end.get("value", "UNBOUNDED")
        end_side = end.get("side", "FOLLOWING")

        spec = sg.exp.WindowSpec(
            kind=op.how.upper(),
            start=start_value,
            start_side=start_side,
            end=end_value,
            end_side=end_side,
            over="OVER",
        )

        order = sg.exp.Order(expressions=order_by) if order_by else None

        # TODO: bit of a hack to return a partial, but similar to `WindowBoundary`
        # there's no sqlglot expression that corresponds to _only_ this information
        return partial(sg.exp.Window, partition_by=group_by, order=order, spec=spec)

    @visit_node.register(ops.WindowFunction)
    def visit_WindowFunction(self, op, *, func, frame, **_: Any):
        return frame(this=func)

    @visit_node.register(ops.Lag)
    @visit_node.register(ops.Lead)
    def visit_LagLead(self, op, *, arg, offset, default, **_):
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
    def visit_Argument(self, op, **_):
        return sg.to_identifier(op.name)

    @visit_node.register(ops.RowID)
    def visit_RowID(self, op, *, table, **_):
        return sg.column(op.name, table=table.alias_or_name)

    @visit_node.register(ops.ScalarUDF)
    def visit_ScalarUDF(self, op, **kw):
        return self.f[op.__full_name__](*kw.values())

    @visit_node.register(ops.AggUDF)
    def visit_AggUDF(self, op, *, where, **kw):
        return self.agg[op.__full_name__](*kw.values(), where=where)

    @visit_node.register(ops.ToJSONMap)
    @visit_node.register(ops.ToJSONArray)
    def visit_ToJSONMap(self, op, *, arg, **_):
        return self.f.try_cast(arg, self.type_mapper.from_ibis(op.dtype))

    @visit_node.register(ops.TimestampDelta)
    @visit_node.register(ops.DateDelta)
    @visit_node.register(ops.TimeDelta)
    def visit_Delta(self, op, *, part, left, right, **_):
        # dialect is necessary due to sqlglot's default behavior
        # of `part` coming last
        return self.f.date_diff(part, right, left, dialect=self.dialect)

    @visit_node.register(ops.TimestampBucket)
    def visit_TimestampBucket(self, op, *, arg, interval, offset, **_):
        origin = self.f.cast("epoch", self.type_mapper.from_ibis(dt.timestamp))
        if offset is not None:
            origin += offset
        return self.f.time_bucket(interval, arg, origin)

    ## relations

    @visit_node.register(ops.DummyTable)
    def visit_DummyTable(self, op, *, values, **_):
        return sg.select(*(value.as_(key) for key, value in values.items()))

    @visit_node.register(ops.UnboundTable)
    def visit_UnboundTable(self, op, **_):
        return sg.table(op.name, quoted=True)

    @visit_node.register(ops.InMemoryTable)
    def visit_InMemoryTable(self, op, **_):
        return sg.table(op.name)

    @visit_node.register(ops.DatabaseTable)
    def visit_DatabaseTable(self, op, *, name, namespace, **_):
        return sg.table(name, db=namespace.schema, catalog=namespace.database)

    @visit_node.register(ops.SelfReference)
    def visit_SelfReference(self, op, *, parent, **_):
        return parent.as_(op.name)

    @visit_node.register(ops.JoinChain)
    def visit_JoinChain(self, op, *, first, rest, fields, **_):
        result = sg.select(*(value.as_(key) for key, value in fields.items())).from_(
            first
        )

        for link in rest:
            if isinstance(link, sg.exp.Alias):
                link = link.this
            result = result.join(link)
        return result

    @visit_node.register(ops.JoinLink)
    def visit_JoinLink(self, op, *, how, table, predicates, **_):
        sides = {
            "inner": None,
            "left": "left",
            "right": "right",
            "semi": "left",
            "anti": "left",
            "cross": None,
            "outer": "full",
        }
        kinds = {
            "inner": "inner",
            "left": "outer",
            "right": "outer",
            "semi": "semi",
            "anti": "anti",
            "cross": "cross",
            "outer": "outer",
        }
        res = sg.exp.Join(
            this=table, side=sides[how], kind=kinds[how], on=sg.and_(*predicates)
        )
        return res

    @visit_node.register(ops.Project)
    def visit_Project(self, op, *, parent, values, **_):
        # needs_alias should never be true here in explicitly, but it may get
        # passed via a (recursive) call to translate_val
        return sg.select(*(value.as_(key) for key, value in values.items())).from_(
            parent
        )

    @visit_node.register(ops.Aggregate)
    def visit_Aggregate(self, op, *, parent, groups, metrics, **_):
        sel = sg.select(
            *(value.as_(key) for key, value in groups.items()),
            *(value.as_(key) for key, value in metrics.items()),
        ).from_(parent)

        if groups:
            sel = sel.group_by(*map(sg.exp.convert, range(1, len(groups) + 1)))

        return sel

    def _add_parens(self, op, sg_expr):
        if type(op) in _BINARY_INFIX_OPS:
            return sg.exp.Paren(this=sg_expr)
        return sg_expr

    @visit_node.register(ops.Filter)
    def visit_Filter(self, op, *, parent, predicates, **_):
        predicates = (
            self._add_parens(raw_predicate, predicate)
            for raw_predicate, predicate in zip(op.predicates, predicates)
        )
        try:
            return parent.where(*predicates)
        except AttributeError:
            return sg.select(STAR).from_(parent).where(*predicates)

    @visit_node.register(ops.Sort)
    def visit_Sort(self, op, *, parent, keys, **_):
        try:
            return parent.order_by(*keys)
        except AttributeError:
            return sg.select(STAR).from_(parent).order_by(*keys)

    @visit_node.register(ops.Union)
    def visit_Union(self, op, *, left, right, distinct, **_):
        if isinstance(left, sg.exp.Table):
            left = sg.select(STAR).from_(left)

        if isinstance(right, sg.exp.Table):
            right = sg.select(STAR).from_(right)

        return sg.union(
            left.args.get("this", left),
            right.args.get("this", right),
            distinct=distinct,
        )

    @visit_node.register(ops.Intersection)
    def visit_Intersection(self, op, *, left, right, distinct, **_):
        if isinstance(left, sg.exp.Table):
            left = sg.select(STAR).from_(left)

        if isinstance(right, sg.exp.Table):
            right = sg.select(STAR).from_(right)

        return sg.intersect(
            left.args.get("this", left),
            right.args.get("this", right),
            distinct=distinct,
        )

    @visit_node.register(ops.Difference)
    def visit_Difference(self, op, *, left, right, distinct, **_):
        if isinstance(left, sg.exp.Table):
            left = sg.select(STAR).from_(left)

        if isinstance(right, sg.exp.Table):
            right = sg.select(STAR).from_(right)

        return sg.except_(
            left.args.get("this", left),
            right.args.get("this", right),
            distinct=distinct,
        )

    @visit_node.register(ops.Limit)
    def visit_Limit(self, op, *, parent, n, offset, **_):
        result = sg.select(STAR).from_(parent)

        if isinstance(n, int):
            result = result.limit(n)
        elif n is not None:
            limit = n
            # TODO: calling `.sql` is a workaround for sqlglot not supporting
            # scalar subqueries in limits
            limit = sg.select(limit).from_(parent).subquery()
            result = result.limit(limit)

        assert offset is not None, "offset is None"

        if not isinstance(offset, int):
            skip = offset
            skip = sg.select(skip).from_(parent).subquery()
        elif not offset:
            return result
        else:
            skip = offset

        return result.offset(skip)

    @visit_node.register(ops.Distinct)
    def visit_Distinct(self, op, *, parent, **_):
        return sg.select(STAR).distinct().from_(parent)

    @visit_node.register(ops.DropNa)
    def visit_DropNa(self, op, *, parent, how, subset, **_):
        if subset is None:
            subset = [
                sg.column(name, table=parent.alias_or_name) for name in op.schema.names
            ]

        if subset:
            predicate = functools.reduce(
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
    def visit_FillNa(self, op, *, parent, replacements, **_):
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
                    sg.exp.Coalesce(
                        this=sg.column(col), expressions=[sg.exp.convert(alt)]
                    ),
                    col,
                )
                if (alt := mapping.get(col)) is not None
                else sg.column(col)
            )
            for col in op.schema.keys()
        ]
        return sg.select(*exprs).from_(parent)

    @visit_node.register(ops.View)
    def visit_View(self, op, *, child, name: str, **_):
        # TODO: find a way to do this without creating a temporary view
        backend = op.child.to_expr()._find_backend()
        backend._create_temp_view(table_name=name, source=sg.select(STAR).from_(child))
        return sg.table(name)

    @visit_node.register(ops.SQLStringView)
    def visit_SQLStringView(self, op, *, query: str, **_):
        table = sg.table(op.name)
        return (
            sg.select(STAR).from_(table).with_(table, as_=query, dialect=self.dialect)
        )

    @visit_node.register(ops.SQLQueryResult)
    def visit_SQLQueryResult(self, op, *, query, **_):
        return sg.parse_one(query, read=self.dialect).subquery()


### Simple Ops

_SIMPLE_OPS = {
    ops.Power: "pow",
    # Unary operations
    ops.IsNan: "isnan",
    ops.IsInf: "isinf",
    ops.Abs: "abs",
    ops.Ceil: "ceil",
    ops.Floor: "floor",
    ops.Exp: "exp",
    ops.Sqrt: "sqrt",
    ops.Ln: "ln",
    ops.Log2: "log2",
    ops.Log10: "log",
    ops.Acos: "acos",
    ops.Asin: "asin",
    ops.Atan: "atan",
    ops.Atan2: "atan2",
    ops.Cos: "cos",
    ops.Sin: "sin",
    ops.Tan: "tan",
    ops.Cot: "cot",
    ops.Pi: "pi",
    ops.RandomScalar: "random",
    ops.Sign: "sign",
    # Unary aggregates
    ops.ApproxCountDistinct: "approx_count_distinct",
    ops.Median: "median",
    ops.Mean: "avg",
    ops.Max: "max",
    ops.Min: "min",
    ops.ArgMin: "arg_min",
    ops.Mode: "mode",
    ops.ArgMax: "arg_max",
    ops.First: "first",
    ops.Last: "last",
    ops.Count: "count",
    ops.All: "bool_and",
    ops.Any: "bool_or",
    ops.ArrayCollect: "list",
    ops.GroupConcat: "string_agg",
    ops.BitOr: "bit_or",
    ops.BitAnd: "bit_and",
    ops.BitXor: "bit_xor",
    # string operations
    ops.StringContains: "contains",
    ops.StringLength: "length",
    ops.Lowercase: "lower",
    ops.Uppercase: "upper",
    ops.Reverse: "reverse",
    ops.StringReplace: "replace",
    ops.StartsWith: "prefix",
    ops.EndsWith: "suffix",
    ops.LPad: "lpad",
    ops.RPad: "rpad",
    ops.StringAscii: "ascii",
    ops.StrRight: "right",
    # Other operations
    ops.IfElse: "if",
    ops.ArrayLength: "length",
    ops.Unnest: "unnest",
    ops.Degrees: "degrees",
    ops.Radians: "radians",
    ops.NullIf: "nullif",
    ops.MapLength: "cardinality",
    ops.MapKeys: "map_keys",
    ops.MapValues: "map_values",
    ops.ArraySort: "list_sort",
    ops.ArrayContains: "list_contains",
    ops.FirstValue: "first_value",
    ops.LastValue: "last_value",
    ops.NTile: "ntile",
    ops.Hash: "hash",
    ops.TimeFromHMS: "make_time",
    ops.StringToTimestamp: "strptime",
    ops.Levenshtein: "levenshtein",
    ops.Repeat: "repeat",
    ops.Map: "map",
    ops.MapMerge: "map_concat",
    ops.JSONGetItem: "json_extract",
    ops.TypeOf: "typeof",
    ops.IntegerRange: "range",
    ops.ArrayFlatten: "flatten",
    ops.ArrayPosition: "list_indexof",
}

_BINARY_INFIX_OPS = {
    # Binary operations
    ops.Add: sg.exp.Add,
    ops.Subtract: sg.exp.Sub,
    ops.Multiply: sg.exp.Mul,
    ops.Divide: sg.exp.Div,
    ops.Modulus: sg.exp.Mod,
    # Comparisons
    ops.GreaterEqual: sg.exp.GTE,
    ops.Greater: sg.exp.GT,
    ops.LessEqual: sg.exp.LTE,
    ops.Less: sg.exp.LT,
    ops.Equals: sg.exp.EQ,
    ops.NotEquals: sg.exp.NEQ,
    # Boolean comparisons
    ops.And: sg.exp.And,
    ops.Or: sg.exp.Or,
    ops.Xor: sg.exp.Xor,
    # Bitwise business
    ops.BitwiseLeftShift: sg.exp.BitwiseLeftShift,
    ops.BitwiseRightShift: sg.exp.BitwiseRightShift,
    ops.BitwiseAnd: sg.exp.BitwiseAnd,
    ops.BitwiseOr: sg.exp.BitwiseOr,
    ops.BitwiseXor: sg.exp.BitwiseXor,
    # Time arithmetic
    ops.DateAdd: sg.exp.Add,
    ops.DateSub: sg.exp.Sub,
    ops.DateDiff: sg.exp.Sub,
    ops.TimestampAdd: sg.exp.Add,
    ops.TimestampSub: sg.exp.Sub,
    ops.TimestampDiff: sg.exp.Sub,
    # Interval Marginalia
    ops.IntervalAdd: sg.exp.Add,
    ops.IntervalMultiply: sg.exp.Mul,
    ops.IntervalSubtract: sg.exp.Sub,
}

for _op, _sym in _BINARY_INFIX_OPS.items():

    @SQLGlotCompiler.visit_node.register(_op)
    def _fmt(self, op, *, _sym: sg.exp.Expression = _sym, left, right, **_):
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
