from __future__ import annotations

import decimal
import json
import math
import operator

try:
    import regex as re
except ImportError:
    import re
from functools import reduce
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
import toolz

import ibis.expr.operations as ops
from ibis.backends.pandas.helpers import isnull


def substring_rowwise(row):
    arg, start, length = row["arg"], row["start"], row["length"]
    if isnull(arg):
        return None
    elif isnull(start):
        return None
    elif isnull(length):
        return arg[start:]
    else:
        return arg[start : start + length]


def substring_serieswise(arg, start, length):
    if length is None:
        return arg.str[start:]
    else:
        return arg.str[start : start + length]


def _sql_like_to_regex(pattern, escape):
    """Convert a SQL `LIKE` pattern to an equivalent Python regular expression.

    Parameters
    ----------
    pattern
        A LIKE pattern with the following semantics:
        * `%` matches zero or more characters
        * `_` matches exactly one character
        * To escape `%` and `_` (or to match the `escape` parameter
          itself), prefix the desired character with `escape`.
    escape
        Escape character

    Returns
    -------
    str
        A regular expression pattern equivalent to the input SQL `LIKE` pattern.

    Examples
    --------
    >>> sql_like_to_regex("6%")  # default is to not escape anything
    '^6.*$'
    >>> sql_like_to_regex("6^%", escape="^")
    '^6%$'
    >>> sql_like_to_regex("6_")
    '^6.$'
    >>> sql_like_to_regex("6/_", escape="/")
    '^6_$'
    >>> sql_like_to_regex("%abc")  # any string ending with "abc"
    '^.*abc$'
    >>> sql_like_to_regex("abc%")  # any string starting with "abc"
    '^abc.*$'

    """
    cur_i = 0
    pattern_length = len(pattern)

    while cur_i < pattern_length:
        nxt_i = cur_i + 1

        cur = pattern[cur_i]
        nxt = pattern[nxt_i] if nxt_i < pattern_length else None

        skip = 1

        if nxt is not None and escape is not None and cur == escape:
            yield nxt
            skip = 2
        elif cur == "%":
            yield ".*"
        elif cur == "_":
            yield "."
        else:
            yield cur

        cur_i += skip


def sql_like_to_regex(pattern, escape=None):
    return f"^{''.join(_sql_like_to_regex(pattern, escape))}$"


def string_sqllike_serieswise(arg, pattern, escape):
    pat = sql_like_to_regex(pattern, escape)
    return arg.str.contains(pat, regex=True)


def string_sqlilike_serieswise(arg, pattern, escape):
    pat = sql_like_to_regex(pattern, escape)
    return arg.str.contains(pat, regex=True, flags=re.IGNORECASE)


def extract_userinfo_elementwise(x):
    url_parts = urlsplit(x)
    username = url_parts.username or ""
    password = url_parts.password or ""
    return f"{username}:{password}"


def extract_queryparam_rowwise(row):
    query = urlsplit(row["arg"]).query
    param_name = row["key"]
    if param_name is not None:
        value = parse_qs(query)[param_name]
        return value if len(value) > 1 else value[0]
    else:
        return query


def array_index_rowwise(row):
    try:
        return row["arg"][row["index"]]
    except IndexError:
        return None


def array_position_rowwise(row):
    try:
        return row["arg"].index(row["other"])
    except ValueError:
        return -1


def array_slice_rowwise(row):
    arg, start, stop = row["arg"], row["start"], row["stop"]
    if isnull(start) and isnull(stop):
        return arg
    elif isnull(start):
        return arg[:stop]
    elif isnull(stop):
        return arg[start:]
    else:
        return arg[start:stop]


def integer_range_rowwise(row):
    if not row["step"]:
        return []
    return list(np.arange(row["start"], row["stop"], row["step"]))


def timestamp_range_rowwise(row):
    if not row["step"]:
        return []
    return list(
        pd.date_range(row["start"], row["stop"], freq=row["step"], inclusive="left")
    )


def _safe_method(mapping, method, *args, **kwargs):
    if isnull(mapping):
        return None
    try:
        method = getattr(mapping, method)
    except AttributeError:
        return None
    else:
        result = method(*args, **kwargs)
        return None if isnull(result) else result


def safe_len(mapping):
    return _safe_method(mapping, "__len__")


def safe_get(mapping, key, default=None):
    return _safe_method(mapping, "get", key, default)


def safe_contains(mapping, key):
    return _safe_method(mapping, "__contains__", key)


def safe_keys(mapping):
    result = _safe_method(mapping, "keys")
    if result is None:
        return None
    # list(...) to unpack iterable
    return np.array(list(result))


def safe_values(mapping):
    result = _safe_method(mapping, "values")
    if result is None or result is pd.NA:
        return None
    # list(...) to unpack iterable
    return np.array(list(result), dtype="object")


def safe_merge(left, right):
    if isnull(left) or isnull(right):
        return None
    else:
        return {**left, **right}


def safe_json_getitem(value, key):
    try:
        # try to deserialize the value -> return None if it's None
        if (js := json.loads(value)) is None:
            return None
    except (json.JSONDecodeError, TypeError):
        # if there's an error related to decoding or a type error return None
        return None

    try:
        # try to extract the value as an array element or mapping key
        return js[key]
    except (KeyError, IndexError, TypeError):
        # KeyError: missing mapping key
        # IndexError: missing sequence key
        # TypeError: `js` doesn't implement __getitem__, either at all or for
        # the type of `key`
        return None


def safe_decimal(func):
    def wrapper(x, **kwargs):
        try:
            return func(x, **kwargs)
        except decimal.InvalidOperation:
            return decimal.Decimal("NaN")

    return wrapper


def round_serieswise(arg, digits):
    if digits is None:
        return np.round(arg).astype("int64")
    else:
        return np.round(arg, digits).astype("float64")


reductions = {
    ops.Min: lambda x: x.min(),
    ops.Max: lambda x: x.max(),
    ops.Sum: lambda x: x.sum(),
    ops.Mean: lambda x: x.mean(),
    ops.Count: lambda x: x.count(),
    ops.Mode: lambda x: x.mode().iat[0],
    ops.Any: lambda x: x.any(),
    ops.All: lambda x: x.all(),
    ops.Median: lambda x: x.median(),
    ops.ApproxMedian: lambda x: x.median(),
    ops.BitAnd: lambda x: np.bitwise_and.reduce(x.values),
    ops.BitOr: lambda x: np.bitwise_or.reduce(x.values),
    ops.BitXor: lambda x: np.bitwise_xor.reduce(x.values),
    ops.Last: lambda x: x.iat[-1],
    ops.First: lambda x: x.iat[0],
    ops.CountDistinct: lambda x: x.nunique(),
    ops.ApproxCountDistinct: lambda x: x.nunique(),
    ops.ArrayCollect: lambda x: x.tolist(),
}

generic = {
    ops.Abs: abs,
    ops.Acos: np.arccos,
    ops.Add: operator.add,
    ops.And: operator.and_,
    ops.Asin: np.arcsin,
    ops.Atan: np.arctan,
    ops.Atan2: np.arctan2,
    ops.BitwiseAnd: lambda x, y: np.bitwise_and(x, y),
    ops.BitwiseLeftShift: lambda x, y: np.left_shift(x, y).astype("int64"),
    ops.BitwiseNot: np.invert,
    ops.BitwiseOr: lambda x, y: np.bitwise_or(x, y),
    ops.BitwiseRightShift: lambda x, y: np.right_shift(x, y).astype("int64"),
    ops.BitwiseXor: lambda x, y: np.bitwise_xor(x, y),
    ops.Ceil: lambda x: np.ceil(x).astype("int64"),
    ops.Cos: np.cos,
    ops.Cot: lambda x: 1 / np.tan(x),
    ops.DateAdd: operator.add,
    ops.DateDiff: operator.sub,
    ops.DateSub: operator.sub,
    ops.Degrees: np.degrees,
    ops.Divide: operator.truediv,
    ops.Equals: operator.eq,
    ops.Exp: np.exp,
    ops.Floor: lambda x: np.floor(x).astype("int64"),
    ops.FloorDivide: operator.floordiv,
    ops.Greater: operator.gt,
    ops.GreaterEqual: operator.ge,
    ops.IdenticalTo: lambda x, y: (x == y) | (pd.isnull(x) & pd.isnull(y)),
    ops.IntervalAdd: operator.add,
    ops.IntervalFloorDivide: operator.floordiv,
    ops.IntervalMultiply: operator.mul,
    ops.IntervalSubtract: operator.sub,
    ops.IsInf: np.isinf,
    ops.IsNull: pd.isnull,
    ops.Less: operator.lt,
    ops.LessEqual: operator.le,
    ops.Ln: np.log,
    ops.Log10: np.log10,
    ops.Log2: np.log2,
    ops.Modulus: operator.mod,
    ops.Multiply: operator.mul,
    ops.Negate: lambda x: not x if isinstance(x, (bool, np.bool_)) else -x,
    ops.Not: lambda x: not x if isinstance(x, (bool, np.bool_)) else ~x,
    ops.NotEquals: operator.ne,
    ops.NotNull: pd.notnull,
    ops.Or: operator.or_,
    ops.Power: operator.pow,
    ops.Radians: np.radians,
    ops.Sign: np.sign,
    ops.Sin: np.sin,
    ops.Sqrt: np.sqrt,
    ops.Subtract: operator.sub,
    ops.Tan: np.tan,
    ops.TimestampAdd: operator.add,
    ops.TimestampDiff: operator.sub,
    ops.TimestampSub: operator.sub,
    ops.Xor: operator.xor,
    ops.E: lambda: np.e,
    ops.Pi: lambda: np.pi,
    ops.TimestampNow: lambda: pd.Timestamp("now", tz="UTC").tz_localize(None),
    ops.StringConcat: lambda xs: reduce(operator.add, xs),
    ops.StringJoin: lambda xs, sep: reduce(lambda x, y: x + sep + y, xs),
    ops.Log: lambda x, base: np.log(x) if base is None else np.log(x) / np.log(base),
}

columnwise = {
    ops.Clip: lambda df: df["arg"].clip(lower=df["lower"], upper=df["upper"]),
    ops.IfElse: lambda df: df["true_expr"].where(
        df["bool_expr"], other=df["false_null_expr"]
    ),
    ops.NullIf: lambda df: df["arg"].where(df["arg"] != df["null_if_expr"]),
    ops.Repeat: lambda df: df["arg"] * df["times"],
}

rowwise = {
    ops.ArrayContains: lambda row: row["other"] in row["arg"],
    ops.ArrayIndex: array_index_rowwise,
    ops.ArrayPosition: array_position_rowwise,
    ops.ArrayRemove: lambda row: [x for x in row["arg"] if x != row["other"]],
    ops.ArrayRepeat: lambda row: np.tile(row["arg"], max(0, row["times"])),
    ops.ArraySlice: array_slice_rowwise,
    ops.ArrayUnion: lambda row: toolz.unique(row["left"] + row["right"]),
    ops.EndsWith: lambda row: row["arg"].endswith(row["end"]),
    ops.IntegerRange: integer_range_rowwise,
    ops.JSONGetItem: lambda row: safe_json_getitem(row["arg"], row["index"]),
    ops.Map: lambda row: dict(zip(row["keys"], row["values"])),
    ops.MapGet: lambda row: safe_get(row["arg"], row["key"], row["default"]),
    ops.MapContains: lambda row: safe_contains(row["arg"], row["key"]),
    ops.MapMerge: lambda row: safe_merge(row["left"], row["right"]),
    ops.TimestampRange: timestamp_range_rowwise,
    ops.LPad: lambda row: row["arg"].rjust(row["length"], row["pad"]),
    ops.RegexExtract: lambda row: re.search(row["pattern"], row["arg"]).group(
        row["index"]
    ),
    ops.RegexReplace: lambda row: re.sub(
        row["pattern"], row["replacement"], row["arg"]
    ),
    ops.RegexSearch: lambda row: re.search(row["pattern"], row["arg"]) is not None,
    ops.RPad: lambda row: row["arg"].ljust(row["length"], row["pad"]),
    ops.StartsWith: lambda row: row["arg"].startswith(row["start"]),
    ops.StringContains: lambda row: row["haystack"].contains(row["needle"]),
    ops.StringFind: lambda row: row["arg"].find(
        row["substr"], row["start"], row["end"]
    ),
    ops.StringReplace: lambda row: row["arg"].replace(
        row["pattern"], row["replacement"]
    ),
    ops.StringSplit: lambda row: row["arg"].split(row["delimiter"]),
    ops.StrRight: lambda row: row["arg"][-row["nchars"] :],
    ops.Translate: lambda row: row["arg"].translate(
        str.maketrans(row["from_str"], row["to_str"])
    ),
    ops.Substring: substring_rowwise,
    ops.ExtractQuery: extract_queryparam_rowwise,
    ops.Strftime: lambda row: row["arg"].strftime(row["format_str"]),
}

serieswise = {
    ops.Between: lambda arg, lower_bound, upper_bound: arg.between(
        lower_bound, upper_bound
    ),
    ops.Capitalize: lambda arg: arg.str.capitalize(),
    ops.Date: lambda arg: arg.dt.floor("d"),
    ops.DayOfWeekIndex: lambda arg: pd.to_datetime(arg).dt.dayofweek,
    ops.DayOfWeekName: lambda arg: pd.to_datetime(arg).dt.day_name(),
    ops.EndsWith: lambda arg, end: arg.str.endswith(end),
    ops.ExtractDay: lambda arg: arg.dt.day,
    ops.ExtractDayOfYear: lambda arg: arg.dt.dayofyear,
    ops.ExtractEpochSeconds: lambda arg: arg.astype("datetime64[s]")
    .astype("int64")
    .astype("int32"),
    ops.ExtractHour: lambda arg: arg.dt.hour,
    ops.ExtractMicrosecond: lambda arg: arg.dt.microsecond,
    ops.ExtractMillisecond: lambda arg: arg.dt.microsecond // 1000,
    ops.ExtractMinute: lambda arg: arg.dt.minute,
    ops.ExtractMonth: lambda arg: arg.dt.month,
    ops.ExtractQuarter: lambda arg: arg.dt.quarter,
    ops.ExtractSecond: lambda arg: arg.dt.second,
    ops.ExtractWeekOfYear: lambda arg: arg.dt.isocalendar().week.astype("int32"),
    ops.ExtractYear: lambda arg: arg.dt.year,
    ops.IsNull: lambda arg: arg.isnull(),
    ops.NotNull: lambda arg: arg.notnull(),
    ops.Lowercase: lambda arg: arg.str.lower(),
    ops.LPad: lambda arg, length, pad: arg.str.rjust(length, fillchar=pad),
    ops.LStrip: lambda arg: arg.str.lstrip(),
    ops.Repeat: lambda arg, times: arg.str.repeat(times),
    ops.Reverse: lambda arg: arg.str[::-1],
    ops.Round: round_serieswise,
    ops.RPad: lambda arg, length, pad: arg.str.ljust(length, fillchar=pad),
    ops.RStrip: lambda arg: arg.str.rstrip(),
    ops.StartsWith: lambda arg, start: arg.str.startswith(start),
    ops.StringAscii: lambda arg: arg.map(ord, na_action="ignore").astype("int32"),
    ops.StringContains: lambda haystack, needle: haystack.str.contains(
        needle, regex=False
    ),
    ops.StringFind: lambda arg, substr, start, end: arg.str.find(substr, start, end),
    ops.StringLength: lambda arg: arg.str.len().astype("int32"),
    ops.StringReplace: lambda arg, pattern, replacement: arg.str.replace(
        pattern, replacement
    ),
    ops.StringSplit: lambda arg, delimiter: arg.str.split(delimiter),
    ops.StringSQLLike: string_sqllike_serieswise,
    ops.StringSQLILike: string_sqlilike_serieswise,
    ops.Strip: lambda arg: arg.str.strip(),
    ops.Strftime: lambda arg, format_str: arg.dt.strftime(format_str),
    ops.StrRight: lambda arg, nchars: arg.str[-nchars:],
    ops.Substring: substring_serieswise,
    ops.Time: lambda arg: arg.dt.time,
    ops.TimestampFromUNIX: lambda arg, unit: pd.to_datetime(arg, unit=unit.short),
    ops.Translate: lambda arg, from_str, to_str: arg.str.translate(
        str.maketrans(from_str, to_str)
    ),
    ops.Uppercase: lambda arg: arg.str.upper(),
}

elementwise = {
    ops.ExtractProtocol: lambda x: getattr(urlsplit(x), "scheme", ""),
    ops.ExtractAuthority: lambda x: getattr(urlsplit(x), "netloc", ""),
    ops.ExtractPath: lambda x: getattr(urlsplit(x), "path", ""),
    ops.ExtractFragment: lambda x: getattr(urlsplit(x), "fragment", ""),
    ops.ExtractHost: lambda x: getattr(urlsplit(x), "hostname", ""),
    ops.ExtractUserInfo: extract_userinfo_elementwise,
    ops.StructField: lambda x, field: safe_get(x, field),
    ops.ArrayLength: len,
    ops.ArrayFlatten: toolz.concat,
    ops.ArraySort: sorted,
    ops.ArrayDistinct: toolz.unique,
    ops.MapLength: safe_len,
    ops.MapKeys: safe_keys,
    ops.MapValues: safe_values,
    ops.Round: lambda x, digits=0: round(x, digits),
}


elementwise_decimal = {
    ops.Round: lambda x, digits=0: round(x, digits),
    ops.Log10: safe_decimal(lambda x: x.log10()),
    ops.Ln: safe_decimal(lambda x: x.ln()),
    ops.Exp: safe_decimal(lambda x: x.exp()),
    ops.Floor: safe_decimal(math.floor),
    ops.Ceil: safe_decimal(math.ceil),
    ops.Sqrt: safe_decimal(lambda x: x.sqrt()),
    ops.Log2: safe_decimal(lambda x: x.ln() / decimal.Decimal(2).ln()),
    ops.Sign: safe_decimal(lambda x: math.copysign(1, x)),
    ops.Log: safe_decimal(lambda x, base: x.ln() / decimal.Decimal(base).ln()),
}


supported_operations = (
    generic.keys()
    | columnwise.keys()
    | rowwise.keys()
    | serieswise.keys()
    | elementwise.keys()
)
