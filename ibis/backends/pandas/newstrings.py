from __future__ import annotations

try:
    import regex as re
except ImportError:
    import re
from functools import reduce

import pandas as pd

import ibis.expr.operations as ops
from ibis.backends.pandas.newpandas import execute
from ibis.backends.pandas.newutils import columnwise, rowwise, sql_like_to_regex
from ibis.util import any_of


def _substring_rowwise(row):
    arg, start, length = row["arg"], row["start"], row["length"]
    if length is None:
        return arg[start:]
    else:
        return arg[start : start + length]


def _substring_columnwise(df, start, length):
    if length is None:
        return df["arg"].str[start:]
    else:
        return df["arg"].str[start : start + length]


_columnwise_functions = {
    ops.Strip: lambda df: df["arg"].str.strip(),
    ops.LStrip: lambda df: df["arg"].str.lstrip(),
    ops.RStrip: lambda df: df["arg"].str.rstrip(),
    ops.Reverse: lambda df: df["arg"].str[::-1],
    ops.Lowercase: lambda df: df["arg"].str.lower(),
    ops.Uppercase: lambda df: df["arg"].str.upper(),
    ops.StringLength: lambda df: df["arg"].str.len().astype("int32"),
    ops.Capitalize: lambda df: df["arg"].str.capitalize(),
    ops.Repeat: lambda df: df["arg"].str.repeat(df["times"]),
    ops.StringReplace: lambda df, pattern, replacement: df["arg"].str.replace(
        pattern, replacement
    ),
    ops.LPad: lambda df, length, pad: df["arg"].str.rjust(length, fillchar=pad),
    ops.RPad: lambda df, length, pad: df["arg"].str.ljust(length, fillchar=pad),
    ops.StrRight: lambda df, nchars: df["arg"].str[-nchars:],
    ops.StartsWith: lambda df, start: df["arg"].str.startswith(start),
    ops.EndsWith: lambda df, end: df["arg"].str.endswith(end),
    ops.StringAscii: lambda df: df["arg"].map(ord).astype("int32"),
    ops.StringFind: lambda df, substr, start, end: df["arg"].str.find(
        substr, start, end
    ),
    ops.StringContains: lambda df, needle: df["haystack"].str.contains(
        needle, regex=False
    ),
    ops.StringSplit: lambda df, delimiter: df["arg"].str.split(delimiter),
    ops.RegexSearch: lambda df, pattern: df["arg"].str.contains(pattern),
    # ops.RegexExtract: lambda df, pattern, index: df["arg"].str.extract(pattern).iloc[:, index],
    ops.RegexReplace: lambda df, pattern, replacement: df["arg"].str.replace(
        pattern, replacement, regex=True
    ),
    ops.StringSQLLike: lambda df, pattern, escape: df["arg"].str.contains(
        sql_like_to_regex(pattern, escape), regex=True
    ),
    ops.Substring: _substring_columnwise,
}

_rowwise_functions = {
    ops.LPad: lambda row: row["arg"].rjust(row["length"], row["pad"]),
    ops.RPad: lambda row: row["arg"].ljust(row["length"], row["pad"]),
    ops.StrRight: lambda row: row["arg"][-row["nchars"] :],
    ops.StartsWith: lambda row: row["arg"].startswith(row["start"]),
    ops.EndsWith: lambda row: row["arg"].endswith(row["end"]),
    ops.StringFind: lambda row: row["arg"].find(
        row["substr"], row["start"], row["end"]
    ),
    ops.StringReplace: lambda row: row["arg"].replace(
        row["pattern"], row["replacement"]
    ),
    ops.StringContains: lambda row: row["haystack"].contains(row["needle"]),
    ops.StringSplit: lambda row: row["arg"].split(row["delimiter"]),
    ops.RegexSearch: lambda row: re.search(row["pattern"], row["arg"]) is not None,
    ops.RegexExtract: lambda row: re.search(row["pattern"], row["arg"]).group(
        row["index"]
    ),
    ops.RegexReplace: lambda row: re.sub(
        row["pattern"], row["replacement"], row["arg"]
    ),
    ops.Substring: _substring_rowwise,
}


def _columnwise_with_fallback(op, argname, kwargs):
    # if any of the arguments except argname are columnar shaped then we need to
    # fallback to rowwise
    arg = kwargs.pop(argname)
    if any_of(kwargs.values(), pd.Series):
        data = {argname: arg, **kwargs}
        func = _rowwise_functions[type(op)]
        return rowwise(func, data)
    else:
        data = {argname: arg}
        func = _columnwise_functions[type(op)]
        return columnwise(func, data, **kwargs)


@execute.register(ops.Strip)
@execute.register(ops.LStrip)
@execute.register(ops.RStrip)
@execute.register(ops.Reverse)
@execute.register(ops.Lowercase)
@execute.register(ops.Uppercase)
@execute.register(ops.StringLength)
@execute.register(ops.Capitalize)
@execute.register(ops.Repeat)
@execute.register(ops.StrRight)
@execute.register(ops.StringAscii)
def execute_columnwise(op, **kwargs):
    func = _columnwise_functions[type(op)]
    return columnwise(func, kwargs)


@execute.register(ops.LPad)
@execute.register(ops.RPad)
@execute.register(ops.StrRight)
@execute.register(ops.StartsWith)
@execute.register(ops.EndsWith)
@execute.register(ops.StringFind)
@execute.register(ops.StringReplace)
@execute.register(ops.StringContains)
@execute.register(ops.StringSplit)
@execute.register(ops.Substring)
@execute.register(ops.RegexSearch)
@execute.register(ops.RegexExtract)
@execute.register(ops.RegexReplace)
@execute.register(ops.StringSQLLike)
def execute_columnwise_or_rowwise(op, **kwargs):
    return _columnwise_with_fallback(op, "arg", kwargs)


@execute.register(ops.RegexExtract)
def execute_rowwise(op, **kwargs):
    func = _rowwise_functions[type(op)]
    return rowwise(func, kwargs)


@execute.register(ops.StringJoin)
def execute_string_join(op, sep, arg):
    return reduce(lambda x, y: x + sep + y, arg)


@execute.register(ops.StringConcat)
def execute_string_concat(op, arg):
    return reduce(lambda x, y: x + y, arg)


@execute.register(ops.StringContains)
def execute_string_contains(op, **kwargs):
    return _columnwise_with_fallback(op, "haystack", kwargs)
