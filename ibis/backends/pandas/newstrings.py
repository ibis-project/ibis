from __future__ import annotations

try:
    import regex as re
except ImportError:
    import re
from functools import reduce
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd

import ibis.expr.operations as ops
from ibis.backends.pandas.newpandas import execute
from ibis.backends.pandas.newutils import (
    asframe,
    columnwise,
    elementwise,
    rowwise,
    serieswise,
    sql_like_to_regex,
)
from ibis.util import any_of


def _extract_userinfo_elementwise(x):
    url_parts = urlsplit(x)
    username = url_parts.username or ""
    password = url_parts.password or ""
    return f"{username}:{password}"


def _extract_queryparam_rowwise(row):
    query = urlsplit(row["arg"]).query
    param_name = row["key"]
    if param_name is not None:
        value = parse_qs(query)[param_name]
        return value if len(value) > 1 else value[0]
    else:
        return query


def _substring_rowwise(row):
    arg, start, length = row["arg"], row["start"], row["length"]
    if length is None:
        return arg[start:]
    else:
        return arg[start : start + length]


def _substring_serieswise(arg, start, length):
    if length is None:
        return arg.str[start:]
    else:
        return arg.str[start : start + length]


_columnwise_functions = {}

_rowwise_functions = {
    ops.EndsWith: lambda row: row["arg"].endswith(row["end"]),
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
    ops.Substring: _substring_rowwise,
    ops.Translate: lambda row: row["arg"].translate(
        str.maketrans(row["from_str"], row["to_str"])
    ),
    ops.ExtractQuery: _extract_queryparam_rowwise,
}

_serieswise_functions = {
    ops.Capitalize: lambda arg: arg.str.capitalize(),
    ops.EndsWith: lambda arg, end: arg.str.endswith(end),
    ops.Lowercase: lambda arg: arg.str.lower(),
    ops.LPad: lambda arg, length, pad: arg.str.rjust(length, fillchar=pad),
    ops.LStrip: lambda arg: arg.str.lstrip(),
    # TODO(kszucs): while these are properly working, the pandas implementation
    # doesn't seem to support POSIX patterns
    # ops.RegexExtract: lambda arg, pattern, index: arg.str.extract(pattern).iloc[
    #     :, index - 1
    # ],
    # ops.RegexReplace: lambda arg, pattern, replacement: arg.str.replace(
    #     pattern, replacement, regex=True
    # ),
    # ops.RegexSearch: lambda arg, pattern: arg.str.contains(pattern, regex=True),
    ops.Repeat: lambda arg, times: arg.str.repeat(times),
    ops.Reverse: lambda arg: arg.str[::-1],
    ops.RPad: lambda arg, length, pad: arg.str.ljust(length, fillchar=pad),
    ops.RStrip: lambda arg: arg.str.rstrip(),
    ops.StartsWith: lambda arg, start: arg.str.startswith(start),
    ops.StringAscii: lambda arg: arg.map(ord).astype("int32"),
    ops.StringContains: lambda haystack, needle: haystack.str.contains(
        needle, regex=False
    ),
    ops.StringFind: lambda arg, substr, start, end: arg.str.find(substr, start, end),
    ops.StringLength: lambda arg: arg.str.len().astype("int32"),
    ops.StringReplace: lambda arg, pattern, replacement: arg.str.replace(
        pattern, replacement
    ),
    ops.StringSplit: lambda arg, delimiter: arg.str.split(delimiter),
    ops.StringSQLLike: lambda arg, pattern, escape: arg.str.contains(
        sql_like_to_regex(pattern, escape), regex=True
    ),
    ops.Strip: lambda arg: arg.str.strip(),
    ops.StrRight: lambda arg, nchars: arg.str[-nchars:],
    ops.Substring: _substring_serieswise,
    ops.Uppercase: lambda arg: arg.str.upper(),
    ops.Translate: lambda arg, from_str, to_str: arg.str.translate(
        str.maketrans(from_str, to_str)
    ),
}

_elementwise_functions = {
    ops.ExtractProtocol: lambda x: getattr(urlsplit(x), "scheme", ""),
    ops.ExtractAuthority: lambda x: getattr(urlsplit(x), "netloc", ""),
    ops.ExtractPath: lambda x: getattr(urlsplit(x), "path", ""),
    ops.ExtractFragment: lambda x: getattr(urlsplit(x), "fragment", ""),
    ops.ExtractHost: lambda x: getattr(urlsplit(x), "hostname", ""),
    ops.ExtractUserInfo: _extract_userinfo_elementwise,
}


def _pick_implementation(op, operands):
    # if the rest of the operands contain columnar operands then prefer the
    # columnwise implementation with a fallback to rowwise implementation,
    # otherwise prefer the serieswise implementation with a fallback to
    # elementwise implementation

    typ = type(op)
    first, *rest = operands.values()
    is_multi_column = any_of(rest, pd.Series)

    if is_multi_column:
        # only columnwise or rowwise implementations can be considered
        if func := _columnwise_functions.get(typ):
            # print(typ, "COLUMNWISE")
            return columnwise(func, operands)
        elif func := _rowwise_functions.get(typ):
            # print(typ, "ROWWISE")
            return rowwise(func, operands)
        else:
            raise TypeError(f"No columnwise or rowwise implementation found for {typ}")
    elif func := _serieswise_functions.get(typ):
        # print(typ, "SERIESWISE")
        return serieswise(func, **operands)
    elif func := _elementwise_functions.get(typ):
        # print(typ, "ELEMENTWISE")
        return elementwise(func, **operands)
    elif func := _rowwise_functions.get(typ):
        # print(typ, "ROWWISE")
        return rowwise(func, operands)
    else:
        raise TypeError(
            f"No serieswise, elementwise or rowwise implementation found for {typ}"
        )


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
@execute.register(ops.Translate)
@execute.register(ops.Strip)
@execute.register(ops.LStrip)
@execute.register(ops.RStrip)
@execute.register(ops.Reverse)
@execute.register(ops.Lowercase)
@execute.register(ops.Uppercase)
@execute.register(ops.StringLength)
@execute.register(ops.Capitalize)
@execute.register(ops.Repeat)
@execute.register(ops.StringAscii)
@execute.register(ops.StrRight)
@execute.register(ops.ExtractProtocol)
@execute.register(ops.ExtractAuthority)
@execute.register(ops.ExtractPath)
@execute.register(ops.ExtractFragment)
@execute.register(ops.ExtractHost)
@execute.register(ops.ExtractUserInfo)
@execute.register(ops.ExtractQuery)
def execute_try_serieswise(op, **kwargs):
    return _pick_implementation(op, kwargs)


@execute.register(ops.StringJoin)
def execute_string_join(op, sep, arg):
    return reduce(lambda x, y: x + sep + y, arg)


@execute.register(ops.StringConcat)
def execute_string_concat(op, arg):
    return reduce(lambda x, y: x + y, arg)


@execute.register(ops.FindInSet)
def execute_find_in_set(op, needle, values):
    (needle, *haystack), _ = asframe((needle, *values), concat=False)
    condlist = [needle == col for col in haystack]
    choicelist = [i for i, _ in enumerate(haystack)]
    result = np.select(condlist, choicelist, default=-1)
    return pd.Series(result, name=op.name)
