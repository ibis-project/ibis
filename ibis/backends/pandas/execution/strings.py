from __future__ import annotations

import itertools
import json
import operator
from functools import partial, reduce
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
import toolz
from pandas.core.groupby import SeriesGroupBy

import ibis.expr.operations as ops
import ibis.util
from ibis.backends.pandas.core import execute, integer_types, scalar_types
from ibis.backends.pandas.dispatch import execute_node
from ibis.backends.pandas.execution.util import get_grouping


@execute_node.register(ops.StringContains, pd.Series, (pd.Series, str))
def execute_string_contains(_, data, needle, **kwargs):
    return data.str.contains(needle)


@execute_node.register(ops.StringSQLLike, pd.Series, str, (str, type(None)))
def execute_string_like_series_string(op, data, pattern, escape, **kwargs):
    new_pattern = sql_like_to_regex(pattern, escape=escape)
    return data.str.contains(new_pattern, regex=True)


@execute_node.register(ops.StringSQLLike, SeriesGroupBy, str, str)
def execute_string_like_series_groupby_string(op, data, pattern, escape, **kwargs):
    return execute_string_like_series_string(
        op, data.obj, pattern, escape, **kwargs
    ).groupby(get_grouping(data.grouper.groupings), group_keys=False)


@execute_node.register(ops.GroupConcat, pd.Series, str, (pd.Series, type(None)))
def execute_group_concat_series_mask(op, data, sep, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data[mask] if mask is not None else data,
        lambda series, sep=sep: sep.join(series.values),
    )


@execute_node.register(ops.GroupConcat, SeriesGroupBy, str, type(None))
def execute_group_concat_series_gb(op, data, sep, _, aggcontext=None, **kwargs):
    return aggcontext.agg(data, lambda data, sep=sep: sep.join(data.values.astype(str)))


@execute_node.register(ops.GroupConcat, SeriesGroupBy, str, SeriesGroupBy)
def execute_group_concat_series_gb_mask(op, data, sep, mask, aggcontext=None, **kwargs):
    def method(series, sep=sep):
        if series.empty:
            return pd.NA
        return sep.join(series.values.astype(str))

    return aggcontext.agg(
        data,
        lambda data, mask=mask.obj, method=method: method(data[mask[data.index]]),
    )


@execute_node.register(ops.RegexExtract, pd.Series, str, integer_types)
def execute_series_regex_extract(op, data, pattern, index, **kwargs):
    pattern = re.compile(pattern)
    return pd.Series(
        [
            None if (match is None or index > match.lastindex) else match[index]
            for match in map(pattern.search, data)
        ],
        dtype=data.dtype,
        name=data.name,
    )


@execute_node.register(ops.RegexExtract, SeriesGroupBy, str, integer_types)
def execute_series_regex_extract_gb(op, data, pattern, index, **kwargs):
    return execute_series_regex_extract(op, data.obj, pattern, index, **kwargs).groupby(
        get_grouping(data.grouper.groupings), group_keys=False
    )


@execute_node.register(ops.RegexReplace, pd.Series, str, str)
def execute_series_regex_replace(op, data, pattern, replacement, **kwargs):
    pattern = re.compile(pattern)

    def replacer(x, pattern=pattern):
        return pattern.sub(replacement, x)

    return data.apply(replacer)


@execute_node.register(ops.RegexReplace, str, str, str)
def execute_str_regex_replace(_, arg, pattern, replacement, **kwargs):
    return re.sub(pattern, replacement, arg)


@execute_node.register(ops.RegexReplace, SeriesGroupBy, str, str)
def execute_series_regex_replace_gb(op, data, pattern, replacement, **kwargs):
    return execute_series_regex_replace(
        data.obj, pattern, replacement, **kwargs
    ).groupby(get_grouping(data.grouper.groupings), group_keys=False)


@execute_node.register(ops.Translate, pd.Series, pd.Series, pd.Series)
def execute_series_translate_series_series(op, data, from_string, to_string, **kwargs):
    tables = [
        str.maketrans(source, target) for source, target in zip(from_string, to_string)
    ]
    return pd.Series(
        [string.translate(table) for string, table in zip(data, tables)],
        dtype=data.dtype,
        name=data.name,
    )


@execute_node.register(ops.Translate, pd.Series, pd.Series, str)
def execute_series_translate_series_scalar(op, data, from_string, to_string, **kwargs):
    tables = [str.maketrans(source, to_string) for source in from_string]
    return pd.Series(
        [string.translate(table) for string, table in zip(data, tables)],
        dtype=data.dtype,
        name=data.name,
    )


@execute_node.register(ops.Translate, pd.Series, str, pd.Series)
def execute_series_translate_scalar_series(op, data, from_string, to_string, **kwargs):
    tables = [str.maketrans(from_string, target) for target in to_string]
    return pd.Series(
        [string.translate(table) for string, table in zip(data, tables)],
        dtype=data.dtype,
        name=data.name,
    )


@execute_node.register(ops.Translate, pd.Series, str, str)
def execute_series_translate_scalar_scalar(op, data, from_string, to_string, **kwargs):
    return data.str.translate(str.maketrans(from_string, to_string))


def haystack_to_series_of_lists(haystack, index=None):
    if index is None:
        index = toolz.first(
            piece.index for piece in haystack if hasattr(piece, "index")
        )
    pieces = reduce(
        operator.add,
        (
            pd.Series(getattr(piece, "values", piece), index=index).map(
                ibis.util.promote_list
            )
            for piece in haystack
        ),
    )
    return pieces


@execute_node.register(ops.FindInSet, pd.Series, tuple)
def execute_series_find_in_set(op, needle, haystack, **kwargs):
    haystack = [execute(arg, **kwargs) for arg in haystack]
    pieces = haystack_to_series_of_lists(haystack, index=needle.index)
    index = itertools.count()
    return pieces.map(
        lambda elements, needle=needle, index=index: (
            ibis.util.safe_index(elements, needle.iat[next(index)])
        )
    )


@execute_node.register(ops.FindInSet, SeriesGroupBy, list)
def execute_series_group_by_find_in_set(op, needle, haystack, **kwargs):
    pieces = [getattr(piece, "obj", piece) for piece in haystack]
    return execute_series_find_in_set(op, needle.obj, pieces, **kwargs).groupby(
        get_grouping(needle.grouper.groupings), group_keys=False
    )


@execute_node.register(ops.FindInSet, scalar_types, list)
def execute_string_group_by_find_in_set(op, needle, haystack, **kwargs):
    # `list` could contain series, series groupbys, or scalars
    # mixing series and series groupbys is not allowed
    series_in_haystack = [
        type(piece)
        for piece in haystack
        if isinstance(piece, (pd.Series, SeriesGroupBy))
    ]

    if not series_in_haystack:
        return ibis.util.safe_index(haystack, needle)

    try:
        (collection_type,) = frozenset(map(type, series_in_haystack))
    except ValueError:
        raise ValueError("Mixing Series and SeriesGroupBy is not allowed")

    pieces = haystack_to_series_of_lists(
        [getattr(piece, "obj", piece) for piece in haystack]
    )

    result = pieces.map(toolz.flip(ibis.util.safe_index)(needle))
    if issubclass(collection_type, pd.Series):
        return result

    assert issubclass(collection_type, SeriesGroupBy)

    return result.groupby(
        get_grouping(
            toolz.first(
                piece.grouper.groupings
                for piece in haystack
                if hasattr(piece, "grouper")
            )
        ),
        group_keys=False,
    )


def try_getitem(value, key):
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


@execute_node.register(ops.JSONGetItem, pd.Series, (str, int))
def execute_json_getitem_series_str_int(_, data, key, **kwargs):
    return pd.Series(map(partial(try_getitem, key=key), data), dtype="object")


@execute_node.register(ops.JSONGetItem, pd.Series, pd.Series)
def execute_json_getitem_series_series(_, data, key, **kwargs):
    return pd.Series(map(try_getitem, data, key), dtype="object")


def _extract_url_field(data, field_name):
    if isinstance(data, str):
        return getattr(urlsplit(data), field_name, "")

    return pd.Series(
        [getattr(urlsplit(string), field_name, "") for string in data],
        dtype=data.dtype,
        name=data.name,
    )


@execute_node.register(ops.ExtractProtocol, (pd.Series, str))
def execute_extract_protocol(op, data, **kwargs):
    return _extract_url_field(data, "scheme")


@execute_node.register(ops.ExtractAuthority, (pd.Series, str))
def execute_extract_authority(op, data, **kwargs):
    return _extract_url_field(data, "netloc")


@execute_node.register(ops.ExtractPath, (pd.Series, str))
def execute_extract_path(op, data, **kwargs):
    return _extract_url_field(data, "path")


@execute_node.register(ops.ExtractFragment, (pd.Series, str))
def execute_extract_fragment(op, data, **kwargs):
    return _extract_url_field(data, "fragment")


@execute_node.register(ops.ExtractHost, (pd.Series, str))
def execute_extract_host(op, data, **kwargs):
    return _extract_url_field(data, "hostname")


@execute_node.register(ops.ExtractQuery, (pd.Series, str), (str, type(None)))
def execute_extract_query(op, data, key, **kwargs):
    def extract_query_param(url, param_name):
        query = urlsplit(url).query
        if param_name is not None:
            value = parse_qs(query)[param_name]
            return value if len(value) > 1 else value[0]
        else:
            return query

    if isinstance(data, str):
        return extract_query_param(data, key)

    return pd.Series(
        [extract_query_param(url, key) for url in data],
        dtype=data.dtype,
        name=data.name,
    )


@execute_node.register(ops.ExtractUserInfo, (pd.Series, str))
def execute_extract_user_info(op, data, **kwargs):
    def extract_user_info(url):
        url_parts = urlsplit(url)

        username = url_parts.username or ""
        password = url_parts.password or ""

        return f"{username}:{password}"

    if isinstance(data, str):
        return extract_user_info(data)

    return pd.Series(
        [extract_user_info(string) for string in data],
        dtype=data.dtype,
        name=data.name,
    )
