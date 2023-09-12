from __future__ import annotations

import functools
import itertools
import operator

import dask.dataframe as dd
import dask.dataframe.groupby as ddgb
import numpy as np
import pandas as pd
import toolz
from pandas import isnull

import ibis
import ibis.expr.operations as ops
from ibis.backends.dask.core import execute
from ibis.backends.dask.dispatch import execute_node
from ibis.backends.dask.execution.util import (
    TypeRegistrationDict,
    make_selected_obj,
    register_types_to_dispatcher,
)
from ibis.backends.pandas.core import integer_types, scalar_types
from ibis.backends.pandas.execution.strings import (
    execute_json_getitem_series_series,
    execute_json_getitem_series_str_int,
    execute_series_regex_extract,
    execute_series_regex_replace,
    execute_series_regex_search,
    execute_series_right,
    execute_series_string_replace,
    execute_series_translate_scalar_scalar,
    execute_series_translate_scalar_series,
    execute_series_translate_series_scalar,
    execute_series_translate_series_series,
    execute_string_capitalize,
    execute_string_contains,
    execute_string_find,
    execute_string_length_series,
    execute_string_like_series_string,
    execute_string_lower,
    execute_string_lpad,
    execute_string_lstrip,
    execute_string_repeat,
    execute_string_reverse,
    execute_string_rpad,
    execute_string_rstrip,
    execute_string_strip,
    execute_string_upper,
    execute_substring_int_int,
    haystack_to_series_of_lists,
)

DASK_DISPATCH_TYPES: TypeRegistrationDict = {
    ops.StringLength: [((dd.Series,), execute_string_length_series)],
    ops.Substring: [
        (
            (
                dd.Series,
                integer_types,
                (type(None), *integer_types),
            ),
            execute_substring_int_int,
        ),
    ],
    ops.Strip: [((dd.Series,), execute_string_strip)],
    ops.LStrip: [((dd.Series,), execute_string_lstrip)],
    ops.RStrip: [((dd.Series,), execute_string_rstrip)],
    ops.LPad: [
        (
            (
                dd.Series,
                (dd.Series,) + integer_types,
                (dd.Series, str),
            ),
            execute_string_lpad,
        ),
    ],
    ops.RPad: [
        (
            (
                dd.Series,
                (dd.Series,) + integer_types,
                (dd.Series, str),
            ),
            execute_string_rpad,
        ),
    ],
    ops.Reverse: [((dd.Series,), execute_string_reverse)],
    ops.StringReplace: [
        (
            (dd.Series, (dd.Series, str), (dd.Series, str)),
            execute_series_string_replace,
        )
    ],
    ops.Lowercase: [((dd.Series,), execute_string_lower)],
    ops.Uppercase: [((dd.Series,), execute_string_upper)],
    ops.Capitalize: [((dd.Series,), execute_string_capitalize)],
    ops.Repeat: [
        ((dd.Series, (dd.Series,) + integer_types), execute_string_repeat),
    ],
    ops.StringFind: [
        (
            (
                dd.Series,
                (dd.Series, str),
                (dd.Series, type(None)) + integer_types,
                (dd.Series, type(None)) + integer_types,
            ),
            execute_string_find,
        )
    ],
    ops.StringContains: [
        (
            (
                dd.Series,
                (dd.Series, str),
            ),
            execute_string_contains,
        )
    ],
    ops.StringSQLLike: [
        (
            (
                dd.Series,
                str,
                (str, type(None)),
            ),
            execute_string_like_series_string,
        ),
    ],
    ops.RegexSearch: [
        (
            (
                dd.Series,
                str,
            ),
            execute_series_regex_search,
        )
    ],
    ops.RegexExtract: [
        (
            (dd.Series, (dd.Series, str), integer_types),
            execute_series_regex_extract,
        ),
    ],
    ops.RegexReplace: [
        (
            (
                dd.Series,
                str,
                str,
            ),
            execute_series_regex_replace,
        ),
    ],
    ops.Translate: [
        (
            (dd.Series, dd.Series, dd.Series),
            execute_series_translate_series_series,
        ),
        ((dd.Series, dd.Series, str), execute_series_translate_series_scalar),
        ((dd.Series, str, dd.Series), execute_series_translate_scalar_series),
        ((dd.Series, str, str), execute_series_translate_scalar_scalar),
    ],
    ops.StrRight: [((dd.Series, integer_types), execute_series_right)],
    ops.JSONGetItem: [
        ((dd.Series, (str, int)), execute_json_getitem_series_str_int),
        ((dd.Series, dd.Series), execute_json_getitem_series_series),
    ],
}
register_types_to_dispatcher(execute_node, DASK_DISPATCH_TYPES)


@execute_node.register(ops.Substring, dd.Series, dd.Series, integer_types)
def execute_substring_series_int(op, data, start, length, **kwargs):
    return execute_substring_series_series(
        op, data, start, dd.from_array(np.repeat(length, len(start))), **kwargs
    )


@execute_node.register(ops.Substring, dd.Series, integer_types, dd.Series)
def execute_string_substring_int_series(op, data, start, length, **kwargs):
    return execute_substring_series_series(
        op,
        data,
        dd.from_array(np.repeat(start, len(length))),
        length,
        **kwargs,
    )


# TODO - substring - #2553
@execute_node.register(ops.Substring, dd.Series, dd.Series, dd.Series)
def execute_substring_series_series(op, data, start, length, **kwargs):
    end = start + length

    # TODO - this is broken
    start_iter = start.items()
    end_iter = end.items()

    def iterate(value, start_iter=start_iter, end_iter=end_iter):
        _, begin = next(start_iter)
        _, end = next(end_iter)
        if (begin is not None and isnull(begin)) or (end is not None and isnull(end)):
            return None
        return value[begin:end]

    return data.map(iterate)


@execute_node.register(ops.StringConcat, tuple)
def execute_node_string_concat(op, values, **kwargs):
    values = [execute(arg, **kwargs) for arg in values]
    return functools.reduce(operator.add, values)


@execute_node.register(ops.StringSQLLike, ddgb.SeriesGroupBy, str, str)
def execute_string_like_series_groupby_string(op, data, pattern, escape, **kwargs):
    return execute_string_like_series_string(
        op, make_selected_obj(data), pattern, escape, **kwargs
    ).groupby(data.grouper.groupings)


# TODO - aggregations - #2553
@execute_node.register(ops.GroupConcat, dd.Series, str, (dd.Series, type(None)))
def execute_group_concat_series_mask(op, data, sep, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data[mask] if mask is not None else data,
        lambda series, sep=sep: sep.join(series.values),
    )


@execute_node.register(ops.GroupConcat, ddgb.SeriesGroupBy, str, type(None))
def execute_group_concat_series_gb(op, data, sep, _, aggcontext=None, **kwargs):
    custom_group_concat = dd.Aggregation(
        name="custom_group_concat",
        chunk=lambda s: s.apply(list),
        agg=lambda s0: s0.apply(
            lambda chunks: sep.join(
                str(s) for s in itertools.chain.from_iterable(chunks)
            )
        ),
    )
    return data.agg(custom_group_concat)


# TODO - aggregations - #2553
@execute_node.register(ops.GroupConcat, ddgb.SeriesGroupBy, str, ddgb.SeriesGroupBy)
def execute_group_concat_series_gb_mask(op, data, sep, mask, aggcontext=None, **kwargs):
    def method(series, sep=sep):
        return sep.join(series.values.astype(str))

    return aggcontext.agg(
        data,
        lambda data, mask=mask.obj, method=method: method(data[mask[data.index]]),
    )


@execute_node.register(ops.StringAscii, dd.Series)
def execute_string_ascii(op, data, **kwargs):
    output_meta = pd.Series([], dtype=np.dtype("int32"), name=data.name)
    return data.map(ord, meta=output_meta)


@execute_node.register(ops.StringAscii, ddgb.SeriesGroupBy)
def execute_string_ascii_group_by(op, data, **kwargs):
    return execute_string_ascii(op, make_selected_obj(data), **kwargs).groupby(
        data.index
    )


@execute_node.register(ops.RegexSearch, ddgb.SeriesGroupBy, str)
def execute_series_regex_search_gb(op, data, pattern, **kwargs):
    return execute_series_regex_search(
        op,
        make_selected_obj(data),
        getattr(pattern, "obj", pattern),
        **kwargs,
    ).groupby(data.index)


@execute_node.register(ops.RegexExtract, ddgb.SeriesGroupBy, str, integer_types)
def execute_series_regex_extract_gb(op, data, pattern, index, **kwargs):
    return execute_series_regex_extract(
        op, make_selected_obj(data), pattern, index, **kwargs
    ).groupby(data.index)


@execute_node.register(ops.RegexReplace, ddgb.SeriesGroupBy, str, str)
def execute_series_regex_replace_gb(op, data, pattern, replacement, **kwargs):
    return execute_series_regex_replace(
        make_selected_obj(data), pattern, replacement, **kwargs
    ).groupby(data.index)


@execute_node.register(ops.StrRight, ddgb.SeriesGroupBy, integer_types)
def execute_series_right_gb(op, data, nchars, **kwargs):
    return execute_series_right(op, make_selected_obj(data), nchars).groupby(data.index)


@execute_node.register(ops.StringJoin, (dd.Series, str), tuple)
def execute_series_join_scalar_sep(op, sep, args, **kwargs):
    data = [execute(arg, **kwargs) for arg in args]
    return functools.reduce(lambda x, y: x + sep + y, data)


def haystack_to_dask_series_of_lists(haystack, index=None):
    pieces = haystack_to_series_of_lists(haystack, index)
    return dd.from_pandas(pieces, npartitions=1)


@execute_node.register(ops.FindInSet, dd.Series, tuple)
def execute_series_find_in_set(op, needle, haystack, **kwargs):
    def find_in_set(index, elements):
        return ibis.util.safe_index(elements, index)

    haystack = [execute(arg, **kwargs) for arg in haystack]
    return needle.apply(find_in_set, args=(haystack,))


@execute_node.register(ops.FindInSet, ddgb.SeriesGroupBy, list)
def execute_series_group_by_find_in_set(op, needle, haystack, **kwargs):
    pieces = [getattr(piece, "obj", piece) for piece in haystack]
    return execute_series_find_in_set(
        op, make_selected_obj(needle), pieces, **kwargs
    ).groupby(needle.index)


# TODO we need this version not pandas
@execute_node.register(ops.FindInSet, scalar_types, list)
def execute_string_group_by_find_in_set(op, needle, haystack, **kwargs):
    # `list` could contain series, series groupbys, or scalars
    # mixing series and series groupbys is not allowed
    series_in_haystack = [
        type(piece)
        for piece in haystack
        if isinstance(piece, (dd.Series, ddgb.SeriesGroupBy))
    ]

    if not series_in_haystack:
        return ibis.util.safe_index(haystack, needle)

    try:
        (collection_type,) = frozenset(map(type, series_in_haystack))
    except ValueError:
        raise ValueError("Mixing Series and ddgb.SeriesGroupBy is not allowed")

    pieces = haystack_to_dask_series_of_lists(
        [getattr(piece, "obj", piece) for piece in haystack]
    )

    result = pieces.map(toolz.flip(ibis.util.safe_index)(needle))
    if issubclass(collection_type, dd.Series):
        return result

    assert issubclass(collection_type, ddgb.SeriesGroupBy)

    return result.groupby(
        toolz.first(
            piece.grouper.groupings for piece in haystack if hasattr(piece, "grouper")
        )
    )
