import six

import pandas as pd

from pandas.core.groupby import SeriesGroupBy

import ibis.expr.operations as ops

from ibis.pandas.dispatch import execute_node
from ibis.pandas.core import integer_types


@execute_node.register(ops.StringLength, pd.Series)
def execute_string_length_series(op, data, **kwargs):
    return data.str.len()


@execute_node.register(
    ops.Substring,
    pd.Series,
    (pd.Series,) + integer_types,
    (pd.Series,) + integer_types
)
def execute_string_substring(op, data, start, length, **kwargs):
    return data.str[start:start + length]


@execute_node.register(ops.Strip, pd.Series)
def execute_string_strip(op, data, **kwargs):
    return data.str.strip()


@execute_node.register(ops.LStrip, pd.Series)
def execute_string_lstrip(op, data, **kwargs):
    return data.str.lstrip()


@execute_node.register(ops.RStrip, pd.Series)
def execute_string_rstrip(op, data, **kwargs):
    return data.str.rstrip()


@execute_node.register(
    ops.LPad,
    pd.Series,
    (pd.Series,) + integer_types,
    (pd.Series,) + six.string_types
)
def execute_string_lpad(op, data, length, pad, **kwargs):
    return data.str.pad(length, side='left', fillchar=pad)


@execute_node.register(
    ops.RPad,
    pd.Series,
    (pd.Series,) + integer_types,
    (pd.Series,) + six.string_types
)
def execute_string_rpad(op, data, length, pad, **kwargs):
    return data.str.pad(length, side='right', fillchar=pad)


@execute_node.register(ops.Reverse, pd.Series)
def execute_string_reverse(op, data, **kwargs):
    return data.str[::-1]


@execute_node.register(ops.Lowercase, pd.Series)
def execute_string_lower(op, data, **kwargs):
    return data.str.lower()


@execute_node.register(ops.Uppercase, pd.Series)
def execute_string_upper(op, data, **kwargs):
    return data.str.upper()


@execute_node.register(ops.Capitalize, pd.Series)
def execute_string_capitalize(op, data, **kwargs):
    return data.str.capitalize()


@execute_node.register(ops.Repeat, pd.Series, (pd.Series,) + integer_types)
def execute_string_repeat(op, data, times, **kwargs):
    return data.str.repeat(times)


@execute_node.register(
    ops.StringFind,
    pd.Series,
    (pd.Series,) + six.string_types,
    (pd.Series, type(None)) + integer_types,
    (pd.Series, type(None)) + integer_types,
)
def execute_string_contains(op, data, needle, start, end, **kwargs):
    return data.str.find(needle, start, end)


@execute_node.register(
    ops.StringSQLLike,
    pd.Series,
    (pd.Series,) + six.string_types,
)
def execute_string_like(op, data, pattern, **kwargs):
    return data.str.contains(pattern, regex=True)


@execute_node.register(
    ops.GroupConcat,
    pd.Series, six.string_types, (pd.Series, type(None))
)
def execute_group_concat_series_mask(
    op, data, sep, mask, context=None, **kwargs
):
    return context.agg(data[mask] if mask is not None else data, sep.join)


@execute_node.register(
    ops.GroupConcat, SeriesGroupBy, six.string_types, type(None)
)
def execute_group_concat_series_gb(op, data, sep, _, context=None, **kwargs):
    return context.agg(data, lambda data, sep=sep: sep.join(data.astype(str)))


@execute_node.register(
    ops.GroupConcat, SeriesGroupBy, six.string_types, SeriesGroupBy
)
def execute_group_concat_series_gb_mask(
    op, data, sep, mask, context=None, **kwargs
):
    method = lambda x, sep=sep: sep.join(x.astype(str))  # noqa: E731
    return context.agg(
        data,
        lambda data, mask=mask.obj, method=method: method(
            data[mask[data.index]]
        ),
    )


@execute_node.register(ops.StringAscii, pd.Series)
def execute_string_ascii(op, data, **kwargs):
    return data.map(ord)


@execute_node.register(ops.StringAscii, SeriesGroupBy)
def execute_string_ascii_group_by(op, data, **kwargs):
    return execute_string_ascii(
        op, data, **kwargs
    ).groupby(data.grouper.groupings)


@execute_node.register(
    ops.RegexSearch, pd.Series, (pd.Series,) + six.string_types
)
def execute_series_regex_search(op, data, pattern, **kwargs):
    return data.str.contains(pattern, regex=True)


@execute_node.register(
    ops.RegexSearch, SeriesGroupBy, (SeriesGroupBy,) + six.string_types
)
def execute_series_regex_search_gb(op, data, pattern, **kwargs):
    return execute_series_regex_search(
        op, data, getattr(pattern, 'obj', pattern), **kwargs
    ).groupby(data.grouper.groupings)


@execute_node.register(
    ops.RegexExtract,
    pd.Series,
    (pd.Series,) + six.string_types,
    (pd.Series,) + integer_types,
)
def execute_series_regex_extract(op, data, pattern, index, **kwargs):
    extracted = data.str.extractall(pattern).iloc[:, index].reset_index(
        drop=True, level=-1
    )
    return extracted.reindex(data.index)


@execute_node.register(
    ops.RegexExtract,
    SeriesGroupBy,
    (SeriesGroupBy,) + six.string_types,
    (SeriesGroupBy,) + integer_types,
)
def execute_series_regex_extract_gb(op, data, pattern, index, **kwargs):
    return execute_series_regex_extract(
        op,
        getattr(data, 'obj', data),
        getattr(pattern, 'obj', pattern),
        getattr(index, 'obj', index),
        **kwargs
    ).groupby(data.grouper.groupings)


@execute_node.register(
    ops.RegexReplace,
    pd.Series,
    (pd.Series,) + six.string_types,
    (pd.Series,) + six.string_types,
)
def execute_series_regex_replace(op, data, pattern, replacement, **kwargs):
    return data.str.replace(pattern, replacement)


@execute_node.register(
    ops.RegexReplace,
    SeriesGroupBy,
    (SeriesGroupBy,) + six.string_types,
    (SeriesGroupBy,) + six.string_types,
)
def execute_series_regex_replace_gb(op, data, pattern, replacement, **kwargs):
    return execute_series_regex_replace(
        getattr(data, 'obj', data),
        getattr(pattern, 'obj', pattern),
        getattr(replacement, 'obj', replacement),
    ).groupby(data.grouper.groupings)
