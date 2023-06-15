from __future__ import annotations

import datetime
import json
from functools import singledispatch
from math import isfinite
from urllib.parse import urlparse

import rich
from rich.align import Align
from rich.console import Console
from rich.text import Text

import ibis
import ibis.expr.datatypes as dt

# A console with all color/markup disabled, used for `__repr__`
simple_console = Console(force_terminal=False)


def _format_nested(values):
    interactive = ibis.options.repr.interactive
    return [
        rich.pretty.Pretty(
            v,
            max_length=interactive.max_length,
            max_string=interactive.max_string,
            max_depth=interactive.max_depth,
        )
        for v in values
    ]


@singledispatch
def format_values(dtype, values):
    return _format_nested(values)


@format_values.register(dt.Map)
def _(dtype, values):
    return _format_nested([None if v is None else dict(v) for v in values])


@format_values.register(dt.JSON)
def _(dtype, values):
    def try_json(v):
        if v is None:
            return None
        try:
            return json.loads(v)
        except Exception:
            return v

    return _format_nested([try_json(v) for v in values])


@format_values.register(dt.Boolean)
@format_values.register(dt.UUID)
def _(dtype, values):
    return [Text(str(v)) for v in values]


@format_values.register(dt.Decimal)
def _(dtype, values):
    if dtype.scale is not None:
        fmt = f"{{:.{dtype.scale}f}}"
        return [Text.styled(fmt.format(v), "bold cyan") for v in values]
    else:
        # No scale specified, convert to float and repr that way
        return format_values(dt.float64, [float(v) for v in values])


@format_values.register(dt.Integer)
def _(dtype, values):
    return [Text.styled(str(int(v)), "bold cyan") for v in values]


@format_values.register(dt.Floating)
def _(dtype, values):
    floats = [float(v) for v in values]
    # Extract and format all finite floats
    finites = [f for f in floats if isfinite(f)]
    if finites and all(f == 0 or 1e-6 < abs(f) < 1e6 for f in finites):
        strs = [f"{f:f}" for f in finites]
        # Trim matching trailing zeros
        while all(s.endswith("0") for s in strs):
            strs = [s[:-1] for s in strs]
        strs = [s + "0" if s.endswith(".") else s for s in strs]
    else:
        strs = [f"{f:e}" for f in finites]
    # Merge together the formatted finite floats with non-finite values
    iterstrs = iter(strs)
    strs2 = (next(iterstrs) if isfinite(f) else str(f) for f in floats)
    return [Text.styled(s, "bold cyan") for s in strs2]


@format_values.register(dt.Timestamp)
def _(dtype, values):
    if all(v.microsecond == 0 for v in values):
        timespec = "seconds"
    elif all(v.microsecond % 1000 == 0 for v in values):
        timespec = "milliseconds"
    else:
        timespec = "microseconds"
    return [
        Text.styled(v.isoformat(sep=" ", timespec=timespec), "magenta") for v in values
    ]


@format_values.register(dt.Date)
def _(dtype, values):
    dates = [v.date() if isinstance(v, datetime.datetime) else v for v in values]
    return [Text.styled(d.isoformat(), "magenta") for d in dates]


@format_values.register(dt.Time)
def _(dtype, values):
    times = [v.time() if isinstance(v, datetime.datetime) else v for v in values]
    if all(t.microsecond == 0 for t in times):
        timespec = "seconds"
    elif all(t.microsecond % 1000 == 0 for t in times):
        timespec = "milliseconds"
    else:
        timespec = "microseconds"
    return [Text.styled(t.isoformat(timespec=timespec), "magenta") for t in times]


@format_values.register(dt.Interval)
def _(dtype, values):
    return [Text.styled(str(v), "magenta") for v in values]


_str_escapes = str.maketrans(
    {
        "\t": r"[orange3]\t[/]",
        "\r": r"[orange3]\r[/]",
        "\n": r"[orange3]\n[/]",
        "\v": r"[orange3]\v[/]",
        "\f": r"[orange3]\f[/]",
    }
)


@format_values.register(dt.String)
def _(dtype, values):
    max_string = ibis.options.repr.interactive.max_string
    out = []
    for v in values:
        v = str(v)
        if v:
            raw_v = v
            if len(v) > max_string:
                v = v[: max_string - 1] + "…"
            v = v[:max_string]
            # Escape all literal `[` so rich doesn't treat them as markup
            v = v.replace("[", r"\[")
            # Replace ascii escape characters dimmed versions of their repr
            v = v.translate(_str_escapes)
            if not v.isprintable():
                # display all unprintable characters as a dimmed version of
                # their repr
                v = "".join(
                    f"[dim]{repr(c)[1:-1]}[/]" if not c.isprintable() else c for c in v
                )
            url = urlparse(raw_v)
            # check both scheme and netloc to avoid rendering e.g.,
            # `https://` as link
            if url.scheme and url.netloc:
                v = f"[link={raw_v}]{v}[/link]"
            text = Text.from_markup(v, style="green")
        else:
            text = Text.styled("~", "dim")
        out.append(text)
    return out


def format_column(dtype, values):
    import pandas as pd

    null_str = Text.styled("NULL", style="dim")
    if dtype.is_floating():
        # We don't want to treat `nan` as `NULL` for floating point types
        def isnull(x):
            return x is None or x is pd.NA

    else:

        def isnull(x):
            o = pd.isna(x)
            # pd.isna broadcasts if `x` is an array
            return o if isinstance(o, bool) else False

    nonnull = [v for v in values if not isnull(v)]
    if nonnull:
        formatted = format_values(dtype, nonnull)
        next_f = iter(formatted).__next__
        out = [null_str if isnull(v) else next_f() for v in values]
    else:
        out = [null_str] * len(values)

    try:
        max_width = max(map(len, out))
    except Exception:  # noqa: BLE001
        max_width = None
        min_width = 20
    else:
        if dtype.is_string():
            min_width = min(20, max_width)
        else:
            min_width = max_width

    return out, min_width, max_width


def format_dtype(dtype):
    max_string = ibis.options.repr.interactive.max_string
    strtyp = str(dtype)
    if len(strtyp) > max_string:
        strtyp = strtyp[: max_string - 1] + "…"
    return Text.styled(strtyp, "dim")


def to_rich_table(table, console_width=None):
    if console_width is None:
        console_width = float("inf")

    orig_ncols = len(table.columns)

    max_columns = ibis.options.repr.interactive.max_columns
    if console_width == float("inf"):
        # When there's infinite display space, only subset columns
        # if an explicit limit has been set.
        if max_columns and max_columns < orig_ncols:
            table = table.select(*table.columns[:max_columns])
    else:
        # Determine the maximum subset of columns that *might* fit in the
        # current console. Note that not every column here may actually fit
        # later on once we know the repr'd width of the data.
        computed_cols = []
        remaining = console_width - 1  # 1 char for left boundary
        for c in table.columns:
            needed = len(c) + 3  # padding + 1 char for right boundary
            if (
                needed < remaining or not computed_cols
            ):  # always select at least one col
                computed_cols.append(c)
                remaining -= needed
            else:
                break
        if max_columns not in (0, None):
            # If an explicit limit on max columns is set, apply it
            computed_cols = computed_cols[:max_columns]
        if orig_ncols > len(computed_cols):
            table = table.select(*computed_cols)

    # Compute the data and return a pandas dataframe
    nrows = ibis.options.repr.interactive.max_rows
    result = table.limit(nrows + 1).to_pyarrow()

    # Now format the columns in order, stopping if the console width would
    # be exceeded.
    col_info = []
    col_data = []
    formatted_dtypes = []
    remaining = console_width - 1  # 1 char for left boundary
    for name, dtype in table.schema().items():
        formatted, min_width, max_width = format_column(
            dtype, result[name].to_pylist()[:nrows]
        )
        dtype_str = format_dtype(dtype)
        if ibis.options.repr.interactive.show_types and not isinstance(
            dtype, (dt.Struct, dt.Map, dt.Array)
        ):
            # Don't truncate non-nested dtypes
            min_width = max(min_width, len(dtype_str))

        min_width = max(min_width, len(name))
        if max_width is not None:
            max_width = max(min_width, max_width)
        needed = min_width + 3  # padding + 1 char for right boundary
        if needed < remaining:
            col_info.append((name, dtype, min_width, max_width))
            col_data.append(formatted)
            formatted_dtypes.append(dtype_str)
            remaining -= needed
        elif not col_info:
            # Always pretty print at least one column. If only one column, we
            # truncate to fill the available space, leaving room for the
            # ellipsis & framing.
            min_width = remaining - 3  # 3 for framing
            if orig_ncols > 1:
                min_width -= 4  # 4 for ellipsis
            col_info.append((name, dtype, min_width, min_width))
            col_data.append(formatted)
            formatted_dtypes.append(dtype_str)
            break
        else:
            if remaining < 4:
                # Not enough space for ellipsis column, drop previous column
                col_info.pop()
                col_data.pop()
                formatted_dtypes.pop()
            break

    # rich's column width computations are super buggy and can result in tables
    # that are much wider than the available console space. To work around this
    # for now we manually compute all column widths rather than letting rich
    # figure it out for us.
    columns_truncated = orig_ncols > len(col_info)
    col_widths = {}
    if console_width == float("inf"):
        # Always use the max_width if there's infinite console space
        for name, _, _, max_width in col_info:
            col_widths[name] = max_width
    else:
        # Allocate the remaining space evenly between the flexible columns
        flex_cols = []
        remaining = console_width - 1
        if columns_truncated:
            remaining -= 4
        for name, _, min_width, max_width in col_info:
            remaining -= min_width + 3
            col_widths[name] = min_width
            if min_width != max_width:
                flex_cols.append((name, max_width))

        while True:
            next_flex_cols = []
            for name, max_width in flex_cols:
                if remaining:
                    remaining -= 1
                    if max_width is not None:
                        col_widths[name] += 1
                    if max_width is None or col_widths[name] < max_width:
                        next_flex_cols.append((name, max_width))
                else:
                    break
            if not next_flex_cols:
                break

    rich_table = rich.table.Table(padding=(0, 1, 0, 1))

    # Configure the columns on the rich table.
    for name, dtype, _, max_width in col_info:
        rich_table.add_column(
            Align(name, align="left"),
            justify="right" if dtype.is_numeric() else "left",
            vertical="middle",
            width=None if max_width is None else col_widths[name],
            min_width=None if max_width is not None else col_widths[name],
            no_wrap=max_width is not None,
        )

    # If the columns are truncated, add a trailing ellipsis column
    if columns_truncated:
        rich_table.add_column(
            Align("…", align="left"),
            justify="left",
            vertical="middle",
            width=1,
            min_width=1,
            no_wrap=True,
        )

        def add_row(*args, **kwargs):
            rich_table.add_row(*args, Align("[dim]…[/]", align="left"), **kwargs)

    else:
        add_row = rich_table.add_row

    if ibis.options.repr.interactive.show_types:
        add_row(
            *(Align(s, align="left") for s in formatted_dtypes),
            end_section=True,
        )

    for row in zip(*col_data):
        add_row(*row)

    # If the rows are truncated, add a trailing ellipsis row
    if len(result) > nrows:
        rich_table.add_row(
            *(Align("[dim]…[/]", align=c.justify) for c in rich_table.columns)
        )

    return rich_table
