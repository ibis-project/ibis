from __future__ import annotations

import pandas as pd
import pandas.testing as tm
import pytest

import ibis
import ibis.expr.operations as ops
from ibis.backends.base.df.timecontext import adjust_context

pytest.importorskip("pyspark")

from ibis.backends.pyspark.compiler import (  # noqa: E402
    compile_window_function,
    compiles,
)
from ibis.backends.pyspark.timecontext import combine_time_context  # noqa: E402


def test_table_with_timecontext(con):
    table = con.table("time_indexed_table")
    context = (pd.Timestamp("20170102"), pd.Timestamp("20170103"))
    result = table.execute(timecontext=context)
    expected = table.execute()
    expected = expected[expected.time.between(*context)]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("contexts", "expected"),
    [
        (
            [
                (pd.Timestamp("20200102"), pd.Timestamp("20200103")),
                (pd.Timestamp("20200101"), pd.Timestamp("20200106")),
            ],
            (pd.Timestamp("20200101"), pd.Timestamp("20200106")),
        ),  # superset
        (
            [
                (pd.Timestamp("20200101"), pd.Timestamp("20200103")),
                (pd.Timestamp("20200102"), pd.Timestamp("20200106")),
            ],
            (pd.Timestamp("20200101"), pd.Timestamp("20200106")),
        ),  # overlap
        (
            [
                (pd.Timestamp("20200101"), pd.Timestamp("20200103")),
                (pd.Timestamp("20200202"), pd.Timestamp("20200206")),
            ],
            (pd.Timestamp("20200101"), pd.Timestamp("20200206")),
        ),  # non-overlap
        (
            [(pd.Timestamp("20200101"), pd.Timestamp("20200103")), None],
            (pd.Timestamp("20200101"), pd.Timestamp("20200103")),
        ),  # None in input
        ([None], None),  # None for all
        (
            [
                (pd.Timestamp("20200102"), pd.Timestamp("20200103")),
                (pd.Timestamp("20200101"), pd.Timestamp("20200106")),
                (pd.Timestamp("20200109"), pd.Timestamp("20200110")),
            ],
            (pd.Timestamp("20200101"), pd.Timestamp("20200110")),
        ),  # complex
    ],
)
def test_combine_time_context(contexts, expected):
    assert combine_time_context(contexts) == expected


def test_adjust_context_scope(con):
    """Test that `adjust_context` has access to `scope` by default."""
    table = con.table("time_indexed_table")

    # Window is the only context-adjusted node that the PySpark backend
    # can compile. Ideally we would test the context adjustment logic for
    # Window itself, but building this test like that would unfortunately
    # affect other tests that involve Window.
    # To avoid that, we'll create a dummy subclass of Window and build the
    # test around that.

    class CustomWindowFunction(ops.WindowFunction):
        pass

    # Tell the Spark backend compiler it should compile CustomWindow just
    # like Window
    compiles(CustomWindowFunction)(compile_window_function)

    # Create an `adjust_context` function for this subclass that simply checks
    # that `scope` is passed in.
    @adjust_context.register(CustomWindowFunction)
    def adjust_context_window_check_scope(op, scope, timecontext):
        """Confirms that `scope` is passed in."""
        assert scope is not None
        return timecontext

    # Do an operation that will trigger context adjustment
    # on a CustomWindow
    value_count = table["value"].count()
    window = ibis.window(
        ibis.interval(hours=1),
        0,
        order_by="time",
        group_by="key",
    )
    frame = window.bind(table)

    # the argument needs to be pull out from the alias
    # any extensions must do the same
    value_count_over_win = CustomWindowFunction(value_count, frame).to_expr()

    expr = table.mutate(value_count_over_win=value_count_over_win)

    context = (pd.Timestamp("20170105"), pd.Timestamp("20170111"))
    expr.execute(timecontext=context)
