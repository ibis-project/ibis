import pandas as pd
import pandas.util.testing as tm
import pytest

import ibis
import ibis.expr.operations as ops
from ibis.pandas.client import PandasClient, PandasTable
from ibis.pandas.dispatch import (
    compute_time_context,
    is_computable_input,
    pre_execute,
)

pytestmark = pytest.mark.pandas


def test_execute_with_timecontext(t):
    # define a pre_execute function that trim data accroding to timecontext
    @pre_execute.register(PandasTable, PandasClient)
    def execute_table_with_timecontext(
        op, client, scope=None, timecontext=None, **kwargs
    ):
        # retreive df from dict
        df = client.dictionary[op.name]
        try:
            begin, end = map(pd.to_datetime, timecontext)
        except ValueError:
            return {op: df}
        # filter time col with time context
        time_col = "plain_datetimes_naive"
        return {op: df[df[time_col] >= begin][df[time_col] < end]}

    expr = t
    # define a time context for time-series data
    context = ("20170101", "20170103")

    # without time context, execute produces every row
    df_all = expr.execute()
    assert len(df_all["plain_datetimes_naive"]) == 3

    # with context set, execute produces only rows within context
    df_within_context = expr.execute(timecontext=context)
    assert len(df_within_context["plain_datetimes_naive"]) == 1

    del pre_execute.funcs[(PandasTable, PandasClient)]
    pre_execute.reorder()
    pre_execute._cache.clear()


merge_asof_minversion = pytest.mark.skipif(
    pd.__version__ < '0.19.2',
    reason="at least pandas-0.19.2 required for merge_asof",
)


@merge_asof_minversion
def test_context_adjustment_asof_join(
    time_keyed_left, time_keyed_right, time_keyed_df1, time_keyed_df2
):
    # pre_execute for trimming table data
    @pre_execute.register(PandasTable, PandasClient)
    def execute_table_with_timecontext(
        op, client, scope=None, timecontext=None, **kwargs
    ):
        df = client.dictionary[op.name]
        try:
            begin, end = map(pd.to_datetime, timecontext)
        except ValueError:
            return {op: df}
        time_col = "time"
        return {op: df[df[time_col] >= begin][df[time_col] < end]}

    # define a custom context adjustment rule for asof_join
    @compute_time_context.register(ops.AsOfJoin)
    def adjust_context_asof_join(op, scope=None, timecontext=None, **kwargs):
        new_timecontexts = [
            timecontext for arg in op.inputs if is_computable_input(arg)
        ]
        # right table should look back or forward
        try:
            begin, end = map(pd.to_datetime, timecontext)
            tolerance = op.tolerance
            if tolerance is not None:
                timedelta = pd.Timedelta(-tolerance.op().right.op().value)
                if timedelta <= pd.Timedelta(0):
                    new_begin = begin + timedelta
                    new_end = end
                else:
                    new_begin = begin
                    new_end = end + timedelta
            # right table is the second node in children
            new_timecontexts[1] = (new_begin, new_end)
        except ValueError:
            pass
        finally:
            return new_timecontexts

    expr = time_keyed_left.asof_join(
        time_keyed_right, 'time', by='key', tolerance=2 * ibis.interval(days=1)
    )[time_keyed_left, time_keyed_right.other_value]
    context = (3, 4)
    result = expr.execute(timecontext=context)

    trimmed_df1 = time_keyed_df1[
        time_keyed_df1["time"] >= pd.to_datetime(context[0])
    ][time_keyed_df1["time"] < pd.to_datetime(context[1])]
    trimmed_df2 = time_keyed_df2[
        time_keyed_df2["time"] >= pd.to_datetime(context[0] - 2)
    ][time_keyed_df2["time"] < pd.to_datetime(context[1])]
    expected = pd.merge_asof(
        trimmed_df1,
        trimmed_df2,
        on='time',
        by='key',
        tolerance=pd.Timedelta('2D'),
    )
    tm.assert_frame_equal(result, expected)
