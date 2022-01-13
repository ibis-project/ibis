from typing import Optional

import pandas as pd
import pandas.testing as tm
import pytest

import ibis
import ibis.expr.operations as ops
from ibis.backends.pyspark.timecontext import combine_time_context
from ibis.expr.scope import Scope
from ibis.expr.timecontext import adjust_context
from ibis.expr.types import TimeContext


def test_table_with_timecontext(client):
    table = client.table('time_indexed_table')
    context = (pd.Timestamp('20170102'), pd.Timestamp('20170103'))
    result = table.execute(timecontext=context)
    expected = table.execute()
    expected = expected[expected.time.between(*context)]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('contexts', 'expected'),
    [
        (
            [
                (pd.Timestamp('20200102'), pd.Timestamp('20200103')),
                (pd.Timestamp('20200101'), pd.Timestamp('20200106')),
            ],
            (pd.Timestamp('20200101'), pd.Timestamp('20200106')),
        ),  # superset
        (
            [
                (pd.Timestamp('20200101'), pd.Timestamp('20200103')),
                (pd.Timestamp('20200102'), pd.Timestamp('20200106')),
            ],
            (pd.Timestamp('20200101'), pd.Timestamp('20200106')),
        ),  # overlap
        (
            [
                (pd.Timestamp('20200101'), pd.Timestamp('20200103')),
                (pd.Timestamp('20200202'), pd.Timestamp('20200206')),
            ],
            (pd.Timestamp('20200101'), pd.Timestamp('20200206')),
        ),  # non-overlap
        (
            [(pd.Timestamp('20200101'), pd.Timestamp('20200103')), None],
            (pd.Timestamp('20200101'), pd.Timestamp('20200103')),
        ),  # None in input
        ([None], None),  # None for all
        (
            [
                (pd.Timestamp('20200102'), pd.Timestamp('20200103')),
                (pd.Timestamp('20200101'), pd.Timestamp('20200106')),
                (pd.Timestamp('20200109'), pd.Timestamp('20200110')),
            ],
            (pd.Timestamp('20200101'), pd.Timestamp('20200110')),
        ),  # complex
    ],
)
def test_combine_time_context(contexts, expected):
    assert combine_time_context(contexts) == expected


def test_adjust_context_scope(client):
    """Test that `adjust_context` has access to `scope` by default."""
    table = client.table('time_indexed_table')

    # WindowOp is the only context-adjusted node that the PySpark backend
    # can compile.
    # Override its `adjust_context` function with a function that imply
    # checks that `scope` has been passed in:

    @adjust_context.register(ops.WindowOp)
    def adjust_context_window_check_scope(
        op: ops.WindowOp,
        timecontext: TimeContext,
        scope: Optional[Scope] = None,
    ) -> TimeContext:
        """Confirms that `scope` is passed in."""
        assert scope is not None
        return timecontext

    # Do an operation that will trigger context adjustment
    # on a WindowOp
    expr = table.mutate(
        win=(
            table['value']
            .count()
            .over(
                ibis.window(
                    ibis.interval(hours=1),
                    0,
                    order_by='time',
                    group_by='key',
                )
            )
        )
    )
    context = (pd.Timestamp('20170105'), pd.Timestamp('20170111'))
    expr.execute(timecontext=context)
