from __future__ import annotations

import ibis


def test_epoch_seconds_truncates_subsecond(con):
    # ``epoch_seconds`` is typed as an integer, so a timestamp with a
    # sub-second component must not retain a fractional part in the result.
    # https://github.com/ibis-project/ibis/issues/11928
    expr = ibis.timestamp("2020-01-01 00:00:00.5").epoch_seconds()
    assert expr.type().is_integer()
    # cast to float so that a retained fractional part would be visible
    assert con.execute(expr.cast("float64")) == 1577836800.0
