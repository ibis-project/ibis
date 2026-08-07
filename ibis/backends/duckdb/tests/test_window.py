from __future__ import annotations

from datetime import datetime

import ibis


def test_compound_preceding_interval():
    table = ibis.memtable(
        {
            "id": [1, 1, 1],
            "date": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 2, 1),
            ],
            "value": [10.0, 20.0, 30.0],
        }
    )
    window = ibis.window(
        group_by="id",
        order_by="date",
        between=(ibis.interval(months=-1, days=1), 0),
    )
    expression = table.mutate(rolling_sum=table.value.sum().over(window)).order_by(
        "date"
    )

    result = ibis.duckdb.connect().execute(expression)

    assert result.rolling_sum.tolist() == [10.0, 30.0, 50.0]
