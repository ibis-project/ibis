from __future__ import annotations

import pandas as pd
import pytest
from dask.dataframe.utils import tm

import ibis

dd = pytest.importorskip("dask.dataframe")


def test_table_from_dataframe(dataframe, ibis_table, con):
    t = con.from_dataframe(dataframe)
    result = t.execute()
    expected = ibis_table.execute()
    tm.assert_frame_equal(result, expected)

    t = con.from_dataframe(dataframe, name="foo")
    expected = ibis_table.execute()
    tm.assert_frame_equal(result, expected)

    t = con.from_dataframe(dataframe, name="foo", client=con)
    expected = ibis_table.execute()
    tm.assert_frame_equal(result, expected)


def test_array_literal_from_series(con):
    values = [1, 2, 3, 4]
    s = dd.from_pandas(pd.Series(values), npartitions=1)
    expr = ibis.array(s)

    assert expr.equals(ibis.array(values))
    assert con.execute(expr) == pytest.approx([1, 2, 3, 4])


def test_execute_parameter_only(con):
    param = ibis.param("int64")
    result = con.execute(param, params={param.op(): 42})
    assert result == 42
