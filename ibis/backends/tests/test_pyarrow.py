from __future__ import annotations

import math

import pytest

import ibis

pa = pytest.importorskip("pyarrow")


@pytest.mark.notimpl(
    "sqlite",
    "During memtable registration, the pa.Table is converted to a pandas DataFrame, losing NaN info",
)
@pytest.mark.parametrize(
    "method",
    [
        pytest.param(lambda pa_arr: pa.table({"f": pa_arr}), id="pa_table"),
        pytest.param(
            lambda pa_arr: {"f": pa_arr},
            id="dict_of_pa_arrays",
            marks=pytest.mark.xfail(
                # https://github.com/ibis-project/ibis/issues/11700
                reason="During ops.InMemoryTable creation, we go through pd.DataFrame, losing NaN info"
            ),
        ),
    ],
)
def test_nans_roundtrip(con, method):
    inp = [1.0, float("nan"), None]
    t = ibis.memtable(method(pa.array(inp)))

    def make_comparable(vals):
        return {"nan" if (isinstance(v, float) and math.isnan(v)) else v for v in vals}

    result = make_comparable(con.to_pyarrow(t.f).to_pylist())
    expected = make_comparable(inp)
    assert result == expected
