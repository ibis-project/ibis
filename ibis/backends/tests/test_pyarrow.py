from __future__ import annotations

import math

import pytest

import ibis
from ibis.backends.tests.conftest import NAN_TREATED_AS_NULL

pa = pytest.importorskip("pyarrow")


@NAN_TREATED_AS_NULL
@pytest.mark.notimpl(
    "exasol", reason="Exasol driver can't handle NaNs during memtable registration"
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
    assert t.schema()["f"] == ibis.dtype("float64")

    def make_comparable(vals):
        return {"nan" if (isinstance(v, float) and math.isnan(v)) else v for v in vals}

    n_nan = con.execute(t.f.isnan().sum())
    n_null = con.execute(t.f.isnull().sum())
    assert (n_nan, n_null) == (1, 1)

    result = make_comparable(con.to_pyarrow(t.f).to_pylist())
    expected = make_comparable(inp)
    assert result == expected
