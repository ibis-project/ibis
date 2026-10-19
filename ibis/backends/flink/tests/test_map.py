from __future__ import annotations

import pytest

import ibis
from ibis.backends.tests.errors import Py4JJavaError
from ibis.conftest import IS_PYFLINK_1X


@pytest.mark.notyet(
    ["flink"],
    condition=IS_PYFLINK_1X,
    raises=Py4JJavaError,
    # the map is fine; Flink 1.x's Calcite can't type the `CAST(ROW(0) AS
    # ROW(...))` default when deriving COALESCE's operand types
    # (UnsupportedOperationException: SqlBasicCall: ROW(0)), the same
    # limitation as `test_field_access_after_case`
    reason="struct literal as a COALESCE operand",
)
def test_map_get_struct_value_with_default(con):
    # struct entries of a map literal are spelled `ROW(CAST(v AS t), ...)` so
    # that Flink 1.x can type them when the map is indexed; make sure that
    # also holds with an explicit default, which goes through COALESCE
    m = ibis.map(ibis.array(["a"]), ibis.array([ibis.struct(dict(x=1))]))
    default = ibis.struct(dict(x=0))

    assert con.execute(m.get("a", default)) == con.execute(ibis.struct(dict(x=1)))
    assert con.execute(m.get("b", default)) == con.execute(default)
