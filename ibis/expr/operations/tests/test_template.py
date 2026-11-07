from __future__ import annotations

import pytest

import ibis
import ibis.expr.datatypes as dt
from ibis.expr.operations.template import TemplateSQLValue
from ibis.tstring import t


def test_set_backend(con, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", None)
    ibis.set_backend(con)
    assert ibis.get_backend() is con


@pytest.mark.parametrize(
    "five",
    [
        pytest.param(5, id="int"),
        pytest.param(ibis.literal(5), id="literal"),
        pytest.param(ibis.literal(5).op(), id="value"),
    ],
)
def test_scalar(five):  # noqa: ARG001
    template = t("{five} + 4")
    op = TemplateSQLValue.from_template(template)
    assert op.dialect == "duckdb"
    assert op.shape.is_scalar()
    assert op.dtype == dt.int32


def test_column():
    col = ibis.memtable({"c": ["a", "b"]}).c  # noqa: F841
    template = t("{col} || 'c'")
    op = TemplateSQLValue.from_template(template)
    assert op.dialect == "duckdb"
    assert op.shape.is_columnar()
    assert op.dtype == dt.string


def test_dialect():
    # When parsed in sqlite dialect, REAL is interpreted as float64,
    # in default duckdb dialect, REAL is interpreted as float32
    five = ibis.literal(5)  # noqa: F841
    template = t("CAST({five} AS REAL)")

    op = TemplateSQLValue.from_template(template, dialect="sqlite")
    assert op.dialect == "sqlite"
    assert op.shape.is_scalar()
    assert op.dtype == dt.float64

    op = TemplateSQLValue.from_template(template)
    assert op.dialect == "duckdb"
    assert op.shape.is_scalar()
    assert op.dtype == dt.float32


def test_no_interpolations():
    template = t("5 + 4")
    op = TemplateSQLValue.from_template(template)
    assert op.dialect == "duckdb"
    assert op.shape.is_scalar()
    assert op.dtype == dt.int32


def test_select_errors():
    five = ibis.literal(5)  # noqa: F841
    template = t("SELECT {five}")
    with pytest.raises(TypeError, match=r".*SELECT CAST\(NULL AS TINYINT\)"):
        TemplateSQLValue.from_template(template)


def test_api():
    five = ibis.literal(5)  # noqa: F841
    template = t("{five} + 4")
    expr = ibis.sql_value(template)
    assert isinstance(expr, ibis.Value)
    assert expr.type().is_integer()
    assert expr.type().nullable


def test_name():
    five = ibis.literal(5)  # noqa: F841
    template = t("{five} + 4")
    expr = ibis.sql_value(template)
    actual = expr.get_name()
    assert actual
    # explicitly not tested
    # expected_name = "TemplateSQL((), (5,))"
    # assert actual == expected_name
