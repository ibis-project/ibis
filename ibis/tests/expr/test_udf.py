from __future__ import annotations

import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import _
from ibis.common.annotations import ValidationError
from ibis.common.deferred import Deferred


@pytest.fixture
def table():
    return ibis.table(
        [
            ("a", "int8"),
            ("b", "string"),
            ("c", "bool"),
        ],
        name="test",
    )


@pytest.mark.parametrize(
    ("klass", "output_type"),
    [
        (ops.ElementWiseVectorizedUDF, ir.IntegerColumn),
        (ops.ReductionVectorizedUDF, ir.IntegerScalar),
        (ops.AnalyticVectorizedUDF, ir.IntegerColumn),
    ],
)
def test_vectorized_udf_operations(table, klass, output_type):
    udf = klass(
        func=lambda a, b, c: a,
        func_args=[table.a, table.b, table.c],
        input_type=[dt.int8(), dt.string(), dt.boolean()],
        return_type=dt.int8(),
    )
    assert udf.func_args[0] == table.a.op()
    assert udf.func_args[1] == table.b.op()
    assert udf.func_args[2] == table.c.op()
    assert udf.input_type == (dt.int8(), dt.string(), dt.boolean())
    assert udf.return_type == dt.int8()

    assert isinstance(udf.to_expr(), output_type)

    with pytest.raises(ValidationError):
        # wrong function type
        klass(
            func=1,
            func_args=[ibis.literal(1), table.b, table.c],
            input_type=[dt.int8(), dt.string(), dt.boolean()],
            return_type=dt.int8(),
        )

    with pytest.raises(ValidationError):
        # scalar type instead of column type
        klass(
            func=lambda a, b, c: a,
            func_args=[ibis.literal(1), table.b, table.c],
            input_type=[dt.int8(), dt.string(), dt.boolean()],
            return_type=dt.int8(),
        )

    with pytest.raises(ValidationError):
        # wrong input type
        klass(
            func=lambda a, b, c: a,
            func_args=[ibis.literal(1), table.b, table.c],
            input_type="int8",
            return_type=dt.int8(),
        )

    with pytest.raises(ValidationError):
        # wrong return type
        klass(
            func=lambda a, b, c: a,
            func_args=[ibis.literal(1), table.b, table.c],
            input_type=[dt.int8(), dt.string(), dt.boolean()],
            return_type=table,
        )


@pytest.mark.parametrize(
    "dec",
    [
        pytest.param(ibis.udf.scalar.builtin, id="scalar-builtin"),
        pytest.param(ibis.udf.scalar.pandas, id="scalar-pandas"),
        pytest.param(ibis.udf.scalar.pyarrow, id="scalar-pyarrow"),
        pytest.param(ibis.udf.scalar.python, id="scalar-python"),
        pytest.param(ibis.udf.agg.builtin, id="agg-builtin"),
    ],
)
def test_udf_from_annotations(dec, table):
    @dec
    def myfunc(x: int, y: str) -> float:
        ...

    assert myfunc(table.a, table.b).type().is_floating()

    with pytest.raises(ValidationError):
        # Wrong arg types
        myfunc(table.b, table.a)


@pytest.mark.parametrize(
    "dec",
    [
        pytest.param(ibis.udf.scalar.builtin, id="scalar-builtin"),
        pytest.param(ibis.udf.scalar.pandas, id="scalar-pandas"),
        pytest.param(ibis.udf.scalar.pyarrow, id="scalar-pyarrow"),
        pytest.param(ibis.udf.scalar.python, id="scalar-python"),
        pytest.param(ibis.udf.agg.builtin, id="agg-builtin"),
    ],
)
def test_udf_from_sig(dec, table):
    @dec(signature=((int, str), float))
    def myfunc(x, y):
        ...

    assert myfunc(table.a, table.b).type().is_floating()

    with pytest.raises(ValidationError):
        # Wrong arg types
        myfunc(table.b, table.a)


@pytest.mark.parametrize(
    "dec",
    [
        pytest.param(ibis.udf.scalar.builtin, id="scalar-builtin"),
        pytest.param(ibis.udf.scalar.pandas, id="scalar-pandas"),
        pytest.param(ibis.udf.scalar.pyarrow, id="scalar-pyarrow"),
        pytest.param(ibis.udf.scalar.python, id="scalar-python"),
        pytest.param(ibis.udf.agg.builtin, id="agg-builtin"),
    ],
)
def test_udf_deferred(dec, table):
    @dec
    def myfunc(x: int) -> int:
        ...

    expr = myfunc(_.a)
    assert isinstance(expr, Deferred)
    assert repr(expr) == "myfunc(_.a)"
    assert expr.resolve(table).equals(myfunc(table.a))
