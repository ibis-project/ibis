import pytest

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir


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
    assert udf.func_args[0].equals(table.a)
    assert udf.func_args[1].equals(table.b)
    assert udf.func_args[2].equals(table.c)
    assert udf.input_type == [dt.int8(), dt.string(), dt.boolean()]
    assert udf.return_type == dt.int8()

    factory = udf.output_type()
    expr = factory(udf)
    assert isinstance(expr, output_type)

    with pytest.raises(com.IbisTypeError):
        # wrong function type
        klass(
            func=1,
            func_args=[ibis.literal(1), table.b, table.c],
            input_type=[dt.int8(), dt.string(), dt.boolean()],
            return_type=dt.int8(),
        )

    with pytest.raises(com.IbisTypeError):
        # scalar type instead of column type
        klass(
            func=lambda a, b, c: a,
            func_args=[ibis.literal(1), table.b, table.c],
            input_type=[dt.int8(), dt.string(), dt.boolean()],
            return_type=dt.int8(),
        )

    with pytest.raises(com.IbisTypeError):
        # wrong input type
        klass(
            func=lambda a, b, c: a,
            func_args=[ibis.literal(1), table.b, table.c],
            input_type="int8",
            return_type=dt.int8(),
        )

    with pytest.raises(com.IbisTypeError):
        # wrong return type
        klass(
            func=lambda a, b, c: a,
            func_args=[ibis.literal(1), table.b, table.c],
            input_type=[dt.int8(), dt.string(), dt.boolean()],
            return_type=table,
        )
