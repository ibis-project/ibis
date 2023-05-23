import pytest

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch

pa = pytest.importorskip('pyarrow')


def test_schema_from_to_pyarrow_schema():
    pyarrow_schema = pa.schema(
        [
            pa.field('a', pa.int64()),
            pa.field('b', pa.string()),
            pa.field('c', pa.bool_()),
        ]
    )
    ibis_schema = sch.schema(pyarrow_schema)
    restored_schema = ibis_schema.to_pyarrow()

    assert ibis_schema == sch.Schema({'a': dt.int64, 'b': dt.string, 'c': dt.boolean})
    assert restored_schema == pyarrow_schema


def test_schema_infer_pyarrow_table():
    table = pa.Table.from_arrays(
        [
            pa.array([1, 2, 3]),
            pa.array(['a', 'b', 'c']),
            pa.array([True, False, True]),
        ],
        ['a', 'b', 'c'],
    )
    s = sch.infer(table)
    assert s == sch.Schema({'a': dt.int64, 'b': dt.string, 'c': dt.boolean})
