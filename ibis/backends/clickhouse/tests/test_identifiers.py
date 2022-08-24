import pytest

import ibis

pytest.importorskip("clickhouse_driver")


def test_column_ref_quoting(translate):
    schema = [('has a space', 'double')]
    table = ibis.table(schema)
    assert translate(table['has a space'].op()) == '`has a space`'


def test_identifier_quoting(translate):
    schema = [('date', 'double'), ('table', 'string')]
    table = ibis.table(schema)
    assert translate(table['date'].op()) == '`date`'
    assert translate(table['table'].op()) == '`table`'
