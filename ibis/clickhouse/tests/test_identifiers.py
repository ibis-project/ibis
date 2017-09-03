import ibis
import pytest


pytest.importorskip('clickhouse_driver')
pytestmark = pytest.mark.clickhouse


def test_column_ref_quoting(translate):
    schema = [('has a space', 'double')]
    table = ibis.table(schema)
    assert translate(table['has a space']) == '`has a space`'


def test_identifier_quoting(translate):
    schema = [('date', 'double'), ('table', 'string')]
    table = ibis.table(schema)
    assert translate(table['date']) == '`date`'
    assert translate(table['table']) == '`table`'


# TODO: fix it
# def test_named_expression(alltypes, translate):
#     a, b = alltypes.get_columns(['int_col', 'float_col'])
#     expr = ((a - b) * a).name('expr')

#     expected = '(`int_col` - `float_col`) * `int_col` AS `expr`'
#     assert translate(expr) == expected
