import pandas.testing as mct
import pytest

import ibis
import ibis.expr.operations as ops

pytest.importorskip('pyspark')


class UnboundDatabaseTable(ops.UnboundTable):
    """Test-only unbound database table."""


def test_execute(client):
    # Conditional imports
    from ibis.pyspark.compiler import compiles

    @compiles(UnboundDatabaseTable)
    def compile_unbound_database_table(t, expr, scope):
        return t.session.table(expr.op().name)

    table1 = UnboundDatabaseTable(
        name='basic_table',
        schema=ibis.schema([('value', 'string'), ('str_col', 'string')]),
    ).to_expr()

    result1 = table1.execute(client=client)
    expected1 = client.execute(table1)
    mct.assert_frame_equal(result1, expected1)

    # Test duplicate client
    table2 = client.table('basic_table')

    result2 = table2.execute(client=client)
    expected2 = client.execute(table2, client=client)
    mct.assert_frame_equal(result2, expected2)
