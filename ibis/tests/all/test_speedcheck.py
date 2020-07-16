from datetime import datetime

import ibis
import ibis.bigquery as bq
import ibis.expr.operations as ops
from ibis.expr.types import TableExpr


def test_large_compile():
    num_columns = 20
    num_joins = 7

    names = [f"col_{i}" for i in range(num_columns)]
    schema = ibis.Schema(names, ['string'] * num_columns)
    ibis_client = bq.BigQueryClient.__new__(bq.BigQueryClient)
    table = TableExpr(ops.SQLQueryResult("select * from t", schema, ibis_client))
    for _ in range(num_joins):
        table = table.mutate(dummy=ibis.literal(""))
        table = table.left_join(table, ["dummy"])[[table]]

    start = datetime.now()
    table.compile()
    delta = datetime.now() - start
    assert delta.total_seconds() < 10
