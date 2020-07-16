from datetime import datetime

import ibis
import ibis.bigquery as bq
import ibis.expr.operations as ops
from ibis.expr.types import TableExpr
import pytest

@pytest.mark.timeout(10)
def test_large_compile():
    num_columns = 20
    num_joins = 7
    
    names = [f"col_{i}" for i in range(num_columns)]
    schema = ibis.Schema(names, ['string'] * num_columns)
    ibis_client = bq.BigQueryClient.__new__(bq.BigQueryClient)
    t = TableExpr(ops.SQLQueryResult("select * from t", schema, ibis_client))
    for i in range(num_joins):
        t = t.mutate(dummy=ibis.literal(""))
        t = t.left_join(t, ["dummy"])[[t]]
    
    start = datetime.now()
    print("start")
    t.compile()
    print("end")
    delta = datetime.now() - start
    # assert delta.total_seconds() < 10

