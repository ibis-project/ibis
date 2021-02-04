import ibis
import ibis.expr.operations as ops
from ibis.expr.types import TableExpr


def _get_ibis_client():
    ibis_client = ibis.bigquery.BigQueryClient
    setattr(ibis_client, "__init__", lambda t: None)
    ibis_client = ibis_client()
    return ibis_client


def _get_test_table(ibis_client):
    names = [
        'Class',
        'end_of_period',
        'M_Total_Dollars_Total',
    ]
    types = ['string', "date", 'float64']
    schema = ibis.Schema(names, types)
    t = TableExpr(ops.SQLQueryResult("select * from t", schema, ibis_client))
    return t


def test_window_compile():
    ibis_client = _get_ibis_client()
    t = _get_test_table(ibis_client)
    order_col = 'end_of_period'
    window = ibis.range_window(
        preceding=ibis.interval(days=62),
        order_by=t[order_col],
        group_by=[t['Class']],
    )
    t = t.mutate(window=t["M_Total_Dollars_Total"].sum().over(window))
    # no exception thrown in compiling
    t.compile()
