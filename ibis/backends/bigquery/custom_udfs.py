from __future__ import annotations

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.bigquery.compiler import BigQueryExprTranslator
from ibis.backends.bigquery.udf import udf

# Based on:
# https://github.com/GoogleCloudPlatform/bigquery-utils/blob/45e1ac51367ab6209f68e04b1660d5b00258c131/udfs/community/typeof.sqlx#L1
typeof_ = udf.sql(
    name="typeof",
    params={"input": "ANY TYPE"},
    output_type=dt.str,
    sql_expression=r"""
    (
        SELECT
            CASE
                -- Process NUMERIC, DATE, DATETIME, TIME, TIMESTAMP,
                WHEN REGEXP_CONTAINS(literal, r'^[A-Z]+ "') THEN REGEXP_EXTRACT(literal, r'^([A-Z]+) "')
                WHEN REGEXP_CONTAINS(literal, r'^-?[0-9]*$') THEN 'INT64'
                WHEN
                    REGEXP_CONTAINS(literal, r'^(-?[0-9]+[.e].*|CAST\("([^"]*)" AS FLOAT64\))$')
                THEN
                    'FLOAT64'
                WHEN literal IN ('true', 'false') THEN 'BOOL'
                WHEN literal LIKE '"%' OR literal LIKE "'%" THEN 'STRING'
                WHEN literal LIKE 'b"%' THEN 'BYTES'
                WHEN literal LIKE '[%' THEN 'ARRAY'
                WHEN REGEXP_CONTAINS(literal, r'^(STRUCT)?\(') THEN 'STRUCT'
                WHEN literal LIKE 'ST_%' THEN 'GEOGRAPHY'
                WHEN literal = 'NULL' THEN 'NULL'
            ELSE
                'UNKNOWN'
            END
        FROM
            UNNEST([FORMAT('%T', input)]) AS literal
    )
    """,
)

BigQueryExprTranslator.rewrites(ops.TypeOf)(lambda op: typeof_(op.arg).op())
