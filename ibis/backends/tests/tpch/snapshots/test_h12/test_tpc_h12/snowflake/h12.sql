WITH t1 AS (
  SELECT
    t3."O_ORDERKEY" AS "o_orderkey",
    t3."O_CUSTKEY" AS "o_custkey",
    t3."O_ORDERSTATUS" AS "o_orderstatus",
    t3."O_TOTALPRICE" AS "o_totalprice",
    t3."O_ORDERDATE" AS "o_orderdate",
    t3."O_ORDERPRIORITY" AS "o_orderpriority",
    t3."O_CLERK" AS "o_clerk",
    t3."O_SHIPPRIORITY" AS "o_shippriority",
    t3."O_COMMENT" AS "o_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS t3
), t0 AS (
  SELECT
    t3."L_ORDERKEY" AS "l_orderkey",
    t3."L_PARTKEY" AS "l_partkey",
    t3."L_SUPPKEY" AS "l_suppkey",
    t3."L_LINENUMBER" AS "l_linenumber",
    t3."L_QUANTITY" AS "l_quantity",
    t3."L_EXTENDEDPRICE" AS "l_extendedprice",
    t3."L_DISCOUNT" AS "l_discount",
    t3."L_TAX" AS "l_tax",
    t3."L_RETURNFLAG" AS "l_returnflag",
    t3."L_LINESTATUS" AS "l_linestatus",
    t3."L_SHIPDATE" AS "l_shipdate",
    t3."L_COMMITDATE" AS "l_commitdate",
    t3."L_RECEIPTDATE" AS "l_receiptdate",
    t3."L_SHIPINSTRUCT" AS "l_shipinstruct",
    t3."L_SHIPMODE" AS "l_shipmode",
    t3."L_COMMENT" AS "l_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS t3
)
SELECT
  t2."l_shipmode",
  t2."high_line_count",
  t2."low_line_count"
FROM (
  SELECT
    t0."l_shipmode" AS "l_shipmode",
    SUM(CASE t1."o_orderpriority" WHEN '1-URGENT' THEN 1 WHEN '2-HIGH' THEN 1 ELSE 0 END) AS "high_line_count",
    SUM(CASE t1."o_orderpriority" WHEN '1-URGENT' THEN 0 WHEN '2-HIGH' THEN 0 ELSE 1 END) AS "low_line_count"
  FROM t1
  JOIN t0
    ON t1."o_orderkey" = t0."l_orderkey"
  WHERE
    t0."l_shipmode" IN ('MAIL', 'SHIP')
    AND t0."l_commitdate" < t0."l_receiptdate"
    AND t0."l_shipdate" < t0."l_commitdate"
    AND t0."l_receiptdate" >= DATE_FROM_PARTS(1994, 1, 1)
    AND t0."l_receiptdate" < DATE_FROM_PARTS(1995, 1, 1)
  GROUP BY
    1
) AS t2
ORDER BY
  t2."l_shipmode" ASC