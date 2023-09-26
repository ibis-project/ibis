WITH t1 AS (
  SELECT
    t2."O_ORDERKEY" AS "o_orderkey",
    t2."O_CUSTKEY" AS "o_custkey",
    t2."O_ORDERSTATUS" AS "o_orderstatus",
    t2."O_TOTALPRICE" AS "o_totalprice",
    t2."O_ORDERDATE" AS "o_orderdate",
    t2."O_ORDERPRIORITY" AS "o_orderpriority",
    t2."O_CLERK" AS "o_clerk",
    t2."O_SHIPPRIORITY" AS "o_shippriority",
    t2."O_COMMENT" AS "o_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS t2
), t0 AS (
  SELECT
    t2."L_ORDERKEY" AS "l_orderkey",
    t2."L_PARTKEY" AS "l_partkey",
    t2."L_SUPPKEY" AS "l_suppkey",
    t2."L_LINENUMBER" AS "l_linenumber",
    t2."L_QUANTITY" AS "l_quantity",
    t2."L_EXTENDEDPRICE" AS "l_extendedprice",
    t2."L_DISCOUNT" AS "l_discount",
    t2."L_TAX" AS "l_tax",
    t2."L_RETURNFLAG" AS "l_returnflag",
    t2."L_LINESTATUS" AS "l_linestatus",
    t2."L_SHIPDATE" AS "l_shipdate",
    t2."L_COMMITDATE" AS "l_commitdate",
    t2."L_RECEIPTDATE" AS "l_receiptdate",
    t2."L_SHIPINSTRUCT" AS "l_shipinstruct",
    t2."L_SHIPMODE" AS "l_shipmode",
    t2."L_COMMENT" AS "l_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS t2
)
SELECT
  t1."o_orderpriority",
  COUNT(*) AS "order_count"
FROM t1
WHERE
  (
    EXISTS(
      SELECT
        1 AS anon_1
      FROM t0
      WHERE
        t0."l_orderkey" = t1."o_orderkey" AND t0."l_commitdate" < t0."l_receiptdate"
    )
  )
  AND t1."o_orderdate" >= DATE_FROM_PARTS(1993, 7, 1)
  AND t1."o_orderdate" < DATE_FROM_PARTS(1993, 10, 1)
GROUP BY
  1
ORDER BY
  t1."o_orderpriority" ASC