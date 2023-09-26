WITH t0 AS (
  SELECT
    t1."L_ORDERKEY" AS "l_orderkey",
    t1."L_PARTKEY" AS "l_partkey",
    t1."L_SUPPKEY" AS "l_suppkey",
    t1."L_LINENUMBER" AS "l_linenumber",
    t1."L_QUANTITY" AS "l_quantity",
    t1."L_EXTENDEDPRICE" AS "l_extendedprice",
    t1."L_DISCOUNT" AS "l_discount",
    t1."L_TAX" AS "l_tax",
    t1."L_RETURNFLAG" AS "l_returnflag",
    t1."L_LINESTATUS" AS "l_linestatus",
    t1."L_SHIPDATE" AS "l_shipdate",
    t1."L_COMMITDATE" AS "l_commitdate",
    t1."L_RECEIPTDATE" AS "l_receiptdate",
    t1."L_SHIPINSTRUCT" AS "l_shipinstruct",
    t1."L_SHIPMODE" AS "l_shipmode",
    t1."L_COMMENT" AS "l_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS t1
)
SELECT
  SUM(t0."l_extendedprice" * t0."l_discount") AS "revenue"
FROM t0
WHERE
  t0."l_shipdate" >= DATE_FROM_PARTS(1994, 1, 1)
  AND t0."l_shipdate" < DATE_FROM_PARTS(1995, 1, 1)
  AND t0."l_discount" BETWEEN 0.05 AND 0.07
  AND t0."l_quantity" < 24