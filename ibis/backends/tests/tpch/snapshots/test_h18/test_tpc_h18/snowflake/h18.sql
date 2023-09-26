WITH t2 AS (
  SELECT
    t5."C_CUSTKEY" AS "c_custkey",
    t5."C_NAME" AS "c_name",
    t5."C_ADDRESS" AS "c_address",
    t5."C_NATIONKEY" AS "c_nationkey",
    t5."C_PHONE" AS "c_phone",
    t5."C_ACCTBAL" AS "c_acctbal",
    t5."C_MKTSEGMENT" AS "c_mktsegment",
    t5."C_COMMENT" AS "c_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."CUSTOMER" AS t5
), t1 AS (
  SELECT
    t5."O_ORDERKEY" AS "o_orderkey",
    t5."O_CUSTKEY" AS "o_custkey",
    t5."O_ORDERSTATUS" AS "o_orderstatus",
    t5."O_TOTALPRICE" AS "o_totalprice",
    t5."O_ORDERDATE" AS "o_orderdate",
    t5."O_ORDERPRIORITY" AS "o_orderpriority",
    t5."O_CLERK" AS "o_clerk",
    t5."O_SHIPPRIORITY" AS "o_shippriority",
    t5."O_COMMENT" AS "o_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS t5
), t0 AS (
  SELECT
    t5."L_ORDERKEY" AS "l_orderkey",
    t5."L_PARTKEY" AS "l_partkey",
    t5."L_SUPPKEY" AS "l_suppkey",
    t5."L_LINENUMBER" AS "l_linenumber",
    t5."L_QUANTITY" AS "l_quantity",
    t5."L_EXTENDEDPRICE" AS "l_extendedprice",
    t5."L_DISCOUNT" AS "l_discount",
    t5."L_TAX" AS "l_tax",
    t5."L_RETURNFLAG" AS "l_returnflag",
    t5."L_LINESTATUS" AS "l_linestatus",
    t5."L_SHIPDATE" AS "l_shipdate",
    t5."L_COMMITDATE" AS "l_commitdate",
    t5."L_RECEIPTDATE" AS "l_receiptdate",
    t5."L_SHIPINSTRUCT" AS "l_shipinstruct",
    t5."L_SHIPMODE" AS "l_shipmode",
    t5."L_COMMENT" AS "l_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS t5
), t3 AS (
  SELECT
    t0."l_orderkey" AS "l_orderkey",
    SUM(t0."l_quantity") AS "qty_sum"
  FROM t0
  GROUP BY
    1
)
SELECT
  t4."c_name",
  t4."c_custkey",
  t4."o_orderkey",
  t4."o_orderdate",
  t4."o_totalprice",
  t4."sum_qty"
FROM (
  SELECT
    t2."c_name" AS "c_name",
    t2."c_custkey" AS "c_custkey",
    t1."o_orderkey" AS "o_orderkey",
    t1."o_orderdate" AS "o_orderdate",
    t1."o_totalprice" AS "o_totalprice",
    SUM(t0."l_quantity") AS "sum_qty"
  FROM t2
  JOIN t1
    ON t2."c_custkey" = t1."o_custkey"
  JOIN t0
    ON t1."o_orderkey" = t0."l_orderkey"
  WHERE
    t1."o_orderkey" IN (
      SELECT
        t5."l_orderkey"
      FROM (
        SELECT
          t3."l_orderkey" AS "l_orderkey",
          t3."qty_sum" AS "qty_sum"
        FROM t3
        WHERE
          t3."qty_sum" > 300
      ) AS t5
    )
  GROUP BY
    1,
    2,
    3,
    4,
    5
) AS t4
ORDER BY
  t4."o_totalprice" DESC,
  t4."o_orderdate" ASC
LIMIT 100