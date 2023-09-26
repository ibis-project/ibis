WITH t1 AS (
  SELECT
    t6."C_CUSTKEY" AS "c_custkey",
    t6."C_NAME" AS "c_name",
    t6."C_ADDRESS" AS "c_address",
    t6."C_NATIONKEY" AS "c_nationkey",
    t6."C_PHONE" AS "c_phone",
    t6."C_ACCTBAL" AS "c_acctbal",
    t6."C_MKTSEGMENT" AS "c_mktsegment",
    t6."C_COMMENT" AS "c_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."CUSTOMER" AS t6
), t0 AS (
  SELECT
    t6."O_ORDERKEY" AS "o_orderkey",
    t6."O_CUSTKEY" AS "o_custkey",
    t6."O_ORDERSTATUS" AS "o_orderstatus",
    t6."O_TOTALPRICE" AS "o_totalprice",
    t6."O_ORDERDATE" AS "o_orderdate",
    t6."O_ORDERPRIORITY" AS "o_orderpriority",
    t6."O_CLERK" AS "o_clerk",
    t6."O_SHIPPRIORITY" AS "o_shippriority",
    t6."O_COMMENT" AS "o_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS t6
), t2 AS (
  SELECT
    t6."L_ORDERKEY" AS "l_orderkey",
    t6."L_PARTKEY" AS "l_partkey",
    t6."L_SUPPKEY" AS "l_suppkey",
    t6."L_LINENUMBER" AS "l_linenumber",
    t6."L_QUANTITY" AS "l_quantity",
    t6."L_EXTENDEDPRICE" AS "l_extendedprice",
    t6."L_DISCOUNT" AS "l_discount",
    t6."L_TAX" AS "l_tax",
    t6."L_RETURNFLAG" AS "l_returnflag",
    t6."L_LINESTATUS" AS "l_linestatus",
    t6."L_SHIPDATE" AS "l_shipdate",
    t6."L_COMMITDATE" AS "l_commitdate",
    t6."L_RECEIPTDATE" AS "l_receiptdate",
    t6."L_SHIPINSTRUCT" AS "l_shipinstruct",
    t6."L_SHIPMODE" AS "l_shipmode",
    t6."L_COMMENT" AS "l_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS t6
), t3 AS (
  SELECT
    t6."N_NATIONKEY" AS "n_nationkey",
    t6."N_NAME" AS "n_name",
    t6."N_REGIONKEY" AS "n_regionkey",
    t6."N_COMMENT" AS "n_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS t6
), t4 AS (
  SELECT
    t1."c_custkey" AS "c_custkey",
    t1."c_name" AS "c_name",
    t1."c_acctbal" AS "c_acctbal",
    t3."n_name" AS "n_name",
    t1."c_address" AS "c_address",
    t1."c_phone" AS "c_phone",
    t1."c_comment" AS "c_comment",
    SUM(t2."l_extendedprice" * (
      1 - t2."l_discount"
    )) AS "revenue"
  FROM t1
  JOIN t0
    ON t1."c_custkey" = t0."o_custkey"
  JOIN t2
    ON t2."l_orderkey" = t0."o_orderkey"
  JOIN t3
    ON t1."c_nationkey" = t3."n_nationkey"
  WHERE
    t0."o_orderdate" >= DATE_FROM_PARTS(1993, 10, 1)
    AND t0."o_orderdate" < DATE_FROM_PARTS(1994, 1, 1)
    AND t2."l_returnflag" = 'R'
  GROUP BY
    1,
    2,
    3,
    4,
    5,
    6,
    7
)
SELECT
  t5."c_custkey",
  t5."c_name",
  t5."revenue",
  t5."c_acctbal",
  t5."n_name",
  t5."c_address",
  t5."c_phone",
  t5."c_comment"
FROM (
  SELECT
    t4."c_custkey" AS "c_custkey",
    t4."c_name" AS "c_name",
    t4."revenue" AS "revenue",
    t4."c_acctbal" AS "c_acctbal",
    t4."n_name" AS "n_name",
    t4."c_address" AS "c_address",
    t4."c_phone" AS "c_phone",
    t4."c_comment" AS "c_comment"
  FROM t4
) AS t5
ORDER BY
  t5."revenue" DESC
LIMIT 20