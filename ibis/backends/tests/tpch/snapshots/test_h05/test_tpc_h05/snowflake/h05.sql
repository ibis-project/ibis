WITH t1 AS (
  SELECT
    t7."C_CUSTKEY" AS "c_custkey",
    t7."C_NAME" AS "c_name",
    t7."C_ADDRESS" AS "c_address",
    t7."C_NATIONKEY" AS "c_nationkey",
    t7."C_PHONE" AS "c_phone",
    t7."C_ACCTBAL" AS "c_acctbal",
    t7."C_MKTSEGMENT" AS "c_mktsegment",
    t7."C_COMMENT" AS "c_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."CUSTOMER" AS t7
), t0 AS (
  SELECT
    t7."O_ORDERKEY" AS "o_orderkey",
    t7."O_CUSTKEY" AS "o_custkey",
    t7."O_ORDERSTATUS" AS "o_orderstatus",
    t7."O_TOTALPRICE" AS "o_totalprice",
    t7."O_ORDERDATE" AS "o_orderdate",
    t7."O_ORDERPRIORITY" AS "o_orderpriority",
    t7."O_CLERK" AS "o_clerk",
    t7."O_SHIPPRIORITY" AS "o_shippriority",
    t7."O_COMMENT" AS "o_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS t7
), t2 AS (
  SELECT
    t7."L_ORDERKEY" AS "l_orderkey",
    t7."L_PARTKEY" AS "l_partkey",
    t7."L_SUPPKEY" AS "l_suppkey",
    t7."L_LINENUMBER" AS "l_linenumber",
    t7."L_QUANTITY" AS "l_quantity",
    t7."L_EXTENDEDPRICE" AS "l_extendedprice",
    t7."L_DISCOUNT" AS "l_discount",
    t7."L_TAX" AS "l_tax",
    t7."L_RETURNFLAG" AS "l_returnflag",
    t7."L_LINESTATUS" AS "l_linestatus",
    t7."L_SHIPDATE" AS "l_shipdate",
    t7."L_COMMITDATE" AS "l_commitdate",
    t7."L_RECEIPTDATE" AS "l_receiptdate",
    t7."L_SHIPINSTRUCT" AS "l_shipinstruct",
    t7."L_SHIPMODE" AS "l_shipmode",
    t7."L_COMMENT" AS "l_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS t7
), t3 AS (
  SELECT
    t7."S_SUPPKEY" AS "s_suppkey",
    t7."S_NAME" AS "s_name",
    t7."S_ADDRESS" AS "s_address",
    t7."S_NATIONKEY" AS "s_nationkey",
    t7."S_PHONE" AS "s_phone",
    t7."S_ACCTBAL" AS "s_acctbal",
    t7."S_COMMENT" AS "s_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS t7
), t4 AS (
  SELECT
    t7."N_NATIONKEY" AS "n_nationkey",
    t7."N_NAME" AS "n_name",
    t7."N_REGIONKEY" AS "n_regionkey",
    t7."N_COMMENT" AS "n_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS t7
), t5 AS (
  SELECT
    t7."R_REGIONKEY" AS "r_regionkey",
    t7."R_NAME" AS "r_name",
    t7."R_COMMENT" AS "r_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."REGION" AS t7
)
SELECT
  t6."n_name",
  t6."revenue"
FROM (
  SELECT
    t4."n_name" AS "n_name",
    SUM(t2."l_extendedprice" * (
      1 - t2."l_discount"
    )) AS "revenue"
  FROM t1
  JOIN t0
    ON t1."c_custkey" = t0."o_custkey"
  JOIN t2
    ON t2."l_orderkey" = t0."o_orderkey"
  JOIN t3
    ON t2."l_suppkey" = t3."s_suppkey"
  JOIN t4
    ON t1."c_nationkey" = t3."s_nationkey" AND t3."s_nationkey" = t4."n_nationkey"
  JOIN t5
    ON t4."n_regionkey" = t5."r_regionkey"
  WHERE
    t5."r_name" = 'ASIA'
    AND t0."o_orderdate" >= DATE_FROM_PARTS(1994, 1, 1)
    AND t0."o_orderdate" < DATE_FROM_PARTS(1995, 1, 1)
  GROUP BY
    1
) AS t6
ORDER BY
  t6."revenue" DESC