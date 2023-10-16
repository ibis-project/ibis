WITH t0 AS (
  SELECT
    t8."L_ORDERKEY" AS "l_orderkey",
    t8."L_PARTKEY" AS "l_partkey",
    t8."L_SUPPKEY" AS "l_suppkey",
    t8."L_LINENUMBER" AS "l_linenumber",
    t8."L_QUANTITY" AS "l_quantity",
    t8."L_EXTENDEDPRICE" AS "l_extendedprice",
    t8."L_DISCOUNT" AS "l_discount",
    t8."L_TAX" AS "l_tax",
    t8."L_RETURNFLAG" AS "l_returnflag",
    t8."L_LINESTATUS" AS "l_linestatus",
    t8."L_SHIPDATE" AS "l_shipdate",
    t8."L_COMMITDATE" AS "l_commitdate",
    t8."L_RECEIPTDATE" AS "l_receiptdate",
    t8."L_SHIPINSTRUCT" AS "l_shipinstruct",
    t8."L_SHIPMODE" AS "l_shipmode",
    t8."L_COMMENT" AS "l_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS t8
), t1 AS (
  SELECT
    t8."S_SUPPKEY" AS "s_suppkey",
    t8."S_NAME" AS "s_name",
    t8."S_ADDRESS" AS "s_address",
    t8."S_NATIONKEY" AS "s_nationkey",
    t8."S_PHONE" AS "s_phone",
    t8."S_ACCTBAL" AS "s_acctbal",
    t8."S_COMMENT" AS "s_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS t8
), t2 AS (
  SELECT
    t8."PS_PARTKEY" AS "ps_partkey",
    t8."PS_SUPPKEY" AS "ps_suppkey",
    t8."PS_AVAILQTY" AS "ps_availqty",
    t8."PS_SUPPLYCOST" AS "ps_supplycost",
    t8."PS_COMMENT" AS "ps_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PARTSUPP" AS t8
), t3 AS (
  SELECT
    t8."P_PARTKEY" AS "p_partkey",
    t8."P_NAME" AS "p_name",
    t8."P_MFGR" AS "p_mfgr",
    t8."P_BRAND" AS "p_brand",
    t8."P_TYPE" AS "p_type",
    t8."P_SIZE" AS "p_size",
    t8."P_CONTAINER" AS "p_container",
    t8."P_RETAILPRICE" AS "p_retailprice",
    t8."P_COMMENT" AS "p_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PART" AS t8
), t4 AS (
  SELECT
    t8."O_ORDERKEY" AS "o_orderkey",
    t8."O_CUSTKEY" AS "o_custkey",
    t8."O_ORDERSTATUS" AS "o_orderstatus",
    t8."O_TOTALPRICE" AS "o_totalprice",
    t8."O_ORDERDATE" AS "o_orderdate",
    t8."O_ORDERPRIORITY" AS "o_orderpriority",
    t8."O_CLERK" AS "o_clerk",
    t8."O_SHIPPRIORITY" AS "o_shippriority",
    t8."O_COMMENT" AS "o_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS t8
), t5 AS (
  SELECT
    t8."N_NATIONKEY" AS "n_nationkey",
    t8."N_NAME" AS "n_name",
    t8."N_REGIONKEY" AS "n_regionkey",
    t8."N_COMMENT" AS "n_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS t8
), t6 AS (
  SELECT
    t0."l_extendedprice" * (
      1 - t0."l_discount"
    ) - t2."ps_supplycost" * t0."l_quantity" AS "amount",
    CAST(DATE_PART(year, t4."o_orderdate") AS SMALLINT) AS "o_year",
    t5."n_name" AS "nation",
    t3."p_name" AS "p_name"
  FROM t0
  JOIN t1
    ON t1."s_suppkey" = t0."l_suppkey"
  JOIN t2
    ON t2."ps_suppkey" = t0."l_suppkey" AND t2."ps_partkey" = t0."l_partkey"
  JOIN t3
    ON t3."p_partkey" = t0."l_partkey"
  JOIN t4
    ON t4."o_orderkey" = t0."l_orderkey"
  JOIN t5
    ON t1."s_nationkey" = t5."n_nationkey"
  WHERE
    t3."p_name" LIKE '%green%'
)
SELECT
  t7."nation",
  t7."o_year",
  t7."sum_profit"
FROM (
  SELECT
    t6."nation" AS "nation",
    t6."o_year" AS "o_year",
    SUM(t6."amount") AS "sum_profit"
  FROM t6
  GROUP BY
    1,
    2
) AS t7
ORDER BY
  t7."nation" ASC,
  t7."o_year" DESC