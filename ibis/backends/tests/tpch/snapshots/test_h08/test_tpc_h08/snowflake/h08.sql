WITH t1 AS (
  SELECT
    t12."P_PARTKEY" AS "p_partkey",
    t12."P_NAME" AS "p_name",
    t12."P_MFGR" AS "p_mfgr",
    t12."P_BRAND" AS "p_brand",
    t12."P_TYPE" AS "p_type",
    t12."P_SIZE" AS "p_size",
    t12."P_CONTAINER" AS "p_container",
    t12."P_RETAILPRICE" AS "p_retailprice",
    t12."P_COMMENT" AS "p_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PART" AS t12
), t0 AS (
  SELECT
    t12."L_ORDERKEY" AS "l_orderkey",
    t12."L_PARTKEY" AS "l_partkey",
    t12."L_SUPPKEY" AS "l_suppkey",
    t12."L_LINENUMBER" AS "l_linenumber",
    t12."L_QUANTITY" AS "l_quantity",
    t12."L_EXTENDEDPRICE" AS "l_extendedprice",
    t12."L_DISCOUNT" AS "l_discount",
    t12."L_TAX" AS "l_tax",
    t12."L_RETURNFLAG" AS "l_returnflag",
    t12."L_LINESTATUS" AS "l_linestatus",
    t12."L_SHIPDATE" AS "l_shipdate",
    t12."L_COMMITDATE" AS "l_commitdate",
    t12."L_RECEIPTDATE" AS "l_receiptdate",
    t12."L_SHIPINSTRUCT" AS "l_shipinstruct",
    t12."L_SHIPMODE" AS "l_shipmode",
    t12."L_COMMENT" AS "l_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS t12
), t2 AS (
  SELECT
    t12."S_SUPPKEY" AS "s_suppkey",
    t12."S_NAME" AS "s_name",
    t12."S_ADDRESS" AS "s_address",
    t12."S_NATIONKEY" AS "s_nationkey",
    t12."S_PHONE" AS "s_phone",
    t12."S_ACCTBAL" AS "s_acctbal",
    t12."S_COMMENT" AS "s_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS t12
), t3 AS (
  SELECT
    t12."O_ORDERKEY" AS "o_orderkey",
    t12."O_CUSTKEY" AS "o_custkey",
    t12."O_ORDERSTATUS" AS "o_orderstatus",
    t12."O_TOTALPRICE" AS "o_totalprice",
    t12."O_ORDERDATE" AS "o_orderdate",
    t12."O_ORDERPRIORITY" AS "o_orderpriority",
    t12."O_CLERK" AS "o_clerk",
    t12."O_SHIPPRIORITY" AS "o_shippriority",
    t12."O_COMMENT" AS "o_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS t12
), t4 AS (
  SELECT
    t12."C_CUSTKEY" AS "c_custkey",
    t12."C_NAME" AS "c_name",
    t12."C_ADDRESS" AS "c_address",
    t12."C_NATIONKEY" AS "c_nationkey",
    t12."C_PHONE" AS "c_phone",
    t12."C_ACCTBAL" AS "c_acctbal",
    t12."C_MKTSEGMENT" AS "c_mktsegment",
    t12."C_COMMENT" AS "c_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."CUSTOMER" AS t12
), t5 AS (
  SELECT
    t12."N_NATIONKEY" AS "n_nationkey",
    t12."N_NAME" AS "n_name",
    t12."N_REGIONKEY" AS "n_regionkey",
    t12."N_COMMENT" AS "n_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS t12
), t6 AS (
  SELECT
    t12."R_REGIONKEY" AS "r_regionkey",
    t12."R_NAME" AS "r_name",
    t12."R_COMMENT" AS "r_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."REGION" AS t12
), t7 AS (
  SELECT
    CAST(DATE_PART(year, t3."o_orderdate") AS SMALLINT) AS "o_year",
    t0."l_extendedprice" * (
      1 - t0."l_discount"
    ) AS "volume",
    t12."n_name" AS "nation",
    t6."r_name" AS "r_name",
    t3."o_orderdate" AS "o_orderdate",
    t1."p_type" AS "p_type"
  FROM t1
  JOIN t0
    ON t1."p_partkey" = t0."l_partkey"
  JOIN t2
    ON t2."s_suppkey" = t0."l_suppkey"
  JOIN t3
    ON t0."l_orderkey" = t3."o_orderkey"
  JOIN t4
    ON t3."o_custkey" = t4."c_custkey"
  JOIN t5
    ON t4."c_nationkey" = t5."n_nationkey"
  JOIN t6
    ON t5."n_regionkey" = t6."r_regionkey"
  JOIN t5 AS t12
    ON t2."s_nationkey" = t12."n_nationkey"
), t8 AS (
  SELECT
    t7."o_year" AS "o_year",
    t7."volume" AS "volume",
    t7."nation" AS "nation",
    t7."r_name" AS "r_name",
    t7."o_orderdate" AS "o_orderdate",
    t7."p_type" AS "p_type"
  FROM t7
  WHERE
    t7."r_name" = 'AMERICA'
    AND t7."o_orderdate" BETWEEN '1995-01-01' AND '1996-12-31'
    AND t7."p_type" = 'ECONOMY ANODIZED STEEL'
), t9 AS (
  SELECT
    t8."o_year" AS "o_year",
    t8."volume" AS "volume",
    t8."nation" AS "nation",
    t8."r_name" AS "r_name",
    t8."o_orderdate" AS "o_orderdate",
    t8."p_type" AS "p_type",
    CASE WHEN (
      t8."nation" = 'BRAZIL'
    ) THEN t8."volume" ELSE 0 END AS "nation_volume"
  FROM t8
), t10 AS (
  SELECT
    t9."o_year" AS "o_year",
    SUM(t9."nation_volume") / SUM(t9."volume") AS "mkt_share"
  FROM t9
  GROUP BY
    1
)
SELECT
  CAST(t11."o_year" AS BIGINT) AS "o_year",
  CAST(t11."mkt_share" AS DECIMAL(38, 10)) AS "mkt_share"
FROM (
  SELECT
    t10."o_year" AS "o_year",
    t10."mkt_share" AS "mkt_share"
  FROM t10
  ORDER BY
    t10."o_year"
) AS t11