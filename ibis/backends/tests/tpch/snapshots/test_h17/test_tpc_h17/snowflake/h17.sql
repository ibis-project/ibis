WITH t0 AS (
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
), t1 AS (
  SELECT
    t3."P_PARTKEY" AS "p_partkey",
    t3."P_NAME" AS "p_name",
    t3."P_MFGR" AS "p_mfgr",
    t3."P_BRAND" AS "p_brand",
    t3."P_TYPE" AS "p_type",
    t3."P_SIZE" AS "p_size",
    t3."P_CONTAINER" AS "p_container",
    t3."P_RETAILPRICE" AS "p_retailprice",
    t3."P_COMMENT" AS "p_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PART" AS t3
)
SELECT
  CAST(t2."avg_yearly" AS DECIMAL(30, 8)) AS "avg_yearly"
FROM (
  SELECT
    SUM(t0."l_extendedprice") / 7.0 AS "avg_yearly"
  FROM t0
  JOIN t1
    ON t1."p_partkey" = t0."l_partkey"
  WHERE
    t1."p_brand" = 'Brand#23'
    AND t1."p_container" = 'MED BOX'
    AND t0."l_quantity" < (
      SELECT
        AVG(t0."l_quantity") AS "Mean(l_quantity)"
      FROM t0
      WHERE
        t0."l_partkey" = t1."p_partkey"
    ) * 0.2
) AS t2