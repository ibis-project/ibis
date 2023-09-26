WITH t2 AS (
  SELECT
    t5."S_SUPPKEY" AS "s_suppkey",
    t5."S_NAME" AS "s_name",
    t5."S_ADDRESS" AS "s_address",
    t5."S_NATIONKEY" AS "s_nationkey",
    t5."S_PHONE" AS "s_phone",
    t5."S_ACCTBAL" AS "s_acctbal",
    t5."S_COMMENT" AS "s_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS t5
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
), t1 AS (
  SELECT
    t0."l_suppkey" AS "l_suppkey",
    SUM(t0."l_extendedprice" * (
      1 - t0."l_discount"
    )) AS "total_revenue"
  FROM t0
  WHERE
    t0."l_shipdate" >= DATE_FROM_PARTS(1996, 1, 1)
    AND t0."l_shipdate" < DATE_FROM_PARTS(1996, 4, 1)
  GROUP BY
    1
), t3 AS (
  SELECT
    t2."s_suppkey" AS "s_suppkey",
    t2."s_name" AS "s_name",
    t2."s_address" AS "s_address",
    t2."s_nationkey" AS "s_nationkey",
    t2."s_phone" AS "s_phone",
    t2."s_acctbal" AS "s_acctbal",
    t2."s_comment" AS "s_comment",
    t1."l_suppkey" AS "l_suppkey",
    t1."total_revenue" AS "total_revenue"
  FROM t2
  JOIN t1
    ON t2."s_suppkey" = t1."l_suppkey"
  WHERE
    t1."total_revenue" = (
      SELECT
        MAX(t1."total_revenue") AS "Max(total_revenue)"
      FROM t1
    )
)
SELECT
  t4."s_suppkey",
  t4."s_name",
  t4."s_address",
  t4."s_phone",
  t4."total_revenue"
FROM (
  SELECT
    t3."s_suppkey" AS "s_suppkey",
    t3."s_name" AS "s_name",
    t3."s_address" AS "s_address",
    t3."s_nationkey" AS "s_nationkey",
    t3."s_phone" AS "s_phone",
    t3."s_acctbal" AS "s_acctbal",
    t3."s_comment" AS "s_comment",
    t3."l_suppkey" AS "l_suppkey",
    t3."total_revenue" AS "total_revenue"
  FROM t3
  ORDER BY
    t3."s_suppkey" ASC
) AS t4