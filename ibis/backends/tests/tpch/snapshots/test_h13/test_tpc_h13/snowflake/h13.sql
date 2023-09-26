WITH t1 AS (
  SELECT
    t4."C_CUSTKEY" AS "c_custkey",
    t4."C_NAME" AS "c_name",
    t4."C_ADDRESS" AS "c_address",
    t4."C_NATIONKEY" AS "c_nationkey",
    t4."C_PHONE" AS "c_phone",
    t4."C_ACCTBAL" AS "c_acctbal",
    t4."C_MKTSEGMENT" AS "c_mktsegment",
    t4."C_COMMENT" AS "c_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."CUSTOMER" AS t4
), t0 AS (
  SELECT
    t4."O_ORDERKEY" AS "o_orderkey",
    t4."O_CUSTKEY" AS "o_custkey",
    t4."O_ORDERSTATUS" AS "o_orderstatus",
    t4."O_TOTALPRICE" AS "o_totalprice",
    t4."O_ORDERDATE" AS "o_orderdate",
    t4."O_ORDERPRIORITY" AS "o_orderpriority",
    t4."O_CLERK" AS "o_clerk",
    t4."O_SHIPPRIORITY" AS "o_shippriority",
    t4."O_COMMENT" AS "o_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS t4
), t2 AS (
  SELECT
    t1."c_custkey" AS "c_custkey",
    COUNT(t0."o_orderkey") AS "c_count"
  FROM t1
  LEFT OUTER JOIN t0
    ON t1."c_custkey" = t0."o_custkey" AND NOT t0."o_comment" LIKE '%special%requests%'
  GROUP BY
    1
)
SELECT
  t3."c_count",
  t3."custdist"
FROM (
  SELECT
    t2."c_count" AS "c_count",
    COUNT(*) AS "custdist"
  FROM t2
  GROUP BY
    1
) AS t3
ORDER BY
  t3."custdist" DESC,
  t3."c_count" DESC