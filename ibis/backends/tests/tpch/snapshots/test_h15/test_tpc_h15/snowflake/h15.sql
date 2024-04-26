WITH "t8" AS (
  SELECT
    "t4"."s_suppkey",
    "t4"."s_name",
    "t4"."s_address",
    "t4"."s_nationkey",
    "t4"."s_phone",
    "t4"."s_acctbal",
    "t4"."s_comment",
    "t7"."l_suppkey",
    "t7"."total_revenue"
  FROM (
    SELECT
      "t0"."S_SUPPKEY" AS "s_suppkey",
      "t0"."S_NAME" AS "s_name",
      "t0"."S_ADDRESS" AS "s_address",
      "t0"."S_NATIONKEY" AS "s_nationkey",
      "t0"."S_PHONE" AS "s_phone",
      "t0"."S_ACCTBAL" AS "s_acctbal",
      "t0"."S_COMMENT" AS "s_comment"
    FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS "t0"
  ) AS "t4"
  INNER JOIN (
    SELECT
      "t5"."l_suppkey",
      SUM("t5"."l_extendedprice" * (
        1 - "t5"."l_discount"
      )) AS "total_revenue"
    FROM (
      SELECT
        "t3"."l_orderkey",
        "t3"."l_partkey",
        "t3"."l_suppkey",
        "t3"."l_linenumber",
        "t3"."l_quantity",
        "t3"."l_extendedprice",
        "t3"."l_discount",
        "t3"."l_tax",
        "t3"."l_returnflag",
        "t3"."l_linestatus",
        "t3"."l_shipdate",
        "t3"."l_commitdate",
        "t3"."l_receiptdate",
        "t3"."l_shipinstruct",
        "t3"."l_shipmode",
        "t3"."l_comment"
      FROM (
        SELECT
          "t1"."L_ORDERKEY" AS "l_orderkey",
          "t1"."L_PARTKEY" AS "l_partkey",
          "t1"."L_SUPPKEY" AS "l_suppkey",
          "t1"."L_LINENUMBER" AS "l_linenumber",
          "t1"."L_QUANTITY" AS "l_quantity",
          "t1"."L_EXTENDEDPRICE" AS "l_extendedprice",
          "t1"."L_DISCOUNT" AS "l_discount",
          "t1"."L_TAX" AS "l_tax",
          "t1"."L_RETURNFLAG" AS "l_returnflag",
          "t1"."L_LINESTATUS" AS "l_linestatus",
          "t1"."L_SHIPDATE" AS "l_shipdate",
          "t1"."L_COMMITDATE" AS "l_commitdate",
          "t1"."L_RECEIPTDATE" AS "l_receiptdate",
          "t1"."L_SHIPINSTRUCT" AS "l_shipinstruct",
          "t1"."L_SHIPMODE" AS "l_shipmode",
          "t1"."L_COMMENT" AS "l_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS "t1"
      ) AS "t3"
      WHERE
        "t3"."l_shipdate" >= DATE_FROM_PARTS(1996, 1, 1)
        AND "t3"."l_shipdate" < DATE_FROM_PARTS(1996, 4, 1)
    ) AS "t5"
    GROUP BY
      1
  ) AS "t7"
    ON "t4"."s_suppkey" = "t7"."l_suppkey"
)
SELECT
  "t12"."s_suppkey",
  "t12"."s_name",
  "t12"."s_address",
  "t12"."s_phone",
  "t12"."total_revenue"
FROM (
  SELECT
    "t11"."s_suppkey",
    "t11"."s_name",
    "t11"."s_address",
    "t11"."s_phone",
    "t11"."total_revenue"
  FROM (
    SELECT
      "t9"."s_suppkey",
      "t9"."s_name",
      "t9"."s_address",
      "t9"."s_nationkey",
      "t9"."s_phone",
      "t9"."s_acctbal",
      "t9"."s_comment",
      "t9"."l_suppkey",
      "t9"."total_revenue"
    FROM "t8" AS "t9"
    WHERE
      "t9"."total_revenue" = (
        SELECT
          MAX("t9"."total_revenue") AS "Max(total_revenue)"
        FROM "t8" AS "t9"
      )
  ) AS "t11"
) AS "t12"
ORDER BY
  "t12"."s_suppkey" ASC