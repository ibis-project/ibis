SELECT
  "t8"."c_count",
  "t8"."custdist"
FROM (
  SELECT
    "t7"."c_count",
    COUNT(*) AS "custdist"
  FROM (
    SELECT
      "t6"."c_custkey",
      COUNT("t6"."o_orderkey") AS "c_count"
    FROM (
      SELECT
        "t4"."c_custkey",
        "t4"."c_name",
        "t4"."c_address",
        "t4"."c_nationkey",
        "t4"."c_phone",
        "t4"."c_acctbal",
        "t4"."c_mktsegment",
        "t4"."c_comment",
        "t5"."o_orderkey",
        "t5"."o_custkey",
        "t5"."o_orderstatus",
        "t5"."o_totalprice",
        "t5"."o_orderdate",
        "t5"."o_orderpriority",
        "t5"."o_clerk",
        "t5"."o_shippriority",
        "t5"."o_comment"
      FROM (
        SELECT
          "t0"."C_CUSTKEY" AS "c_custkey",
          "t0"."C_NAME" AS "c_name",
          "t0"."C_ADDRESS" AS "c_address",
          "t0"."C_NATIONKEY" AS "c_nationkey",
          "t0"."C_PHONE" AS "c_phone",
          "t0"."C_ACCTBAL" AS "c_acctbal",
          "t0"."C_MKTSEGMENT" AS "c_mktsegment",
          "t0"."C_COMMENT" AS "c_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."CUSTOMER" AS "t0"
      ) AS "t4"
      LEFT OUTER JOIN (
        SELECT
          "t1"."O_ORDERKEY" AS "o_orderkey",
          "t1"."O_CUSTKEY" AS "o_custkey",
          "t1"."O_ORDERSTATUS" AS "o_orderstatus",
          "t1"."O_TOTALPRICE" AS "o_totalprice",
          "t1"."O_ORDERDATE" AS "o_orderdate",
          "t1"."O_ORDERPRIORITY" AS "o_orderpriority",
          "t1"."O_CLERK" AS "o_clerk",
          "t1"."O_SHIPPRIORITY" AS "o_shippriority",
          "t1"."O_COMMENT" AS "o_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS "t1"
      ) AS "t5"
        ON "t4"."c_custkey" = "t5"."o_custkey"
        AND NOT (
          "t5"."o_comment" LIKE '%special%requests%'
        )
    ) AS "t6"
    GROUP BY
      1
  ) AS "t7"
  GROUP BY
    1
) AS "t8"
ORDER BY
  "t8"."custdist" DESC NULLS LAST,
  "t8"."c_count" DESC NULLS LAST