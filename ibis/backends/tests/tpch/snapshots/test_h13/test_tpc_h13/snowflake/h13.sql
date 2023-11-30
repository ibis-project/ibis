SELECT
  *
FROM (
  SELECT
    "t6"."c_count" AS "c_count",
    COUNT(*) AS "custdist"
  FROM (
    SELECT
      "t5"."c_custkey" AS "c_custkey",
      COUNT("t5"."o_orderkey") AS "c_count"
    FROM (
      SELECT
        "t2"."c_custkey" AS "c_custkey",
        "t2"."c_name" AS "c_name",
        "t2"."c_address" AS "c_address",
        "t2"."c_nationkey" AS "c_nationkey",
        "t2"."c_phone" AS "c_phone",
        "t2"."c_acctbal" AS "c_acctbal",
        "t2"."c_mktsegment" AS "c_mktsegment",
        "t2"."c_comment" AS "c_comment",
        "t3"."o_orderkey" AS "o_orderkey",
        "t3"."o_custkey" AS "o_custkey",
        "t3"."o_orderstatus" AS "o_orderstatus",
        "t3"."o_totalprice" AS "o_totalprice",
        "t3"."o_orderdate" AS "o_orderdate",
        "t3"."o_orderpriority" AS "o_orderpriority",
        "t3"."o_clerk" AS "o_clerk",
        "t3"."o_shippriority" AS "o_shippriority",
        "t3"."o_comment" AS "o_comment"
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
        FROM "CUSTOMER" AS "t0"
      ) AS "t2"
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
        FROM "ORDERS" AS "t1"
      ) AS "t3"
        ON "t2"."c_custkey" = "t3"."o_custkey"
        AND NOT (
          "t3"."o_comment" LIKE '%special%requests%'
        )
    ) AS "t5"
    GROUP BY
      1
  ) AS "t6"
  GROUP BY
    1
) AS "t7"
ORDER BY
  "t7"."custdist" DESC NULLS LAST,
  "t7"."c_count" DESC NULLS LAST