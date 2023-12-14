SELECT
  "t8"."c_count" AS "c_count",
  "t8"."custdist" AS "custdist"
FROM (
  SELECT
    "t7"."c_count" AS "c_count",
    COUNT(*) AS "custdist"
  FROM (
    SELECT
      "t6"."c_custkey" AS "c_custkey",
      COUNT("t6"."o_orderkey") AS "c_count"
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
        "t4"."o_orderkey" AS "o_orderkey",
        "t4"."o_custkey" AS "o_custkey",
        "t4"."o_orderstatus" AS "o_orderstatus",
        "t4"."o_totalprice" AS "o_totalprice",
        "t4"."o_orderdate" AS "o_orderdate",
        "t4"."o_orderpriority" AS "o_orderpriority",
        "t4"."o_clerk" AS "o_clerk",
        "t4"."o_shippriority" AS "o_shippriority",
        "t4"."o_comment" AS "o_comment"
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
      ) AS "t4"
        ON "t2"."c_custkey" = "t4"."o_custkey"
        AND NOT (
          "t4"."o_comment" LIKE '%special%requests%'
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