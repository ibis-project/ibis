SELECT
  "t6"."s_suppkey",
  "t6"."s_name",
  "t6"."s_address",
  "t6"."s_phone",
  "t6"."total_revenue"
FROM (
  SELECT
    "t2"."s_suppkey",
    "t2"."s_name",
    "t2"."s_address",
    "t2"."s_nationkey",
    "t2"."s_phone",
    "t2"."s_acctbal",
    "t2"."s_comment",
    "t5"."l_suppkey",
    "t5"."total_revenue"
  FROM "supplier" AS "t2"
  INNER JOIN (
    SELECT
      "t3"."l_suppkey",
      SUM("t3"."l_extendedprice" * (
        CAST(1 AS TINYINT) - "t3"."l_discount"
      )) AS "total_revenue"
    FROM (
      SELECT
        "t1"."l_orderkey",
        "t1"."l_partkey",
        "t1"."l_suppkey",
        "t1"."l_linenumber",
        "t1"."l_quantity",
        "t1"."l_extendedprice",
        "t1"."l_discount",
        "t1"."l_tax",
        "t1"."l_returnflag",
        "t1"."l_linestatus",
        "t1"."l_shipdate",
        "t1"."l_commitdate",
        "t1"."l_receiptdate",
        "t1"."l_shipinstruct",
        "t1"."l_shipmode",
        "t1"."l_comment"
      FROM "lineitem" AS "t1"
      WHERE
        "t1"."l_shipdate" >= MAKE_DATE(1996, 1, 1)
        AND "t1"."l_shipdate" < MAKE_DATE(1996, 4, 1)
    ) AS "t3"
    GROUP BY
      1
  ) AS "t5"
    ON "t2"."s_suppkey" = "t5"."l_suppkey"
) AS "t6"
WHERE
  "t6"."total_revenue" = (
    SELECT
      MAX("t6"."total_revenue") AS "Max(total_revenue)"
    FROM (
      SELECT
        "t2"."s_suppkey",
        "t2"."s_name",
        "t2"."s_address",
        "t2"."s_nationkey",
        "t2"."s_phone",
        "t2"."s_acctbal",
        "t2"."s_comment",
        "t5"."l_suppkey",
        "t5"."total_revenue"
      FROM "supplier" AS "t2"
      INNER JOIN (
        SELECT
          "t3"."l_suppkey",
          SUM("t3"."l_extendedprice" * (
            CAST(1 AS TINYINT) - "t3"."l_discount"
          )) AS "total_revenue"
        FROM (
          SELECT
            "t1"."l_orderkey",
            "t1"."l_partkey",
            "t1"."l_suppkey",
            "t1"."l_linenumber",
            "t1"."l_quantity",
            "t1"."l_extendedprice",
            "t1"."l_discount",
            "t1"."l_tax",
            "t1"."l_returnflag",
            "t1"."l_linestatus",
            "t1"."l_shipdate",
            "t1"."l_commitdate",
            "t1"."l_receiptdate",
            "t1"."l_shipinstruct",
            "t1"."l_shipmode",
            "t1"."l_comment"
          FROM "lineitem" AS "t1"
          WHERE
            "t1"."l_shipdate" >= MAKE_DATE(1996, 1, 1)
            AND "t1"."l_shipdate" < MAKE_DATE(1996, 4, 1)
        ) AS "t3"
        GROUP BY
          1
      ) AS "t5"
        ON "t2"."s_suppkey" = "t5"."l_suppkey"
    ) AS "t6"
  )
ORDER BY
  "t6"."s_suppkey" ASC