SELECT
  "t7"."s_suppkey",
  "t7"."s_name",
  "t7"."s_address",
  "t7"."s_phone",
  "t7"."total_revenue"
FROM (
  SELECT
    "t4"."s_suppkey",
    "t4"."s_name",
    "t4"."s_address",
    "t4"."s_nationkey",
    "t4"."s_phone",
    "t4"."s_acctbal",
    "t4"."s_comment",
    "t6"."l_suppkey",
    "t6"."total_revenue"
  FROM (
    SELECT
      "t0"."s_suppkey",
      "t0"."s_name",
      "t0"."s_address",
      "t0"."s_nationkey",
      "t0"."s_phone",
      CAST("t0"."s_acctbal" AS DECIMAL(15, 2)) AS "s_acctbal",
      "t0"."s_comment"
    FROM "hive"."ibis_sf1"."supplier" AS "t0"
  ) AS "t4"
  INNER JOIN (
    SELECT
      "t3"."l_suppkey",
      SUM("t3"."l_extendedprice" * (
        1 - "t3"."l_discount"
      )) AS "total_revenue"
    FROM (
      SELECT
        "t1"."l_orderkey",
        "t1"."l_partkey",
        "t1"."l_suppkey",
        "t1"."l_linenumber",
        CAST("t1"."l_quantity" AS DECIMAL(15, 2)) AS "l_quantity",
        CAST("t1"."l_extendedprice" AS DECIMAL(15, 2)) AS "l_extendedprice",
        CAST("t1"."l_discount" AS DECIMAL(15, 2)) AS "l_discount",
        CAST("t1"."l_tax" AS DECIMAL(15, 2)) AS "l_tax",
        "t1"."l_returnflag",
        "t1"."l_linestatus",
        "t1"."l_shipdate",
        "t1"."l_commitdate",
        "t1"."l_receiptdate",
        "t1"."l_shipinstruct",
        "t1"."l_shipmode",
        "t1"."l_comment"
      FROM "hive"."ibis_sf1"."lineitem" AS "t1"
      WHERE
        "t1"."l_shipdate" >= FROM_ISO8601_DATE('1996-01-01')
        AND "t1"."l_shipdate" < FROM_ISO8601_DATE('1996-04-01')
    ) AS "t3"
    GROUP BY
      1
  ) AS "t6"
    ON "t4"."s_suppkey" = "t6"."l_suppkey"
) AS "t7"
WHERE
  "t7"."total_revenue" = (
    SELECT
      MAX("t7"."total_revenue") AS "Max(total_revenue)"
    FROM (
      SELECT
        "t4"."s_suppkey",
        "t4"."s_name",
        "t4"."s_address",
        "t4"."s_nationkey",
        "t4"."s_phone",
        "t4"."s_acctbal",
        "t4"."s_comment",
        "t6"."l_suppkey",
        "t6"."total_revenue"
      FROM (
        SELECT
          "t0"."s_suppkey",
          "t0"."s_name",
          "t0"."s_address",
          "t0"."s_nationkey",
          "t0"."s_phone",
          CAST("t0"."s_acctbal" AS DECIMAL(15, 2)) AS "s_acctbal",
          "t0"."s_comment"
        FROM "hive"."ibis_sf1"."supplier" AS "t0"
      ) AS "t4"
      INNER JOIN (
        SELECT
          "t3"."l_suppkey",
          SUM("t3"."l_extendedprice" * (
            1 - "t3"."l_discount"
          )) AS "total_revenue"
        FROM (
          SELECT
            "t1"."l_orderkey",
            "t1"."l_partkey",
            "t1"."l_suppkey",
            "t1"."l_linenumber",
            CAST("t1"."l_quantity" AS DECIMAL(15, 2)) AS "l_quantity",
            CAST("t1"."l_extendedprice" AS DECIMAL(15, 2)) AS "l_extendedprice",
            CAST("t1"."l_discount" AS DECIMAL(15, 2)) AS "l_discount",
            CAST("t1"."l_tax" AS DECIMAL(15, 2)) AS "l_tax",
            "t1"."l_returnflag",
            "t1"."l_linestatus",
            "t1"."l_shipdate",
            "t1"."l_commitdate",
            "t1"."l_receiptdate",
            "t1"."l_shipinstruct",
            "t1"."l_shipmode",
            "t1"."l_comment"
          FROM "hive"."ibis_sf1"."lineitem" AS "t1"
          WHERE
            "t1"."l_shipdate" >= FROM_ISO8601_DATE('1996-01-01')
            AND "t1"."l_shipdate" < FROM_ISO8601_DATE('1996-04-01')
        ) AS "t3"
        GROUP BY
          1
      ) AS "t6"
        ON "t4"."s_suppkey" = "t6"."l_suppkey"
    ) AS "t7"
  )
ORDER BY
  "t7"."s_suppkey" ASC