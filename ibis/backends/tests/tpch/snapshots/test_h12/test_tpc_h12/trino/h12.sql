SELECT
  *
FROM (
  SELECT
    "t7"."l_shipmode",
    SUM(
      CASE "t7"."o_orderpriority" WHEN '1-URGENT' THEN 1 WHEN '2-HIGH' THEN 1 ELSE 0 END
    ) AS "high_line_count",
    SUM(
      CASE "t7"."o_orderpriority" WHEN '1-URGENT' THEN 0 WHEN '2-HIGH' THEN 0 ELSE 1 END
    ) AS "low_line_count"
  FROM (
    SELECT
      *
    FROM (
      SELECT
        "t4"."o_orderkey",
        "t4"."o_custkey",
        "t4"."o_orderstatus",
        "t4"."o_totalprice",
        "t4"."o_orderdate",
        "t4"."o_orderpriority",
        "t4"."o_clerk",
        "t4"."o_shippriority",
        "t4"."o_comment",
        "t5"."l_orderkey",
        "t5"."l_partkey",
        "t5"."l_suppkey",
        "t5"."l_linenumber",
        "t5"."l_quantity",
        "t5"."l_extendedprice",
        "t5"."l_discount",
        "t5"."l_tax",
        "t5"."l_returnflag",
        "t5"."l_linestatus",
        "t5"."l_shipdate",
        "t5"."l_commitdate",
        "t5"."l_receiptdate",
        "t5"."l_shipinstruct",
        "t5"."l_shipmode",
        "t5"."l_comment"
      FROM (
        SELECT
          "t0"."o_orderkey",
          "t0"."o_custkey",
          "t0"."o_orderstatus",
          CAST("t0"."o_totalprice" AS DECIMAL(15, 2)) AS "o_totalprice",
          "t0"."o_orderdate",
          "t0"."o_orderpriority",
          "t0"."o_clerk",
          "t0"."o_shippriority",
          "t0"."o_comment"
        FROM "hive"."ibis_sf1"."orders" AS "t0"
      ) AS "t4"
      INNER JOIN (
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
      ) AS "t5"
        ON "t4"."o_orderkey" = "t5"."l_orderkey"
    ) AS "t6"
    WHERE
      "t6"."l_shipmode" IN ('MAIL', 'SHIP')
      AND "t6"."l_commitdate" < "t6"."l_receiptdate"
      AND "t6"."l_shipdate" < "t6"."l_commitdate"
      AND "t6"."l_receiptdate" >= FROM_ISO8601_DATE('1994-01-01')
      AND "t6"."l_receiptdate" < FROM_ISO8601_DATE('1995-01-01')
  ) AS "t7"
  GROUP BY
    1
) AS "t8"
ORDER BY
  "t8"."l_shipmode" ASC