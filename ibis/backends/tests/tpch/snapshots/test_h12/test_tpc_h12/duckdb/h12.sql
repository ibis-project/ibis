SELECT
  "t6"."l_shipmode",
  "t6"."high_line_count",
  "t6"."low_line_count"
FROM (
  SELECT
    "t5"."l_shipmode",
    SUM(
      CASE "t5"."o_orderpriority"
        WHEN '1-URGENT'
        THEN CAST(1 AS TINYINT)
        WHEN '2-HIGH'
        THEN CAST(1 AS TINYINT)
        ELSE CAST(0 AS TINYINT)
      END
    ) AS "high_line_count",
    SUM(
      CASE "t5"."o_orderpriority"
        WHEN '1-URGENT'
        THEN CAST(0 AS TINYINT)
        WHEN '2-HIGH'
        THEN CAST(0 AS TINYINT)
        ELSE CAST(1 AS TINYINT)
      END
    ) AS "low_line_count"
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
      "t4"."l_orderkey",
      "t4"."l_partkey",
      "t4"."l_suppkey",
      "t4"."l_linenumber",
      "t4"."l_quantity",
      "t4"."l_extendedprice",
      "t4"."l_discount",
      "t4"."l_tax",
      "t4"."l_returnflag",
      "t4"."l_linestatus",
      "t4"."l_shipdate",
      "t4"."l_commitdate",
      "t4"."l_receiptdate",
      "t4"."l_shipinstruct",
      "t4"."l_shipmode",
      "t4"."l_comment"
    FROM (
      SELECT
        "t2"."o_orderkey",
        "t2"."o_custkey",
        "t2"."o_orderstatus",
        "t2"."o_totalprice",
        "t2"."o_orderdate",
        "t2"."o_orderpriority",
        "t2"."o_clerk",
        "t2"."o_shippriority",
        "t2"."o_comment",
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
      FROM "orders" AS "t2"
      INNER JOIN "lineitem" AS "t3"
        ON "t2"."o_orderkey" = "t3"."l_orderkey"
    ) AS "t4"
    WHERE
      "t4"."l_shipmode" IN ('MAIL', 'SHIP')
      AND "t4"."l_commitdate" < "t4"."l_receiptdate"
      AND "t4"."l_shipdate" < "t4"."l_commitdate"
      AND "t4"."l_receiptdate" >= MAKE_DATE(1994, 1, 1)
      AND "t4"."l_receiptdate" < MAKE_DATE(1995, 1, 1)
  ) AS "t5"
  GROUP BY
    1
) AS "t6"
ORDER BY
  "t6"."l_shipmode" ASC