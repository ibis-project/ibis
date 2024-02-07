SELECT
  "t5"."o_orderpriority",
  "t5"."order_count"
FROM (
  SELECT
    "t4"."o_orderpriority",
    COUNT(*) AS "order_count"
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
      "t2"."o_comment"
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
    ) AS "t2"
    WHERE
      EXISTS(
        SELECT
          1
        FROM "hive"."ibis_sf1"."lineitem" AS "t1"
        WHERE
          (
            "t1"."l_orderkey" = "t2"."o_orderkey"
          )
          AND (
            "t1"."l_commitdate" < "t1"."l_receiptdate"
          )
      )
      AND "t2"."o_orderdate" >= FROM_ISO8601_DATE('1993-07-01')
      AND "t2"."o_orderdate" < FROM_ISO8601_DATE('1993-10-01')
  ) AS "t4"
  GROUP BY
    1
) AS "t5"
ORDER BY
  "t5"."o_orderpriority" ASC