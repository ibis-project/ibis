SELECT
  *
FROM (
  SELECT
    "t3"."o_orderpriority",
    COUNT(*) AS "order_count"
  FROM (
    SELECT
      "t3"."o_orderkey",
      "t3"."o_custkey",
      "t3"."o_orderstatus",
      "t3"."o_totalprice",
      "t3"."o_orderdate",
      "t3"."o_clerk",
      "t3"."o_shippriority",
      "t3"."o_comment",
      "t3"."o_orderpriority"
    FROM (
      SELECT
        *
      FROM "orders" AS "t0"
      WHERE
        EXISTS(
          SELECT
            1
          FROM "lineitem" AS "t1"
          WHERE
            (
              "t1"."l_orderkey" = "t0"."o_orderkey"
            )
            AND (
              "t1"."l_commitdate" < "t1"."l_receiptdate"
            )
        )
        AND "t0"."o_orderdate" >= DATE_TRUNC('DAY', '1993-07-01')
        AND "t0"."o_orderdate" < DATE_TRUNC('DAY', '1993-10-01')
    ) AS "t3"
  ) AS t3
  GROUP BY
    "t3"."o_orderpriority"
) AS "t4"
ORDER BY
  "t4"."o_orderpriority" ASC