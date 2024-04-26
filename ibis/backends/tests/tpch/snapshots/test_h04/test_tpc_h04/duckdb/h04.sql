SELECT
  *
FROM (
  SELECT
    "t3"."o_orderpriority",
    COUNT(*) AS "order_count"
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
      AND "t0"."o_orderdate" >= MAKE_DATE(1993, 7, 1)
      AND "t0"."o_orderdate" < MAKE_DATE(1993, 10, 1)
  ) AS "t3"
  GROUP BY
    1
) AS "t4"
ORDER BY
  "t4"."o_orderpriority" ASC