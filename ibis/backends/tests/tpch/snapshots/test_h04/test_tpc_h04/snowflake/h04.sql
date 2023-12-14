SELECT
  "t4"."o_orderpriority" AS "o_orderpriority",
  "t4"."order_count" AS "order_count"
FROM (
  SELECT
    "t3"."o_orderpriority" AS "o_orderpriority",
    COUNT(*) AS "order_count"
  FROM (
    SELECT
      "t0"."O_ORDERKEY" AS "o_orderkey",
      "t0"."O_CUSTKEY" AS "o_custkey",
      "t0"."O_ORDERSTATUS" AS "o_orderstatus",
      "t0"."O_TOTALPRICE" AS "o_totalprice",
      "t0"."O_ORDERDATE" AS "o_orderdate",
      "t0"."O_ORDERPRIORITY" AS "o_orderpriority",
      "t0"."O_CLERK" AS "o_clerk",
      "t0"."O_SHIPPRIORITY" AS "o_shippriority",
      "t0"."O_COMMENT" AS "o_comment"
    FROM "ORDERS" AS "t0"
    WHERE
      EXISTS(
        (
          SELECT
            1 AS "1"
          FROM "LINEITEM" AS "t1"
          WHERE
            (
              "t1"."L_ORDERKEY" = "t0"."O_ORDERKEY"
            )
            AND (
              "t1"."L_COMMITDATE" < "t1"."L_RECEIPTDATE"
            )
        )
      )
      AND "t0"."O_ORDERDATE" >= DATEFROMPARTS(1993, 7, 1)
      AND "t0"."O_ORDERDATE" < DATEFROMPARTS(1993, 10, 1)
  ) AS "t3"
  GROUP BY
    1
) AS "t4"
ORDER BY
  "t4"."o_orderpriority" ASC