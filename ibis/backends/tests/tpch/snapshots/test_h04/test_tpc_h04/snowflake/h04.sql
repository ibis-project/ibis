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
        "t0"."O_ORDERKEY" AS "o_orderkey",
        "t0"."O_CUSTKEY" AS "o_custkey",
        "t0"."O_ORDERSTATUS" AS "o_orderstatus",
        "t0"."O_TOTALPRICE" AS "o_totalprice",
        "t0"."O_ORDERDATE" AS "o_orderdate",
        "t0"."O_ORDERPRIORITY" AS "o_orderpriority",
        "t0"."O_CLERK" AS "o_clerk",
        "t0"."O_SHIPPRIORITY" AS "o_shippriority",
        "t0"."O_COMMENT" AS "o_comment"
      FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS "t0"
    ) AS "t2"
    WHERE
      EXISTS(
        SELECT
          1
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS "t1"
        WHERE
          (
            "t1"."L_ORDERKEY" = "t2"."o_orderkey"
          )
          AND (
            "t1"."L_COMMITDATE" < "t1"."L_RECEIPTDATE"
          )
      )
      AND "t2"."o_orderdate" >= DATE_FROM_PARTS(1993, 7, 1)
      AND "t2"."o_orderdate" < DATE_FROM_PARTS(1993, 10, 1)
  ) AS "t4"
  GROUP BY
    1
) AS "t5"
ORDER BY
  "t5"."o_orderpriority" ASC