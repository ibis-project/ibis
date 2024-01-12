SELECT
  "t9"."l_shipmode",
  "t9"."high_line_count",
  "t9"."low_line_count"
FROM (
  SELECT
    "t8"."l_shipmode",
    SUM(
      CASE "t8"."o_orderpriority" WHEN '1-URGENT' THEN 1 WHEN '2-HIGH' THEN 1 ELSE 0 END
    ) AS "high_line_count",
    SUM(
      CASE "t8"."o_orderpriority" WHEN '1-URGENT' THEN 0 WHEN '2-HIGH' THEN 0 ELSE 1 END
    ) AS "low_line_count"
  FROM (
    SELECT
      "t7"."o_orderkey",
      "t7"."o_custkey",
      "t7"."o_orderstatus",
      "t7"."o_totalprice",
      "t7"."o_orderdate",
      "t7"."o_orderpriority",
      "t7"."o_clerk",
      "t7"."o_shippriority",
      "t7"."o_comment",
      "t7"."l_orderkey",
      "t7"."l_partkey",
      "t7"."l_suppkey",
      "t7"."l_linenumber",
      "t7"."l_quantity",
      "t7"."l_extendedprice",
      "t7"."l_discount",
      "t7"."l_tax",
      "t7"."l_returnflag",
      "t7"."l_linestatus",
      "t7"."l_shipdate",
      "t7"."l_commitdate",
      "t7"."l_receiptdate",
      "t7"."l_shipinstruct",
      "t7"."l_shipmode",
      "t7"."l_comment"
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
      ) AS "t4"
      INNER JOIN (
        SELECT
          "t1"."L_ORDERKEY" AS "l_orderkey",
          "t1"."L_PARTKEY" AS "l_partkey",
          "t1"."L_SUPPKEY" AS "l_suppkey",
          "t1"."L_LINENUMBER" AS "l_linenumber",
          "t1"."L_QUANTITY" AS "l_quantity",
          "t1"."L_EXTENDEDPRICE" AS "l_extendedprice",
          "t1"."L_DISCOUNT" AS "l_discount",
          "t1"."L_TAX" AS "l_tax",
          "t1"."L_RETURNFLAG" AS "l_returnflag",
          "t1"."L_LINESTATUS" AS "l_linestatus",
          "t1"."L_SHIPDATE" AS "l_shipdate",
          "t1"."L_COMMITDATE" AS "l_commitdate",
          "t1"."L_RECEIPTDATE" AS "l_receiptdate",
          "t1"."L_SHIPINSTRUCT" AS "l_shipinstruct",
          "t1"."L_SHIPMODE" AS "l_shipmode",
          "t1"."L_COMMENT" AS "l_comment"
        FROM "LINEITEM" AS "t1"
      ) AS "t5"
        ON "t4"."o_orderkey" = "t5"."l_orderkey"
    ) AS "t7"
    WHERE
      "t7"."l_shipmode" IN ('MAIL', 'SHIP')
      AND "t7"."l_commitdate" < "t7"."l_receiptdate"
      AND "t7"."l_shipdate" < "t7"."l_commitdate"
      AND "t7"."l_receiptdate" >= DATE_FROM_PARTS(1994, 1, 1)
      AND "t7"."l_receiptdate" < DATE_FROM_PARTS(1995, 1, 1)
  ) AS "t8"
  GROUP BY
    1
) AS "t9"
ORDER BY
  "t9"."l_shipmode" ASC