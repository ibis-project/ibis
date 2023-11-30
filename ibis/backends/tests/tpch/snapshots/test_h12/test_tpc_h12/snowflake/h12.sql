SELECT
  *
FROM (
  SELECT
    "t6"."l_shipmode" AS "l_shipmode",
    SUM(
      CASE "t6"."o_orderpriority" WHEN '1-URGENT' THEN 1 WHEN '2-HIGH' THEN 1 ELSE 0 END
    ) AS "high_line_count",
    SUM(
      CASE "t6"."o_orderpriority" WHEN '1-URGENT' THEN 0 WHEN '2-HIGH' THEN 0 ELSE 1 END
    ) AS "low_line_count"
  FROM (
    SELECT
      *
    FROM (
      SELECT
        "t2"."o_orderkey" AS "o_orderkey",
        "t2"."o_custkey" AS "o_custkey",
        "t2"."o_orderstatus" AS "o_orderstatus",
        "t2"."o_totalprice" AS "o_totalprice",
        "t2"."o_orderdate" AS "o_orderdate",
        "t2"."o_orderpriority" AS "o_orderpriority",
        "t2"."o_clerk" AS "o_clerk",
        "t2"."o_shippriority" AS "o_shippriority",
        "t2"."o_comment" AS "o_comment",
        "t3"."l_orderkey" AS "l_orderkey",
        "t3"."l_partkey" AS "l_partkey",
        "t3"."l_suppkey" AS "l_suppkey",
        "t3"."l_linenumber" AS "l_linenumber",
        "t3"."l_quantity" AS "l_quantity",
        "t3"."l_extendedprice" AS "l_extendedprice",
        "t3"."l_discount" AS "l_discount",
        "t3"."l_tax" AS "l_tax",
        "t3"."l_returnflag" AS "l_returnflag",
        "t3"."l_linestatus" AS "l_linestatus",
        "t3"."l_shipdate" AS "l_shipdate",
        "t3"."l_commitdate" AS "l_commitdate",
        "t3"."l_receiptdate" AS "l_receiptdate",
        "t3"."l_shipinstruct" AS "l_shipinstruct",
        "t3"."l_shipmode" AS "l_shipmode",
        "t3"."l_comment" AS "l_comment"
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
      ) AS "t2"
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
      ) AS "t3"
        ON "t2"."o_orderkey" = "t3"."l_orderkey"
    ) AS "t5"
    WHERE
      "t5"."l_shipmode" IN ('MAIL', 'SHIP')
      AND (
        "t5"."l_commitdate" < "t5"."l_receiptdate"
      )
      AND (
        "t5"."l_shipdate" < "t5"."l_commitdate"
      )
      AND (
        "t5"."l_receiptdate" >= DATEFROMPARTS(1994, 1, 1)
      )
      AND (
        "t5"."l_receiptdate" < DATEFROMPARTS(1995, 1, 1)
      )
  ) AS "t6"
  GROUP BY
    1
) AS "t7"
ORDER BY
  "t7"."l_shipmode" ASC