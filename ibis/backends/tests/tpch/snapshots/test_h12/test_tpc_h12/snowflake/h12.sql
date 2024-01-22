SELECT
  "t8"."l_shipmode",
  "t8"."high_line_count",
  "t8"."low_line_count"
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
      "t6"."o_orderkey",
      "t6"."o_custkey",
      "t6"."o_orderstatus",
      "t6"."o_totalprice",
      "t6"."o_orderdate",
      "t6"."o_orderpriority",
      "t6"."o_clerk",
      "t6"."o_shippriority",
      "t6"."o_comment",
      "t6"."l_orderkey",
      "t6"."l_partkey",
      "t6"."l_suppkey",
      "t6"."l_linenumber",
      "t6"."l_quantity",
      "t6"."l_extendedprice",
      "t6"."l_discount",
      "t6"."l_tax",
      "t6"."l_returnflag",
      "t6"."l_linestatus",
      "t6"."l_shipdate",
      "t6"."l_commitdate",
      "t6"."l_receiptdate",
      "t6"."l_shipinstruct",
      "t6"."l_shipmode",
      "t6"."l_comment"
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
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS "t0"
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
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS "t1"
      ) AS "t5"
        ON "t4"."o_orderkey" = "t5"."l_orderkey"
    ) AS "t6"
    WHERE
      "t6"."l_shipmode" IN ('MAIL', 'SHIP')
      AND "t6"."l_commitdate" < "t6"."l_receiptdate"
      AND "t6"."l_shipdate" < "t6"."l_commitdate"
      AND "t6"."l_receiptdate" >= DATE_FROM_PARTS(1994, 1, 1)
      AND "t6"."l_receiptdate" < DATE_FROM_PARTS(1995, 1, 1)
  ) AS "t7"
  GROUP BY
    1
) AS "t8"
ORDER BY
  "t8"."l_shipmode" ASC