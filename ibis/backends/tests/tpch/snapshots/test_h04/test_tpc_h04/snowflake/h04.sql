SELECT
  *
FROM (
  SELECT
    "t6"."o_orderpriority" AS "o_orderpriority",
    COUNT(*) AS "order_count"
  FROM (
    SELECT
      *
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
    WHERE
      EXISTS(
        (
          SELECT
            1 AS "1"
          FROM (
            SELECT
              *
            FROM (
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
            WHERE
              (
                (
                  "t3"."l_orderkey" = "t2"."o_orderkey"
                )
                AND (
                  "t3"."l_commitdate" < "t3"."l_receiptdate"
                )
              )
          ) AS "t4"
        )
      )
      AND (
        "t2"."o_orderdate" >= DATEFROMPARTS(1993, 7, 1)
      )
      AND (
        "t2"."o_orderdate" < DATEFROMPARTS(1993, 10, 1)
      )
  ) AS "t6"
  GROUP BY
    1
) AS "t7"
ORDER BY
  "t7"."o_orderpriority" ASC