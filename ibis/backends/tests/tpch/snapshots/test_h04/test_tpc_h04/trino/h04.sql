SELECT
  *
FROM (
  SELECT
    "t5"."o_orderpriority",
    COUNT(*) AS "order_count"
  FROM (
    SELECT
      *
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
        FROM (
          SELECT
            "t1"."l_orderkey",
            "t1"."l_partkey",
            "t1"."l_suppkey",
            "t1"."l_linenumber",
            CAST("t1"."l_quantity" AS DECIMAL(15, 2)) AS "l_quantity",
            CAST("t1"."l_extendedprice" AS DECIMAL(15, 2)) AS "l_extendedprice",
            CAST("t1"."l_discount" AS DECIMAL(15, 2)) AS "l_discount",
            CAST("t1"."l_tax" AS DECIMAL(15, 2)) AS "l_tax",
            "t1"."l_returnflag",
            "t1"."l_linestatus",
            "t1"."l_shipdate",
            "t1"."l_commitdate",
            "t1"."l_receiptdate",
            "t1"."l_shipinstruct",
            "t1"."l_shipmode",
            "t1"."l_comment"
          FROM "hive"."ibis_sf1"."lineitem" AS "t1"
        ) AS "t3"
        WHERE
          (
            "t3"."l_orderkey" = "t2"."o_orderkey"
          )
          AND (
            "t3"."l_commitdate" < "t3"."l_receiptdate"
          )
      )
      AND "t2"."o_orderdate" >= FROM_ISO8601_DATE('1993-07-01')
      AND "t2"."o_orderdate" < FROM_ISO8601_DATE('1993-10-01')
  ) AS "t5"
  GROUP BY
    1
) AS "t6"
ORDER BY
  "t6"."o_orderpriority" ASC