SELECT
  *
FROM (
  SELECT
    "t9"."c_name",
    "t9"."c_custkey",
    "t9"."o_orderkey",
    "t9"."o_orderdate",
    "t9"."o_totalprice",
    SUM("t9"."l_quantity") AS "sum_qty"
  FROM (
    SELECT
      *
    FROM (
      SELECT
        "t3"."c_custkey",
        "t3"."c_name",
        "t3"."c_address",
        "t3"."c_nationkey",
        "t3"."c_phone",
        "t3"."c_acctbal",
        "t3"."c_mktsegment",
        "t3"."c_comment",
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
      FROM "customer" AS "t3"
      INNER JOIN "orders" AS "t4"
        ON "t3"."c_custkey" = "t4"."o_custkey"
      INNER JOIN "lineitem" AS "t5"
        ON "t4"."o_orderkey" = "t5"."l_orderkey"
    ) AS "t7"
    WHERE
      "t7"."o_orderkey" IN (
        SELECT
          "t6"."l_orderkey"
        FROM (
          SELECT
            "t2"."l_orderkey",
            SUM("t2"."l_quantity") AS "qty_sum"
          FROM "lineitem" AS "t2"
          GROUP BY
            1
        ) AS "t6"
        WHERE
          "t6"."qty_sum" > CAST(300 AS SMALLINT)
      )
  ) AS "t9"
  GROUP BY
    1,
    2,
    3,
    4,
    5
) AS "t10"
ORDER BY
  "t10"."o_totalprice" DESC,
  "t10"."o_orderdate" ASC
LIMIT 100