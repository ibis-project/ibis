SELECT
  "t8"."l_orderkey",
  "t8"."revenue",
  "t8"."o_orderdate",
  "t8"."o_shippriority"
FROM (
  SELECT
    "t7"."l_orderkey",
    "t7"."o_orderdate",
    "t7"."o_shippriority",
    SUM("t7"."l_extendedprice" * (
      CAST(1 AS TINYINT) - "t7"."l_discount"
    )) AS "revenue"
  FROM (
    SELECT
      "t6"."c_custkey",
      "t6"."c_name",
      "t6"."c_address",
      "t6"."c_nationkey",
      "t6"."c_phone",
      "t6"."c_acctbal",
      "t6"."c_mktsegment",
      "t6"."c_comment",
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
        ON "t5"."l_orderkey" = "t4"."o_orderkey"
    ) AS "t6"
    WHERE
      "t6"."c_mktsegment" = 'BUILDING'
      AND "t6"."o_orderdate" < MAKE_DATE(1995, 3, 15)
      AND "t6"."l_shipdate" > MAKE_DATE(1995, 3, 15)
  ) AS "t7"
  GROUP BY
    1,
    2,
    3
) AS "t8"
ORDER BY
  "t8"."revenue" DESC,
  "t8"."o_orderdate" ASC
LIMIT 10