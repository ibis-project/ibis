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
      1 - "t7"."l_discount"
    )) AS "revenue"
  FROM (
    SELECT
      "t7"."c_custkey",
      "t7"."c_name",
      "t7"."c_address",
      "t7"."c_nationkey",
      "t7"."c_phone",
      "t7"."c_acctbal",
      "t7"."c_mktsegment",
      "t7"."c_comment",
      "t7"."o_orderkey",
      "t7"."o_custkey",
      "t7"."o_orderstatus",
      "t7"."o_totalprice",
      "t7"."o_orderpriority",
      "t7"."o_clerk",
      "t7"."o_comment",
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
      "t7"."l_comment",
      "t7"."l_orderkey",
      "t7"."o_orderdate",
      "t7"."o_shippriority"
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
          ON "t5"."l_orderkey" = "t4"."o_orderkey"
      ) AS "t6"
      WHERE
        "t6"."c_mktsegment" = 'BUILDING'
        AND "t6"."o_orderdate" < DATE_TRUNC('DAY', '1995-03-15')
        AND "t6"."l_shipdate" > DATE_TRUNC('DAY', '1995-03-15')
    ) AS "t7"
  ) AS t7
  GROUP BY
    "t7"."l_orderkey",
    "t7"."o_orderdate",
    "t7"."o_shippriority"
) AS "t8"
ORDER BY
  "t8"."revenue" DESC NULLS LAST,
  "t8"."o_orderdate" ASC
LIMIT 10