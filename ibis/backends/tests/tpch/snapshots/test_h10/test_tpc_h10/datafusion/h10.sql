SELECT
  "t10"."c_custkey",
  "t10"."c_name",
  "t10"."revenue",
  "t10"."c_acctbal",
  "t10"."n_name",
  "t10"."c_address",
  "t10"."c_phone",
  "t10"."c_comment"
FROM (
  SELECT
    "t9"."c_custkey",
    "t9"."c_name",
    "t9"."c_acctbal",
    "t9"."n_name",
    "t9"."c_address",
    "t9"."c_phone",
    "t9"."c_comment",
    SUM("t9"."l_extendedprice" * (
      1 - "t9"."l_discount"
    )) AS "revenue"
  FROM (
    SELECT
      "t9"."c_nationkey",
      "t9"."c_mktsegment",
      "t9"."o_orderkey",
      "t9"."o_custkey",
      "t9"."o_orderstatus",
      "t9"."o_totalprice",
      "t9"."o_orderdate",
      "t9"."o_orderpriority",
      "t9"."o_clerk",
      "t9"."o_shippriority",
      "t9"."o_comment",
      "t9"."l_orderkey",
      "t9"."l_partkey",
      "t9"."l_suppkey",
      "t9"."l_linenumber",
      "t9"."l_quantity",
      "t9"."l_extendedprice",
      "t9"."l_discount",
      "t9"."l_tax",
      "t9"."l_returnflag",
      "t9"."l_linestatus",
      "t9"."l_shipdate",
      "t9"."l_commitdate",
      "t9"."l_receiptdate",
      "t9"."l_shipinstruct",
      "t9"."l_shipmode",
      "t9"."l_comment",
      "t9"."n_nationkey",
      "t9"."n_regionkey",
      "t9"."n_comment",
      "t9"."c_custkey",
      "t9"."c_name",
      "t9"."c_acctbal",
      "t9"."n_name",
      "t9"."c_address",
      "t9"."c_phone",
      "t9"."c_comment"
    FROM (
      SELECT
        *
      FROM (
        SELECT
          "t4"."c_custkey",
          "t4"."c_name",
          "t4"."c_address",
          "t4"."c_nationkey",
          "t4"."c_phone",
          "t4"."c_acctbal",
          "t4"."c_mktsegment",
          "t4"."c_comment",
          "t5"."o_orderkey",
          "t5"."o_custkey",
          "t5"."o_orderstatus",
          "t5"."o_totalprice",
          "t5"."o_orderdate",
          "t5"."o_orderpriority",
          "t5"."o_clerk",
          "t5"."o_shippriority",
          "t5"."o_comment",
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
          "t6"."l_comment",
          "t7"."n_nationkey",
          "t7"."n_name",
          "t7"."n_regionkey",
          "t7"."n_comment"
        FROM "customer" AS "t4"
        INNER JOIN "orders" AS "t5"
          ON "t4"."c_custkey" = "t5"."o_custkey"
        INNER JOIN "lineitem" AS "t6"
          ON "t6"."l_orderkey" = "t5"."o_orderkey"
        INNER JOIN "nation" AS "t7"
          ON "t4"."c_nationkey" = "t7"."n_nationkey"
      ) AS "t8"
      WHERE
        "t8"."o_orderdate" >= DATE_TRUNC('DAY', '1993-10-01')
        AND "t8"."o_orderdate" < DATE_TRUNC('DAY', '1994-01-01')
        AND "t8"."l_returnflag" = 'R'
    ) AS "t9"
  ) AS t9
  GROUP BY
    "t9"."c_custkey",
    "t9"."c_name",
    "t9"."c_acctbal",
    "t9"."n_name",
    "t9"."c_address",
    "t9"."c_phone",
    "t9"."c_comment"
) AS "t10"
ORDER BY
  "t10"."revenue" DESC NULLS LAST
LIMIT 20