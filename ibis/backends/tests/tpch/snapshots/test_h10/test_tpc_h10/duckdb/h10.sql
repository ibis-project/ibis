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
      CAST(1 AS TINYINT) - "t9"."l_discount"
    )) AS "revenue"
  FROM (
    SELECT
      "t8"."c_custkey",
      "t8"."c_name",
      "t8"."c_address",
      "t8"."c_nationkey",
      "t8"."c_phone",
      "t8"."c_acctbal",
      "t8"."c_mktsegment",
      "t8"."c_comment",
      "t8"."o_orderkey",
      "t8"."o_custkey",
      "t8"."o_orderstatus",
      "t8"."o_totalprice",
      "t8"."o_orderdate",
      "t8"."o_orderpriority",
      "t8"."o_clerk",
      "t8"."o_shippriority",
      "t8"."o_comment",
      "t8"."l_orderkey",
      "t8"."l_partkey",
      "t8"."l_suppkey",
      "t8"."l_linenumber",
      "t8"."l_quantity",
      "t8"."l_extendedprice",
      "t8"."l_discount",
      "t8"."l_tax",
      "t8"."l_returnflag",
      "t8"."l_linestatus",
      "t8"."l_shipdate",
      "t8"."l_commitdate",
      "t8"."l_receiptdate",
      "t8"."l_shipinstruct",
      "t8"."l_shipmode",
      "t8"."l_comment",
      "t8"."n_nationkey",
      "t8"."n_name",
      "t8"."n_regionkey",
      "t8"."n_comment"
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
      "t8"."o_orderdate" >= MAKE_DATE(1993, 10, 1)
      AND "t8"."o_orderdate" < MAKE_DATE(1994, 1, 1)
      AND "t8"."l_returnflag" = 'R'
  ) AS "t9"
  GROUP BY
    1,
    2,
    3,
    4,
    5,
    6,
    7
) AS "t10"
ORDER BY
  "t10"."revenue" DESC
LIMIT 20