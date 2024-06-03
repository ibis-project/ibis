SELECT
  *
FROM (
  SELECT
    "t13"."n_name",
    SUM("t13"."l_extendedprice" * (
      1 - "t13"."l_discount"
    )) AS "revenue"
  FROM (
    SELECT
      "t13"."c_custkey",
      "t13"."c_name",
      "t13"."c_address",
      "t13"."c_nationkey",
      "t13"."c_phone",
      "t13"."c_acctbal",
      "t13"."c_mktsegment",
      "t13"."c_comment",
      "t13"."o_orderkey",
      "t13"."o_custkey",
      "t13"."o_orderstatus",
      "t13"."o_totalprice",
      "t13"."o_orderdate",
      "t13"."o_orderpriority",
      "t13"."o_clerk",
      "t13"."o_shippriority",
      "t13"."o_comment",
      "t13"."l_orderkey",
      "t13"."l_partkey",
      "t13"."l_suppkey",
      "t13"."l_linenumber",
      "t13"."l_quantity",
      "t13"."l_extendedprice",
      "t13"."l_discount",
      "t13"."l_tax",
      "t13"."l_returnflag",
      "t13"."l_linestatus",
      "t13"."l_shipdate",
      "t13"."l_commitdate",
      "t13"."l_receiptdate",
      "t13"."l_shipinstruct",
      "t13"."l_shipmode",
      "t13"."l_comment",
      "t13"."s_suppkey",
      "t13"."s_name",
      "t13"."s_address",
      "t13"."s_nationkey",
      "t13"."s_phone",
      "t13"."s_acctbal",
      "t13"."s_comment",
      "t13"."n_nationkey",
      "t13"."n_regionkey",
      "t13"."n_comment",
      "t13"."r_regionkey",
      "t13"."r_name",
      "t13"."r_comment",
      "t13"."n_name"
    FROM (
      SELECT
        *
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
          "t7"."o_orderkey",
          "t7"."o_custkey",
          "t7"."o_orderstatus",
          "t7"."o_totalprice",
          "t7"."o_orderdate",
          "t7"."o_orderpriority",
          "t7"."o_clerk",
          "t7"."o_shippriority",
          "t7"."o_comment",
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
          "t9"."s_suppkey",
          "t9"."s_name",
          "t9"."s_address",
          "t9"."s_nationkey",
          "t9"."s_phone",
          "t9"."s_acctbal",
          "t9"."s_comment",
          "t10"."n_nationkey",
          "t10"."n_name",
          "t10"."n_regionkey",
          "t10"."n_comment",
          "t11"."r_regionkey",
          "t11"."r_name",
          "t11"."r_comment"
        FROM "customer" AS "t6"
        INNER JOIN "orders" AS "t7"
          ON "t6"."c_custkey" = "t7"."o_custkey"
        INNER JOIN "lineitem" AS "t8"
          ON "t8"."l_orderkey" = "t7"."o_orderkey"
        INNER JOIN "supplier" AS "t9"
          ON "t8"."l_suppkey" = "t9"."s_suppkey"
        INNER JOIN "nation" AS "t10"
          ON "t6"."c_nationkey" = "t9"."s_nationkey"
          AND "t9"."s_nationkey" = "t10"."n_nationkey"
        INNER JOIN "region" AS "t11"
          ON "t10"."n_regionkey" = "t11"."r_regionkey"
      ) AS "t12"
      WHERE
        "t12"."r_name" = 'ASIA'
        AND "t12"."o_orderdate" >= DATE_TRUNC('DAY', '1994-01-01')
        AND "t12"."o_orderdate" < DATE_TRUNC('DAY', '1995-01-01')
    ) AS "t13"
  ) AS t13
  GROUP BY
    "t13"."n_name"
) AS "t14"
ORDER BY
  "t14"."revenue" DESC NULLS LAST