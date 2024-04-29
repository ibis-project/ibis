SELECT
  *
FROM (
  SELECT
    "t13"."n_name",
    SUM("t13"."l_extendedprice" * (
      CAST(1 AS TINYINT) - "t13"."l_discount"
    )) AS "revenue"
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
      AND "t12"."o_orderdate" >= MAKE_DATE(1994, 1, 1)
      AND "t12"."o_orderdate" < MAKE_DATE(1995, 1, 1)
  ) AS "t13"
  GROUP BY
    1
) AS "t14"
ORDER BY
  "t14"."revenue" DESC