WITH "t5" AS (
  SELECT
    "t2"."l_orderkey",
    "t2"."l_partkey",
    "t2"."l_suppkey",
    "t2"."l_linenumber",
    CAST("t2"."l_quantity" AS DECIMAL(15, 2)) AS "l_quantity",
    CAST("t2"."l_extendedprice" AS DECIMAL(15, 2)) AS "l_extendedprice",
    CAST("t2"."l_discount" AS DECIMAL(15, 2)) AS "l_discount",
    CAST("t2"."l_tax" AS DECIMAL(15, 2)) AS "l_tax",
    "t2"."l_returnflag",
    "t2"."l_linestatus",
    "t2"."l_shipdate",
    "t2"."l_commitdate",
    "t2"."l_receiptdate",
    "t2"."l_shipinstruct",
    "t2"."l_shipmode",
    "t2"."l_comment"
  FROM "hive"."ibis_sf1"."lineitem" AS "t2"
)
SELECT
  "t14"."c_name",
  "t14"."c_custkey",
  "t14"."o_orderkey",
  "t14"."o_orderdate",
  "t14"."o_totalprice",
  "t14"."sum_qty"
FROM (
  SELECT
    "t13"."c_name",
    "t13"."c_custkey",
    "t13"."o_orderkey",
    "t13"."o_orderdate",
    "t13"."o_totalprice",
    SUM("t13"."l_quantity") AS "sum_qty"
  FROM (
    SELECT
      "t11"."c_custkey",
      "t11"."c_name",
      "t11"."c_address",
      "t11"."c_nationkey",
      "t11"."c_phone",
      "t11"."c_acctbal",
      "t11"."c_mktsegment",
      "t11"."c_comment",
      "t11"."o_orderkey",
      "t11"."o_custkey",
      "t11"."o_orderstatus",
      "t11"."o_totalprice",
      "t11"."o_orderdate",
      "t11"."o_orderpriority",
      "t11"."o_clerk",
      "t11"."o_shippriority",
      "t11"."o_comment",
      "t11"."l_orderkey",
      "t11"."l_partkey",
      "t11"."l_suppkey",
      "t11"."l_linenumber",
      "t11"."l_quantity",
      "t11"."l_extendedprice",
      "t11"."l_discount",
      "t11"."l_tax",
      "t11"."l_returnflag",
      "t11"."l_linestatus",
      "t11"."l_shipdate",
      "t11"."l_commitdate",
      "t11"."l_receiptdate",
      "t11"."l_shipinstruct",
      "t11"."l_shipmode",
      "t11"."l_comment"
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
        "t9"."l_comment"
      FROM (
        SELECT
          "t0"."c_custkey",
          "t0"."c_name",
          "t0"."c_address",
          "t0"."c_nationkey",
          "t0"."c_phone",
          CAST("t0"."c_acctbal" AS DECIMAL(15, 2)) AS "c_acctbal",
          "t0"."c_mktsegment",
          "t0"."c_comment"
        FROM "hive"."ibis_sf1"."customer" AS "t0"
      ) AS "t6"
      INNER JOIN (
        SELECT
          "t1"."o_orderkey",
          "t1"."o_custkey",
          "t1"."o_orderstatus",
          CAST("t1"."o_totalprice" AS DECIMAL(15, 2)) AS "o_totalprice",
          "t1"."o_orderdate",
          "t1"."o_orderpriority",
          "t1"."o_clerk",
          "t1"."o_shippriority",
          "t1"."o_comment"
        FROM "hive"."ibis_sf1"."orders" AS "t1"
      ) AS "t7"
        ON "t6"."c_custkey" = "t7"."o_custkey"
      INNER JOIN "t5" AS "t9"
        ON "t7"."o_orderkey" = "t9"."l_orderkey"
    ) AS "t11"
    WHERE
      "t11"."o_orderkey" IN (
        SELECT
          "t10"."l_orderkey"
        FROM (
          SELECT
            "t8"."l_orderkey",
            SUM("t8"."l_quantity") AS "qty_sum"
          FROM "t5" AS "t8"
          GROUP BY
            1
        ) AS "t10"
        WHERE
          "t10"."qty_sum" > 300
      )
  ) AS "t13"
  GROUP BY
    1,
    2,
    3,
    4,
    5
) AS "t14"
ORDER BY
  "t14"."o_totalprice" DESC,
  "t14"."o_orderdate" ASC
LIMIT 100