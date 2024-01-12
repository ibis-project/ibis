SELECT
  "t15"."c_name",
  "t15"."c_custkey",
  "t15"."o_orderkey",
  "t15"."o_orderdate",
  "t15"."o_totalprice",
  "t15"."sum_qty"
FROM (
  SELECT
    "t14"."c_name",
    "t14"."c_custkey",
    "t14"."o_orderkey",
    "t14"."o_orderdate",
    "t14"."o_totalprice",
    SUM("t14"."l_quantity") AS "sum_qty"
  FROM (
    SELECT
      "t12"."c_custkey",
      "t12"."c_name",
      "t12"."c_address",
      "t12"."c_nationkey",
      "t12"."c_phone",
      "t12"."c_acctbal",
      "t12"."c_mktsegment",
      "t12"."c_comment",
      "t12"."o_orderkey",
      "t12"."o_custkey",
      "t12"."o_orderstatus",
      "t12"."o_totalprice",
      "t12"."o_orderdate",
      "t12"."o_orderpriority",
      "t12"."o_clerk",
      "t12"."o_shippriority",
      "t12"."o_comment",
      "t12"."l_orderkey",
      "t12"."l_partkey",
      "t12"."l_suppkey",
      "t12"."l_linenumber",
      "t12"."l_quantity",
      "t12"."l_extendedprice",
      "t12"."l_discount",
      "t12"."l_tax",
      "t12"."l_returnflag",
      "t12"."l_linestatus",
      "t12"."l_shipdate",
      "t12"."l_commitdate",
      "t12"."l_receiptdate",
      "t12"."l_shipinstruct",
      "t12"."l_shipmode",
      "t12"."l_comment"
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
        "t8"."l_comment"
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
      INNER JOIN (
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
      ) AS "t8"
        ON "t7"."o_orderkey" = "t8"."l_orderkey"
    ) AS "t12"
    WHERE
      "t12"."o_orderkey" IN (
        SELECT
          "t9"."l_orderkey"
        FROM (
          SELECT
            "t5"."l_orderkey",
            SUM("t5"."l_quantity") AS "qty_sum"
          FROM (
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
          ) AS "t5"
          GROUP BY
            1
        ) AS "t9"
        WHERE
          "t9"."qty_sum" > 300
      )
  ) AS "t14"
  GROUP BY
    1,
    2,
    3,
    4,
    5
) AS "t15"
ORDER BY
  "t15"."o_totalprice" DESC,
  "t15"."o_orderdate" ASC
LIMIT 100