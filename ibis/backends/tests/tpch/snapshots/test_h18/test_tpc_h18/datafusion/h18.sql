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
      "t9"."c_address",
      "t9"."c_nationkey",
      "t9"."c_phone",
      "t9"."c_acctbal",
      "t9"."c_mktsegment",
      "t9"."c_comment",
      "t9"."o_custkey",
      "t9"."o_orderstatus",
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
      "t9"."c_name",
      "t9"."c_custkey",
      "t9"."o_orderkey",
      "t9"."o_orderdate",
      "t9"."o_totalprice"
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
            FROM (
              SELECT
                "t2"."l_partkey",
                "t2"."l_suppkey",
                "t2"."l_linenumber",
                "t2"."l_quantity",
                "t2"."l_extendedprice",
                "t2"."l_discount",
                "t2"."l_tax",
                "t2"."l_returnflag",
                "t2"."l_linestatus",
                "t2"."l_shipdate",
                "t2"."l_commitdate",
                "t2"."l_receiptdate",
                "t2"."l_shipinstruct",
                "t2"."l_shipmode",
                "t2"."l_comment",
                "t2"."l_orderkey"
              FROM "lineitem" AS "t2"
            ) AS t2
            GROUP BY
              "t2"."l_orderkey"
          ) AS "t6"
          WHERE
            "t6"."qty_sum" > 300
        )
    ) AS "t9"
  ) AS t9
  GROUP BY
    "t9"."c_name",
    "t9"."c_custkey",
    "t9"."o_orderkey",
    "t9"."o_orderdate",
    "t9"."o_totalprice"
) AS "t10"
ORDER BY
  "t10"."o_totalprice" DESC NULLS LAST,
  "t10"."o_orderdate" ASC
LIMIT 100