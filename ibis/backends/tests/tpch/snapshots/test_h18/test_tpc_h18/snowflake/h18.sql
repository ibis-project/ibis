SELECT
  "t14"."c_name" AS "c_name",
  "t14"."c_custkey" AS "c_custkey",
  "t14"."o_orderkey" AS "o_orderkey",
  "t14"."o_orderdate" AS "o_orderdate",
  "t14"."o_totalprice" AS "o_totalprice",
  "t14"."sum_qty" AS "sum_qty"
FROM (
  SELECT
    "t13"."c_name" AS "c_name",
    "t13"."c_custkey" AS "c_custkey",
    "t13"."o_orderkey" AS "o_orderkey",
    "t13"."o_orderdate" AS "o_orderdate",
    "t13"."o_totalprice" AS "o_totalprice",
    SUM("t13"."l_quantity") AS "sum_qty"
  FROM (
    SELECT
      "t11"."c_custkey" AS "c_custkey",
      "t11"."c_name" AS "c_name",
      "t11"."c_address" AS "c_address",
      "t11"."c_nationkey" AS "c_nationkey",
      "t11"."c_phone" AS "c_phone",
      "t11"."c_acctbal" AS "c_acctbal",
      "t11"."c_mktsegment" AS "c_mktsegment",
      "t11"."c_comment" AS "c_comment",
      "t11"."o_orderkey" AS "o_orderkey",
      "t11"."o_custkey" AS "o_custkey",
      "t11"."o_orderstatus" AS "o_orderstatus",
      "t11"."o_totalprice" AS "o_totalprice",
      "t11"."o_orderdate" AS "o_orderdate",
      "t11"."o_orderpriority" AS "o_orderpriority",
      "t11"."o_clerk" AS "o_clerk",
      "t11"."o_shippriority" AS "o_shippriority",
      "t11"."o_comment" AS "o_comment",
      "t11"."l_orderkey" AS "l_orderkey",
      "t11"."l_partkey" AS "l_partkey",
      "t11"."l_suppkey" AS "l_suppkey",
      "t11"."l_linenumber" AS "l_linenumber",
      "t11"."l_quantity" AS "l_quantity",
      "t11"."l_extendedprice" AS "l_extendedprice",
      "t11"."l_discount" AS "l_discount",
      "t11"."l_tax" AS "l_tax",
      "t11"."l_returnflag" AS "l_returnflag",
      "t11"."l_linestatus" AS "l_linestatus",
      "t11"."l_shipdate" AS "l_shipdate",
      "t11"."l_commitdate" AS "l_commitdate",
      "t11"."l_receiptdate" AS "l_receiptdate",
      "t11"."l_shipinstruct" AS "l_shipinstruct",
      "t11"."l_shipmode" AS "l_shipmode",
      "t11"."l_comment" AS "l_comment"
    FROM (
      SELECT
        "t3"."c_custkey" AS "c_custkey",
        "t3"."c_name" AS "c_name",
        "t3"."c_address" AS "c_address",
        "t3"."c_nationkey" AS "c_nationkey",
        "t3"."c_phone" AS "c_phone",
        "t3"."c_acctbal" AS "c_acctbal",
        "t3"."c_mktsegment" AS "c_mktsegment",
        "t3"."c_comment" AS "c_comment",
        "t6"."o_orderkey" AS "o_orderkey",
        "t6"."o_custkey" AS "o_custkey",
        "t6"."o_orderstatus" AS "o_orderstatus",
        "t6"."o_totalprice" AS "o_totalprice",
        "t6"."o_orderdate" AS "o_orderdate",
        "t6"."o_orderpriority" AS "o_orderpriority",
        "t6"."o_clerk" AS "o_clerk",
        "t6"."o_shippriority" AS "o_shippriority",
        "t6"."o_comment" AS "o_comment",
        "t7"."l_orderkey" AS "l_orderkey",
        "t7"."l_partkey" AS "l_partkey",
        "t7"."l_suppkey" AS "l_suppkey",
        "t7"."l_linenumber" AS "l_linenumber",
        "t7"."l_quantity" AS "l_quantity",
        "t7"."l_extendedprice" AS "l_extendedprice",
        "t7"."l_discount" AS "l_discount",
        "t7"."l_tax" AS "l_tax",
        "t7"."l_returnflag" AS "l_returnflag",
        "t7"."l_linestatus" AS "l_linestatus",
        "t7"."l_shipdate" AS "l_shipdate",
        "t7"."l_commitdate" AS "l_commitdate",
        "t7"."l_receiptdate" AS "l_receiptdate",
        "t7"."l_shipinstruct" AS "l_shipinstruct",
        "t7"."l_shipmode" AS "l_shipmode",
        "t7"."l_comment" AS "l_comment"
      FROM (
        SELECT
          "t0"."C_CUSTKEY" AS "c_custkey",
          "t0"."C_NAME" AS "c_name",
          "t0"."C_ADDRESS" AS "c_address",
          "t0"."C_NATIONKEY" AS "c_nationkey",
          "t0"."C_PHONE" AS "c_phone",
          "t0"."C_ACCTBAL" AS "c_acctbal",
          "t0"."C_MKTSEGMENT" AS "c_mktsegment",
          "t0"."C_COMMENT" AS "c_comment"
        FROM "CUSTOMER" AS "t0"
      ) AS "t3"
      INNER JOIN (
        SELECT
          "t1"."O_ORDERKEY" AS "o_orderkey",
          "t1"."O_CUSTKEY" AS "o_custkey",
          "t1"."O_ORDERSTATUS" AS "o_orderstatus",
          "t1"."O_TOTALPRICE" AS "o_totalprice",
          "t1"."O_ORDERDATE" AS "o_orderdate",
          "t1"."O_ORDERPRIORITY" AS "o_orderpriority",
          "t1"."O_CLERK" AS "o_clerk",
          "t1"."O_SHIPPRIORITY" AS "o_shippriority",
          "t1"."O_COMMENT" AS "o_comment"
        FROM "ORDERS" AS "t1"
      ) AS "t6"
        ON "t3"."c_custkey" = "t6"."o_custkey"
      INNER JOIN (
        SELECT
          "t2"."L_ORDERKEY" AS "l_orderkey",
          "t2"."L_PARTKEY" AS "l_partkey",
          "t2"."L_SUPPKEY" AS "l_suppkey",
          "t2"."L_LINENUMBER" AS "l_linenumber",
          "t2"."L_QUANTITY" AS "l_quantity",
          "t2"."L_EXTENDEDPRICE" AS "l_extendedprice",
          "t2"."L_DISCOUNT" AS "l_discount",
          "t2"."L_TAX" AS "l_tax",
          "t2"."L_RETURNFLAG" AS "l_returnflag",
          "t2"."L_LINESTATUS" AS "l_linestatus",
          "t2"."L_SHIPDATE" AS "l_shipdate",
          "t2"."L_COMMITDATE" AS "l_commitdate",
          "t2"."L_RECEIPTDATE" AS "l_receiptdate",
          "t2"."L_SHIPINSTRUCT" AS "l_shipinstruct",
          "t2"."L_SHIPMODE" AS "l_shipmode",
          "t2"."L_COMMENT" AS "l_comment"
        FROM "LINEITEM" AS "t2"
      ) AS "t7"
        ON "t6"."o_orderkey" = "t7"."l_orderkey"
    ) AS "t11"
    WHERE
      "t11"."o_orderkey" IN ((
        SELECT
          "t8"."l_orderkey" AS "l_orderkey"
        FROM (
          SELECT
            "t5"."l_orderkey" AS "l_orderkey",
            SUM("t5"."l_quantity") AS "qty_sum"
          FROM (
            SELECT
              "t2"."L_ORDERKEY" AS "l_orderkey",
              "t2"."L_PARTKEY" AS "l_partkey",
              "t2"."L_SUPPKEY" AS "l_suppkey",
              "t2"."L_LINENUMBER" AS "l_linenumber",
              "t2"."L_QUANTITY" AS "l_quantity",
              "t2"."L_EXTENDEDPRICE" AS "l_extendedprice",
              "t2"."L_DISCOUNT" AS "l_discount",
              "t2"."L_TAX" AS "l_tax",
              "t2"."L_RETURNFLAG" AS "l_returnflag",
              "t2"."L_LINESTATUS" AS "l_linestatus",
              "t2"."L_SHIPDATE" AS "l_shipdate",
              "t2"."L_COMMITDATE" AS "l_commitdate",
              "t2"."L_RECEIPTDATE" AS "l_receiptdate",
              "t2"."L_SHIPINSTRUCT" AS "l_shipinstruct",
              "t2"."L_SHIPMODE" AS "l_shipmode",
              "t2"."L_COMMENT" AS "l_comment"
            FROM "LINEITEM" AS "t2"
          ) AS "t5"
          GROUP BY
            1
        ) AS "t8"
        WHERE
          "t8"."qty_sum" > 300
      ))
  ) AS "t13"
  GROUP BY
    1,
    2,
    3,
    4,
    5
) AS "t14"
ORDER BY
  "t14"."o_totalprice" DESC NULLS LAST,
  "t14"."o_orderdate" ASC
LIMIT 100