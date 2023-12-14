SELECT
  "t12"."l_orderkey" AS "l_orderkey",
  "t12"."revenue" AS "revenue",
  "t12"."o_orderdate" AS "o_orderdate",
  "t12"."o_shippriority" AS "o_shippriority"
FROM (
  SELECT
    "t11"."l_orderkey" AS "l_orderkey",
    "t11"."o_orderdate" AS "o_orderdate",
    "t11"."o_shippriority" AS "o_shippriority",
    SUM("t11"."l_extendedprice" * (
      1 - "t11"."l_discount"
    )) AS "revenue"
  FROM (
    SELECT
      "t10"."c_custkey" AS "c_custkey",
      "t10"."c_name" AS "c_name",
      "t10"."c_address" AS "c_address",
      "t10"."c_nationkey" AS "c_nationkey",
      "t10"."c_phone" AS "c_phone",
      "t10"."c_acctbal" AS "c_acctbal",
      "t10"."c_mktsegment" AS "c_mktsegment",
      "t10"."c_comment" AS "c_comment",
      "t10"."o_orderkey" AS "o_orderkey",
      "t10"."o_custkey" AS "o_custkey",
      "t10"."o_orderstatus" AS "o_orderstatus",
      "t10"."o_totalprice" AS "o_totalprice",
      "t10"."o_orderdate" AS "o_orderdate",
      "t10"."o_orderpriority" AS "o_orderpriority",
      "t10"."o_clerk" AS "o_clerk",
      "t10"."o_shippriority" AS "o_shippriority",
      "t10"."o_comment" AS "o_comment",
      "t10"."l_orderkey" AS "l_orderkey",
      "t10"."l_partkey" AS "l_partkey",
      "t10"."l_suppkey" AS "l_suppkey",
      "t10"."l_linenumber" AS "l_linenumber",
      "t10"."l_quantity" AS "l_quantity",
      "t10"."l_extendedprice" AS "l_extendedprice",
      "t10"."l_discount" AS "l_discount",
      "t10"."l_tax" AS "l_tax",
      "t10"."l_returnflag" AS "l_returnflag",
      "t10"."l_linestatus" AS "l_linestatus",
      "t10"."l_shipdate" AS "l_shipdate",
      "t10"."l_commitdate" AS "l_commitdate",
      "t10"."l_receiptdate" AS "l_receiptdate",
      "t10"."l_shipinstruct" AS "l_shipinstruct",
      "t10"."l_shipmode" AS "l_shipmode",
      "t10"."l_comment" AS "l_comment"
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
        ON "t7"."l_orderkey" = "t6"."o_orderkey"
    ) AS "t10"
    WHERE
      "t10"."c_mktsegment" = 'BUILDING'
      AND "t10"."o_orderdate" < DATEFROMPARTS(1995, 3, 15)
      AND "t10"."l_shipdate" > DATEFROMPARTS(1995, 3, 15)
  ) AS "t11"
  GROUP BY
    1,
    2,
    3
) AS "t12"
ORDER BY
  "t12"."revenue" DESC NULLS LAST,
  "t12"."o_orderdate" ASC
LIMIT 10