SELECT
  *
FROM (
  SELECT
    "t10"."l_orderkey" AS "l_orderkey",
    "t10"."revenue" AS "revenue",
    "t10"."o_orderdate" AS "o_orderdate",
    "t10"."o_shippriority" AS "o_shippriority"
  FROM (
    SELECT
      "t9"."l_orderkey" AS "l_orderkey",
      "t9"."o_orderdate" AS "o_orderdate",
      "t9"."o_shippriority" AS "o_shippriority",
      SUM("t9"."l_extendedprice" * (
        1 - "t9"."l_discount"
      )) AS "revenue"
    FROM (
      SELECT
        *
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
          "t4"."o_orderkey" AS "o_orderkey",
          "t4"."o_custkey" AS "o_custkey",
          "t4"."o_orderstatus" AS "o_orderstatus",
          "t4"."o_totalprice" AS "o_totalprice",
          "t4"."o_orderdate" AS "o_orderdate",
          "t4"."o_orderpriority" AS "o_orderpriority",
          "t4"."o_clerk" AS "o_clerk",
          "t4"."o_shippriority" AS "o_shippriority",
          "t4"."o_comment" AS "o_comment",
          "t5"."l_orderkey" AS "l_orderkey",
          "t5"."l_partkey" AS "l_partkey",
          "t5"."l_suppkey" AS "l_suppkey",
          "t5"."l_linenumber" AS "l_linenumber",
          "t5"."l_quantity" AS "l_quantity",
          "t5"."l_extendedprice" AS "l_extendedprice",
          "t5"."l_discount" AS "l_discount",
          "t5"."l_tax" AS "l_tax",
          "t5"."l_returnflag" AS "l_returnflag",
          "t5"."l_linestatus" AS "l_linestatus",
          "t5"."l_shipdate" AS "l_shipdate",
          "t5"."l_commitdate" AS "l_commitdate",
          "t5"."l_receiptdate" AS "l_receiptdate",
          "t5"."l_shipinstruct" AS "l_shipinstruct",
          "t5"."l_shipmode" AS "l_shipmode",
          "t5"."l_comment" AS "l_comment"
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
        ) AS "t4"
          ON "t3"."c_custkey" = "t4"."o_custkey"
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
        ) AS "t5"
          ON "t5"."l_orderkey" = "t4"."o_orderkey"
      ) AS "t8"
      WHERE
        (
          "t8"."c_mktsegment" = 'BUILDING'
        )
        AND (
          "t8"."o_orderdate" < DATEFROMPARTS(1995, 3, 15)
        )
        AND (
          "t8"."l_shipdate" > DATEFROMPARTS(1995, 3, 15)
        )
    ) AS "t9"
    GROUP BY
      1,
      2,
      3
  ) AS "t10"
) AS "t11"
ORDER BY
  "t11"."revenue" DESC NULLS LAST,
  "t11"."o_orderdate" ASC
LIMIT 10