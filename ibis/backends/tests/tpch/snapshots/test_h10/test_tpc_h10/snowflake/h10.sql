SELECT
  *
FROM (
  SELECT
    "t13"."c_custkey" AS "c_custkey",
    "t13"."c_name" AS "c_name",
    "t13"."revenue" AS "revenue",
    "t13"."c_acctbal" AS "c_acctbal",
    "t13"."n_name" AS "n_name",
    "t13"."c_address" AS "c_address",
    "t13"."c_phone" AS "c_phone",
    "t13"."c_comment" AS "c_comment"
  FROM (
    SELECT
      "t12"."c_custkey" AS "c_custkey",
      "t12"."c_name" AS "c_name",
      "t12"."c_acctbal" AS "c_acctbal",
      "t12"."n_name" AS "n_name",
      "t12"."c_address" AS "c_address",
      "t12"."c_phone" AS "c_phone",
      "t12"."c_comment" AS "c_comment",
      SUM("t12"."l_extendedprice" * (
        1 - "t12"."l_discount"
      )) AS "revenue"
    FROM (
      SELECT
        *
      FROM (
        SELECT
          "t4"."c_custkey" AS "c_custkey",
          "t4"."c_name" AS "c_name",
          "t4"."c_address" AS "c_address",
          "t4"."c_nationkey" AS "c_nationkey",
          "t4"."c_phone" AS "c_phone",
          "t4"."c_acctbal" AS "c_acctbal",
          "t4"."c_mktsegment" AS "c_mktsegment",
          "t4"."c_comment" AS "c_comment",
          "t5"."o_orderkey" AS "o_orderkey",
          "t5"."o_custkey" AS "o_custkey",
          "t5"."o_orderstatus" AS "o_orderstatus",
          "t5"."o_totalprice" AS "o_totalprice",
          "t5"."o_orderdate" AS "o_orderdate",
          "t5"."o_orderpriority" AS "o_orderpriority",
          "t5"."o_clerk" AS "o_clerk",
          "t5"."o_shippriority" AS "o_shippriority",
          "t5"."o_comment" AS "o_comment",
          "t6"."l_orderkey" AS "l_orderkey",
          "t6"."l_partkey" AS "l_partkey",
          "t6"."l_suppkey" AS "l_suppkey",
          "t6"."l_linenumber" AS "l_linenumber",
          "t6"."l_quantity" AS "l_quantity",
          "t6"."l_extendedprice" AS "l_extendedprice",
          "t6"."l_discount" AS "l_discount",
          "t6"."l_tax" AS "l_tax",
          "t6"."l_returnflag" AS "l_returnflag",
          "t6"."l_linestatus" AS "l_linestatus",
          "t6"."l_shipdate" AS "l_shipdate",
          "t6"."l_commitdate" AS "l_commitdate",
          "t6"."l_receiptdate" AS "l_receiptdate",
          "t6"."l_shipinstruct" AS "l_shipinstruct",
          "t6"."l_shipmode" AS "l_shipmode",
          "t6"."l_comment" AS "l_comment",
          "t7"."n_nationkey" AS "n_nationkey",
          "t7"."n_name" AS "n_name",
          "t7"."n_regionkey" AS "n_regionkey",
          "t7"."n_comment" AS "n_comment"
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
        ) AS "t4"
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
        ) AS "t5"
          ON "t4"."c_custkey" = "t5"."o_custkey"
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
        ) AS "t6"
          ON "t6"."l_orderkey" = "t5"."o_orderkey"
        INNER JOIN (
          SELECT
            "t3"."N_NATIONKEY" AS "n_nationkey",
            "t3"."N_NAME" AS "n_name",
            "t3"."N_REGIONKEY" AS "n_regionkey",
            "t3"."N_COMMENT" AS "n_comment"
          FROM "NATION" AS "t3"
        ) AS "t7"
          ON "t4"."c_nationkey" = "t7"."n_nationkey"
      ) AS "t11"
      WHERE
        (
          "t11"."o_orderdate" >= DATEFROMPARTS(1993, 10, 1)
        )
        AND (
          "t11"."o_orderdate" < DATEFROMPARTS(1994, 1, 1)
        )
        AND (
          "t11"."l_returnflag" = 'R'
        )
    ) AS "t12"
    GROUP BY
      1,
      2,
      3,
      4,
      5,
      6,
      7
  ) AS "t13"
) AS "t14"
ORDER BY
  "t14"."revenue" DESC NULLS LAST
LIMIT 20