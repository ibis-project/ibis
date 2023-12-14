SELECT
  "t16"."c_custkey" AS "c_custkey",
  "t16"."c_name" AS "c_name",
  "t16"."revenue" AS "revenue",
  "t16"."c_acctbal" AS "c_acctbal",
  "t16"."n_name" AS "n_name",
  "t16"."c_address" AS "c_address",
  "t16"."c_phone" AS "c_phone",
  "t16"."c_comment" AS "c_comment"
FROM (
  SELECT
    "t15"."c_custkey" AS "c_custkey",
    "t15"."c_name" AS "c_name",
    "t15"."c_acctbal" AS "c_acctbal",
    "t15"."n_name" AS "n_name",
    "t15"."c_address" AS "c_address",
    "t15"."c_phone" AS "c_phone",
    "t15"."c_comment" AS "c_comment",
    SUM("t15"."l_extendedprice" * (
      1 - "t15"."l_discount"
    )) AS "revenue"
  FROM (
    SELECT
      "t14"."c_custkey" AS "c_custkey",
      "t14"."c_name" AS "c_name",
      "t14"."c_address" AS "c_address",
      "t14"."c_nationkey" AS "c_nationkey",
      "t14"."c_phone" AS "c_phone",
      "t14"."c_acctbal" AS "c_acctbal",
      "t14"."c_mktsegment" AS "c_mktsegment",
      "t14"."c_comment" AS "c_comment",
      "t14"."o_orderkey" AS "o_orderkey",
      "t14"."o_custkey" AS "o_custkey",
      "t14"."o_orderstatus" AS "o_orderstatus",
      "t14"."o_totalprice" AS "o_totalprice",
      "t14"."o_orderdate" AS "o_orderdate",
      "t14"."o_orderpriority" AS "o_orderpriority",
      "t14"."o_clerk" AS "o_clerk",
      "t14"."o_shippriority" AS "o_shippriority",
      "t14"."o_comment" AS "o_comment",
      "t14"."l_orderkey" AS "l_orderkey",
      "t14"."l_partkey" AS "l_partkey",
      "t14"."l_suppkey" AS "l_suppkey",
      "t14"."l_linenumber" AS "l_linenumber",
      "t14"."l_quantity" AS "l_quantity",
      "t14"."l_extendedprice" AS "l_extendedprice",
      "t14"."l_discount" AS "l_discount",
      "t14"."l_tax" AS "l_tax",
      "t14"."l_returnflag" AS "l_returnflag",
      "t14"."l_linestatus" AS "l_linestatus",
      "t14"."l_shipdate" AS "l_shipdate",
      "t14"."l_commitdate" AS "l_commitdate",
      "t14"."l_receiptdate" AS "l_receiptdate",
      "t14"."l_shipinstruct" AS "l_shipinstruct",
      "t14"."l_shipmode" AS "l_shipmode",
      "t14"."l_comment" AS "l_comment",
      "t14"."n_nationkey" AS "n_nationkey",
      "t14"."n_name" AS "n_name",
      "t14"."n_regionkey" AS "n_regionkey",
      "t14"."n_comment" AS "n_comment"
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
        "t8"."o_orderkey" AS "o_orderkey",
        "t8"."o_custkey" AS "o_custkey",
        "t8"."o_orderstatus" AS "o_orderstatus",
        "t8"."o_totalprice" AS "o_totalprice",
        "t8"."o_orderdate" AS "o_orderdate",
        "t8"."o_orderpriority" AS "o_orderpriority",
        "t8"."o_clerk" AS "o_clerk",
        "t8"."o_shippriority" AS "o_shippriority",
        "t8"."o_comment" AS "o_comment",
        "t9"."l_orderkey" AS "l_orderkey",
        "t9"."l_partkey" AS "l_partkey",
        "t9"."l_suppkey" AS "l_suppkey",
        "t9"."l_linenumber" AS "l_linenumber",
        "t9"."l_quantity" AS "l_quantity",
        "t9"."l_extendedprice" AS "l_extendedprice",
        "t9"."l_discount" AS "l_discount",
        "t9"."l_tax" AS "l_tax",
        "t9"."l_returnflag" AS "l_returnflag",
        "t9"."l_linestatus" AS "l_linestatus",
        "t9"."l_shipdate" AS "l_shipdate",
        "t9"."l_commitdate" AS "l_commitdate",
        "t9"."l_receiptdate" AS "l_receiptdate",
        "t9"."l_shipinstruct" AS "l_shipinstruct",
        "t9"."l_shipmode" AS "l_shipmode",
        "t9"."l_comment" AS "l_comment",
        "t10"."n_nationkey" AS "n_nationkey",
        "t10"."n_name" AS "n_name",
        "t10"."n_regionkey" AS "n_regionkey",
        "t10"."n_comment" AS "n_comment"
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
      ) AS "t8"
        ON "t4"."c_custkey" = "t8"."o_custkey"
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
      ) AS "t9"
        ON "t9"."l_orderkey" = "t8"."o_orderkey"
      INNER JOIN (
        SELECT
          "t3"."N_NATIONKEY" AS "n_nationkey",
          "t3"."N_NAME" AS "n_name",
          "t3"."N_REGIONKEY" AS "n_regionkey",
          "t3"."N_COMMENT" AS "n_comment"
        FROM "NATION" AS "t3"
      ) AS "t10"
        ON "t4"."c_nationkey" = "t10"."n_nationkey"
    ) AS "t14"
    WHERE
      "t14"."o_orderdate" >= DATEFROMPARTS(1993, 10, 1)
      AND "t14"."o_orderdate" < DATEFROMPARTS(1994, 1, 1)
      AND "t14"."l_returnflag" = 'R'
  ) AS "t15"
  GROUP BY
    1,
    2,
    3,
    4,
    5,
    6,
    7
) AS "t16"
ORDER BY
  "t16"."revenue" DESC NULLS LAST
LIMIT 20