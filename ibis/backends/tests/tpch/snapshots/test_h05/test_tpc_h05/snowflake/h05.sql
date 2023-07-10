SELECT
  "t24"."n_name" AS "n_name",
  "t24"."revenue" AS "revenue"
FROM (
  SELECT
    "t23"."n_name" AS "n_name",
    SUM("t23"."l_extendedprice" * (
      1 - "t23"."l_discount"
    )) AS "revenue"
  FROM (
    SELECT
      "t22"."c_custkey" AS "c_custkey",
      "t22"."c_name" AS "c_name",
      "t22"."c_address" AS "c_address",
      "t22"."c_nationkey" AS "c_nationkey",
      "t22"."c_phone" AS "c_phone",
      "t22"."c_acctbal" AS "c_acctbal",
      "t22"."c_mktsegment" AS "c_mktsegment",
      "t22"."c_comment" AS "c_comment",
      "t22"."o_orderkey" AS "o_orderkey",
      "t22"."o_custkey" AS "o_custkey",
      "t22"."o_orderstatus" AS "o_orderstatus",
      "t22"."o_totalprice" AS "o_totalprice",
      "t22"."o_orderdate" AS "o_orderdate",
      "t22"."o_orderpriority" AS "o_orderpriority",
      "t22"."o_clerk" AS "o_clerk",
      "t22"."o_shippriority" AS "o_shippriority",
      "t22"."o_comment" AS "o_comment",
      "t22"."l_orderkey" AS "l_orderkey",
      "t22"."l_partkey" AS "l_partkey",
      "t22"."l_suppkey" AS "l_suppkey",
      "t22"."l_linenumber" AS "l_linenumber",
      "t22"."l_quantity" AS "l_quantity",
      "t22"."l_extendedprice" AS "l_extendedprice",
      "t22"."l_discount" AS "l_discount",
      "t22"."l_tax" AS "l_tax",
      "t22"."l_returnflag" AS "l_returnflag",
      "t22"."l_linestatus" AS "l_linestatus",
      "t22"."l_shipdate" AS "l_shipdate",
      "t22"."l_commitdate" AS "l_commitdate",
      "t22"."l_receiptdate" AS "l_receiptdate",
      "t22"."l_shipinstruct" AS "l_shipinstruct",
      "t22"."l_shipmode" AS "l_shipmode",
      "t22"."l_comment" AS "l_comment",
      "t22"."s_suppkey" AS "s_suppkey",
      "t22"."s_name" AS "s_name",
      "t22"."s_address" AS "s_address",
      "t22"."s_nationkey" AS "s_nationkey",
      "t22"."s_phone" AS "s_phone",
      "t22"."s_acctbal" AS "s_acctbal",
      "t22"."s_comment" AS "s_comment",
      "t22"."n_nationkey" AS "n_nationkey",
      "t22"."n_name" AS "n_name",
      "t22"."n_regionkey" AS "n_regionkey",
      "t22"."n_comment" AS "n_comment",
      "t22"."r_regionkey" AS "r_regionkey",
      "t22"."r_name" AS "r_name",
      "t22"."r_comment" AS "r_comment"
    FROM (
      SELECT
        "t6"."c_custkey" AS "c_custkey",
        "t6"."c_name" AS "c_name",
        "t6"."c_address" AS "c_address",
        "t6"."c_nationkey" AS "c_nationkey",
        "t6"."c_phone" AS "c_phone",
        "t6"."c_acctbal" AS "c_acctbal",
        "t6"."c_mktsegment" AS "c_mktsegment",
        "t6"."c_comment" AS "c_comment",
        "t12"."o_orderkey" AS "o_orderkey",
        "t12"."o_custkey" AS "o_custkey",
        "t12"."o_orderstatus" AS "o_orderstatus",
        "t12"."o_totalprice" AS "o_totalprice",
        "t12"."o_orderdate" AS "o_orderdate",
        "t12"."o_orderpriority" AS "o_orderpriority",
        "t12"."o_clerk" AS "o_clerk",
        "t12"."o_shippriority" AS "o_shippriority",
        "t12"."o_comment" AS "o_comment",
        "t13"."l_orderkey" AS "l_orderkey",
        "t13"."l_partkey" AS "l_partkey",
        "t13"."l_suppkey" AS "l_suppkey",
        "t13"."l_linenumber" AS "l_linenumber",
        "t13"."l_quantity" AS "l_quantity",
        "t13"."l_extendedprice" AS "l_extendedprice",
        "t13"."l_discount" AS "l_discount",
        "t13"."l_tax" AS "l_tax",
        "t13"."l_returnflag" AS "l_returnflag",
        "t13"."l_linestatus" AS "l_linestatus",
        "t13"."l_shipdate" AS "l_shipdate",
        "t13"."l_commitdate" AS "l_commitdate",
        "t13"."l_receiptdate" AS "l_receiptdate",
        "t13"."l_shipinstruct" AS "l_shipinstruct",
        "t13"."l_shipmode" AS "l_shipmode",
        "t13"."l_comment" AS "l_comment",
        "t14"."s_suppkey" AS "s_suppkey",
        "t14"."s_name" AS "s_name",
        "t14"."s_address" AS "s_address",
        "t14"."s_nationkey" AS "s_nationkey",
        "t14"."s_phone" AS "s_phone",
        "t14"."s_acctbal" AS "s_acctbal",
        "t14"."s_comment" AS "s_comment",
        "t15"."n_nationkey" AS "n_nationkey",
        "t15"."n_name" AS "n_name",
        "t15"."n_regionkey" AS "n_regionkey",
        "t15"."n_comment" AS "n_comment",
        "t16"."r_regionkey" AS "r_regionkey",
        "t16"."r_name" AS "r_name",
        "t16"."r_comment" AS "r_comment"
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
      ) AS "t6"
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
      ) AS "t12"
        ON "t6"."c_custkey" = "t12"."o_custkey"
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
      ) AS "t13"
        ON "t13"."l_orderkey" = "t12"."o_orderkey"
      INNER JOIN (
        SELECT
          "t3"."S_SUPPKEY" AS "s_suppkey",
          "t3"."S_NAME" AS "s_name",
          "t3"."S_ADDRESS" AS "s_address",
          "t3"."S_NATIONKEY" AS "s_nationkey",
          "t3"."S_PHONE" AS "s_phone",
          "t3"."S_ACCTBAL" AS "s_acctbal",
          "t3"."S_COMMENT" AS "s_comment"
        FROM "SUPPLIER" AS "t3"
      ) AS "t14"
        ON "t13"."l_suppkey" = "t14"."s_suppkey"
      INNER JOIN (
        SELECT
          "t4"."N_NATIONKEY" AS "n_nationkey",
          "t4"."N_NAME" AS "n_name",
          "t4"."N_REGIONKEY" AS "n_regionkey",
          "t4"."N_COMMENT" AS "n_comment"
        FROM "NATION" AS "t4"
      ) AS "t15"
        ON "t6"."c_nationkey" = "t14"."s_nationkey"
        AND "t14"."s_nationkey" = "t15"."n_nationkey"
      INNER JOIN (
        SELECT
          "t5"."R_REGIONKEY" AS "r_regionkey",
          "t5"."R_NAME" AS "r_name",
          "t5"."R_COMMENT" AS "r_comment"
        FROM "REGION" AS "t5"
      ) AS "t16"
        ON "t15"."n_regionkey" = "t16"."r_regionkey"
    ) AS "t22"
    WHERE
      "t22"."r_name" = 'ASIA'
      AND "t22"."o_orderdate" >= DATEFROMPARTS(1994, 1, 1)
      AND "t22"."o_orderdate" < DATEFROMPARTS(1995, 1, 1)
  ) AS "t23"
  GROUP BY
    1
) AS "t24"
ORDER BY
  "t24"."revenue" DESC NULLS LAST