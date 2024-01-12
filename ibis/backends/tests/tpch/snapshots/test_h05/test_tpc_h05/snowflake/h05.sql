SELECT
  "t25"."n_name",
  "t25"."revenue"
FROM (
  SELECT
    "t24"."n_name",
    SUM("t24"."l_extendedprice" * (
      1 - "t24"."l_discount"
    )) AS "revenue"
  FROM (
    SELECT
      "t23"."c_custkey",
      "t23"."c_name",
      "t23"."c_address",
      "t23"."c_nationkey",
      "t23"."c_phone",
      "t23"."c_acctbal",
      "t23"."c_mktsegment",
      "t23"."c_comment",
      "t23"."o_orderkey",
      "t23"."o_custkey",
      "t23"."o_orderstatus",
      "t23"."o_totalprice",
      "t23"."o_orderdate",
      "t23"."o_orderpriority",
      "t23"."o_clerk",
      "t23"."o_shippriority",
      "t23"."o_comment",
      "t23"."l_orderkey",
      "t23"."l_partkey",
      "t23"."l_suppkey",
      "t23"."l_linenumber",
      "t23"."l_quantity",
      "t23"."l_extendedprice",
      "t23"."l_discount",
      "t23"."l_tax",
      "t23"."l_returnflag",
      "t23"."l_linestatus",
      "t23"."l_shipdate",
      "t23"."l_commitdate",
      "t23"."l_receiptdate",
      "t23"."l_shipinstruct",
      "t23"."l_shipmode",
      "t23"."l_comment",
      "t23"."s_suppkey",
      "t23"."s_name",
      "t23"."s_address",
      "t23"."s_nationkey",
      "t23"."s_phone",
      "t23"."s_acctbal",
      "t23"."s_comment",
      "t23"."n_nationkey",
      "t23"."n_name",
      "t23"."n_regionkey",
      "t23"."n_comment",
      "t23"."r_regionkey",
      "t23"."r_name",
      "t23"."r_comment"
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
        "t13"."o_orderkey",
        "t13"."o_custkey",
        "t13"."o_orderstatus",
        "t13"."o_totalprice",
        "t13"."o_orderdate",
        "t13"."o_orderpriority",
        "t13"."o_clerk",
        "t13"."o_shippriority",
        "t13"."o_comment",
        "t14"."l_orderkey",
        "t14"."l_partkey",
        "t14"."l_suppkey",
        "t14"."l_linenumber",
        "t14"."l_quantity",
        "t14"."l_extendedprice",
        "t14"."l_discount",
        "t14"."l_tax",
        "t14"."l_returnflag",
        "t14"."l_linestatus",
        "t14"."l_shipdate",
        "t14"."l_commitdate",
        "t14"."l_receiptdate",
        "t14"."l_shipinstruct",
        "t14"."l_shipmode",
        "t14"."l_comment",
        "t15"."s_suppkey",
        "t15"."s_name",
        "t15"."s_address",
        "t15"."s_nationkey",
        "t15"."s_phone",
        "t15"."s_acctbal",
        "t15"."s_comment",
        "t16"."n_nationkey",
        "t16"."n_name",
        "t16"."n_regionkey",
        "t16"."n_comment",
        "t17"."r_regionkey",
        "t17"."r_name",
        "t17"."r_comment"
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
      ) AS "t12"
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
      ) AS "t13"
        ON "t12"."c_custkey" = "t13"."o_custkey"
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
      ) AS "t14"
        ON "t14"."l_orderkey" = "t13"."o_orderkey"
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
      ) AS "t15"
        ON "t14"."l_suppkey" = "t15"."s_suppkey"
      INNER JOIN (
        SELECT
          "t4"."N_NATIONKEY" AS "n_nationkey",
          "t4"."N_NAME" AS "n_name",
          "t4"."N_REGIONKEY" AS "n_regionkey",
          "t4"."N_COMMENT" AS "n_comment"
        FROM "NATION" AS "t4"
      ) AS "t16"
        ON "t12"."c_nationkey" = "t15"."s_nationkey"
        AND "t15"."s_nationkey" = "t16"."n_nationkey"
      INNER JOIN (
        SELECT
          "t5"."R_REGIONKEY" AS "r_regionkey",
          "t5"."R_NAME" AS "r_name",
          "t5"."R_COMMENT" AS "r_comment"
        FROM "REGION" AS "t5"
      ) AS "t17"
        ON "t16"."n_regionkey" = "t17"."r_regionkey"
    ) AS "t23"
    WHERE
      "t23"."r_name" = 'ASIA'
      AND "t23"."o_orderdate" >= DATE_FROM_PARTS(1994, 1, 1)
      AND "t23"."o_orderdate" < DATE_FROM_PARTS(1995, 1, 1)
  ) AS "t24"
  GROUP BY
    1
) AS "t25"
ORDER BY
  "t25"."revenue" DESC NULLS LAST