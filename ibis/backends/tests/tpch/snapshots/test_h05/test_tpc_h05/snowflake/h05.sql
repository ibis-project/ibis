SELECT
  "t20"."n_name",
  "t20"."revenue"
FROM (
  SELECT
    "t19"."n_name",
    SUM("t19"."l_extendedprice" * (
      1 - "t19"."l_discount"
    )) AS "revenue"
  FROM (
    SELECT
      "t18"."c_custkey",
      "t18"."c_name",
      "t18"."c_address",
      "t18"."c_nationkey",
      "t18"."c_phone",
      "t18"."c_acctbal",
      "t18"."c_mktsegment",
      "t18"."c_comment",
      "t18"."o_orderkey",
      "t18"."o_custkey",
      "t18"."o_orderstatus",
      "t18"."o_totalprice",
      "t18"."o_orderdate",
      "t18"."o_orderpriority",
      "t18"."o_clerk",
      "t18"."o_shippriority",
      "t18"."o_comment",
      "t18"."l_orderkey",
      "t18"."l_partkey",
      "t18"."l_suppkey",
      "t18"."l_linenumber",
      "t18"."l_quantity",
      "t18"."l_extendedprice",
      "t18"."l_discount",
      "t18"."l_tax",
      "t18"."l_returnflag",
      "t18"."l_linestatus",
      "t18"."l_shipdate",
      "t18"."l_commitdate",
      "t18"."l_receiptdate",
      "t18"."l_shipinstruct",
      "t18"."l_shipmode",
      "t18"."l_comment",
      "t18"."s_suppkey",
      "t18"."s_name",
      "t18"."s_address",
      "t18"."s_nationkey",
      "t18"."s_phone",
      "t18"."s_acctbal",
      "t18"."s_comment",
      "t18"."n_nationkey",
      "t18"."n_name",
      "t18"."n_regionkey",
      "t18"."n_comment",
      "t18"."r_regionkey",
      "t18"."r_name",
      "t18"."r_comment"
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
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."CUSTOMER" AS "t0"
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
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS "t1"
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
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS "t2"
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
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS "t3"
      ) AS "t15"
        ON "t14"."l_suppkey" = "t15"."s_suppkey"
      INNER JOIN (
        SELECT
          "t4"."N_NATIONKEY" AS "n_nationkey",
          "t4"."N_NAME" AS "n_name",
          "t4"."N_REGIONKEY" AS "n_regionkey",
          "t4"."N_COMMENT" AS "n_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS "t4"
      ) AS "t16"
        ON "t12"."c_nationkey" = "t15"."s_nationkey"
        AND "t15"."s_nationkey" = "t16"."n_nationkey"
      INNER JOIN (
        SELECT
          "t5"."R_REGIONKEY" AS "r_regionkey",
          "t5"."R_NAME" AS "r_name",
          "t5"."R_COMMENT" AS "r_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."REGION" AS "t5"
      ) AS "t17"
        ON "t16"."n_regionkey" = "t17"."r_regionkey"
    ) AS "t18"
    WHERE
      "t18"."r_name" = 'ASIA'
      AND "t18"."o_orderdate" >= DATE_FROM_PARTS(1994, 1, 1)
      AND "t18"."o_orderdate" < DATE_FROM_PARTS(1995, 1, 1)
  ) AS "t19"
  GROUP BY
    1
) AS "t20"
ORDER BY
  "t20"."revenue" DESC NULLS LAST