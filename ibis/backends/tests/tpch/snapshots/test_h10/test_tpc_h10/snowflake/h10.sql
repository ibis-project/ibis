SELECT
  "t14"."c_custkey",
  "t14"."c_name",
  "t14"."revenue",
  "t14"."c_acctbal",
  "t14"."n_name",
  "t14"."c_address",
  "t14"."c_phone",
  "t14"."c_comment"
FROM (
  SELECT
    "t13"."c_custkey",
    "t13"."c_name",
    "t13"."c_acctbal",
    "t13"."n_name",
    "t13"."c_address",
    "t13"."c_phone",
    "t13"."c_comment",
    SUM("t13"."l_extendedprice" * (
      1 - "t13"."l_discount"
    )) AS "revenue"
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
      "t12"."l_comment",
      "t12"."n_nationkey",
      "t12"."n_name",
      "t12"."n_regionkey",
      "t12"."n_comment"
    FROM (
      SELECT
        "t8"."c_custkey",
        "t8"."c_name",
        "t8"."c_address",
        "t8"."c_nationkey",
        "t8"."c_phone",
        "t8"."c_acctbal",
        "t8"."c_mktsegment",
        "t8"."c_comment",
        "t9"."o_orderkey",
        "t9"."o_custkey",
        "t9"."o_orderstatus",
        "t9"."o_totalprice",
        "t9"."o_orderdate",
        "t9"."o_orderpriority",
        "t9"."o_clerk",
        "t9"."o_shippriority",
        "t9"."o_comment",
        "t10"."l_orderkey",
        "t10"."l_partkey",
        "t10"."l_suppkey",
        "t10"."l_linenumber",
        "t10"."l_quantity",
        "t10"."l_extendedprice",
        "t10"."l_discount",
        "t10"."l_tax",
        "t10"."l_returnflag",
        "t10"."l_linestatus",
        "t10"."l_shipdate",
        "t10"."l_commitdate",
        "t10"."l_receiptdate",
        "t10"."l_shipinstruct",
        "t10"."l_shipmode",
        "t10"."l_comment",
        "t11"."n_nationkey",
        "t11"."n_name",
        "t11"."n_regionkey",
        "t11"."n_comment"
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
      ) AS "t8"
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
      ) AS "t9"
        ON "t8"."c_custkey" = "t9"."o_custkey"
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
      ) AS "t10"
        ON "t10"."l_orderkey" = "t9"."o_orderkey"
      INNER JOIN (
        SELECT
          "t3"."N_NATIONKEY" AS "n_nationkey",
          "t3"."N_NAME" AS "n_name",
          "t3"."N_REGIONKEY" AS "n_regionkey",
          "t3"."N_COMMENT" AS "n_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS "t3"
      ) AS "t11"
        ON "t8"."c_nationkey" = "t11"."n_nationkey"
    ) AS "t12"
    WHERE
      "t12"."o_orderdate" >= DATE_FROM_PARTS(1993, 10, 1)
      AND "t12"."o_orderdate" < DATE_FROM_PARTS(1994, 1, 1)
      AND "t12"."l_returnflag" = 'R'
  ) AS "t13"
  GROUP BY
    1,
    2,
    3,
    4,
    5,
    6,
    7
) AS "t14"
ORDER BY
  "t14"."revenue" DESC NULLS LAST
LIMIT 20