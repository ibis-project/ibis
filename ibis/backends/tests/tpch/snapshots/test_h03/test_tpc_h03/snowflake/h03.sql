SELECT
  "t11"."l_orderkey",
  "t11"."revenue",
  "t11"."o_orderdate",
  "t11"."o_shippriority"
FROM (
  SELECT
    "t10"."l_orderkey",
    "t10"."o_orderdate",
    "t10"."o_shippriority",
    SUM("t10"."l_extendedprice" * (
      1 - "t10"."l_discount"
    )) AS "revenue"
  FROM (
    SELECT
      "t9"."c_custkey",
      "t9"."c_name",
      "t9"."c_address",
      "t9"."c_nationkey",
      "t9"."c_phone",
      "t9"."c_acctbal",
      "t9"."c_mktsegment",
      "t9"."c_comment",
      "t9"."o_orderkey",
      "t9"."o_custkey",
      "t9"."o_orderstatus",
      "t9"."o_totalprice",
      "t9"."o_orderdate",
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
      "t9"."l_comment"
    FROM (
      SELECT
        "t6"."c_custkey",
        "t6"."c_name",
        "t6"."c_address",
        "t6"."c_nationkey",
        "t6"."c_phone",
        "t6"."c_acctbal",
        "t6"."c_mktsegment",
        "t6"."c_comment",
        "t7"."o_orderkey",
        "t7"."o_custkey",
        "t7"."o_orderstatus",
        "t7"."o_totalprice",
        "t7"."o_orderdate",
        "t7"."o_orderpriority",
        "t7"."o_clerk",
        "t7"."o_shippriority",
        "t7"."o_comment",
        "t8"."l_orderkey",
        "t8"."l_partkey",
        "t8"."l_suppkey",
        "t8"."l_linenumber",
        "t8"."l_quantity",
        "t8"."l_extendedprice",
        "t8"."l_discount",
        "t8"."l_tax",
        "t8"."l_returnflag",
        "t8"."l_linestatus",
        "t8"."l_shipdate",
        "t8"."l_commitdate",
        "t8"."l_receiptdate",
        "t8"."l_shipinstruct",
        "t8"."l_shipmode",
        "t8"."l_comment"
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
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS "t1"
      ) AS "t7"
        ON "t6"."c_custkey" = "t7"."o_custkey"
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
      ) AS "t8"
        ON "t8"."l_orderkey" = "t7"."o_orderkey"
    ) AS "t9"
    WHERE
      "t9"."c_mktsegment" = 'BUILDING'
      AND "t9"."o_orderdate" < DATE_FROM_PARTS(1995, 3, 15)
      AND "t9"."l_shipdate" > DATE_FROM_PARTS(1995, 3, 15)
  ) AS "t10"
  GROUP BY
    1,
    2,
    3
) AS "t11"
ORDER BY
  "t11"."revenue" DESC NULLS LAST,
  "t11"."o_orderdate" ASC
LIMIT 10