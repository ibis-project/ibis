SELECT
  *
FROM (
  SELECT
    "t18"."n_name" AS "n_name",
    SUM("t18"."l_extendedprice" * (
      1 - "t18"."l_discount"
    )) AS "revenue"
  FROM (
    SELECT
      *
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
        "t7"."o_orderkey" AS "o_orderkey",
        "t7"."o_custkey" AS "o_custkey",
        "t7"."o_orderstatus" AS "o_orderstatus",
        "t7"."o_totalprice" AS "o_totalprice",
        "t7"."o_orderdate" AS "o_orderdate",
        "t7"."o_orderpriority" AS "o_orderpriority",
        "t7"."o_clerk" AS "o_clerk",
        "t7"."o_shippriority" AS "o_shippriority",
        "t7"."o_comment" AS "o_comment",
        "t8"."l_orderkey" AS "l_orderkey",
        "t8"."l_partkey" AS "l_partkey",
        "t8"."l_suppkey" AS "l_suppkey",
        "t8"."l_linenumber" AS "l_linenumber",
        "t8"."l_quantity" AS "l_quantity",
        "t8"."l_extendedprice" AS "l_extendedprice",
        "t8"."l_discount" AS "l_discount",
        "t8"."l_tax" AS "l_tax",
        "t8"."l_returnflag" AS "l_returnflag",
        "t8"."l_linestatus" AS "l_linestatus",
        "t8"."l_shipdate" AS "l_shipdate",
        "t8"."l_commitdate" AS "l_commitdate",
        "t8"."l_receiptdate" AS "l_receiptdate",
        "t8"."l_shipinstruct" AS "l_shipinstruct",
        "t8"."l_shipmode" AS "l_shipmode",
        "t8"."l_comment" AS "l_comment",
        "t9"."s_suppkey" AS "s_suppkey",
        "t9"."s_name" AS "s_name",
        "t9"."s_address" AS "s_address",
        "t9"."s_nationkey" AS "s_nationkey",
        "t9"."s_phone" AS "s_phone",
        "t9"."s_acctbal" AS "s_acctbal",
        "t9"."s_comment" AS "s_comment",
        "t10"."n_nationkey" AS "n_nationkey",
        "t10"."n_name" AS "n_name",
        "t10"."n_regionkey" AS "n_regionkey",
        "t10"."n_comment" AS "n_comment",
        "t11"."r_regionkey" AS "r_regionkey",
        "t11"."r_name" AS "r_name",
        "t11"."r_comment" AS "r_comment"
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
        FROM "LINEITEM" AS "t2"
      ) AS "t8"
        ON "t8"."l_orderkey" = "t7"."o_orderkey"
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
      ) AS "t9"
        ON "t8"."l_suppkey" = "t9"."s_suppkey"
      INNER JOIN (
        SELECT
          "t4"."N_NATIONKEY" AS "n_nationkey",
          "t4"."N_NAME" AS "n_name",
          "t4"."N_REGIONKEY" AS "n_regionkey",
          "t4"."N_COMMENT" AS "n_comment"
        FROM "NATION" AS "t4"
      ) AS "t10"
        ON "t6"."c_nationkey" = "t9"."s_nationkey" AND "t9"."s_nationkey" = "t10"."n_nationkey"
      INNER JOIN (
        SELECT
          "t5"."R_REGIONKEY" AS "r_regionkey",
          "t5"."R_NAME" AS "r_name",
          "t5"."R_COMMENT" AS "r_comment"
        FROM "REGION" AS "t5"
      ) AS "t11"
        ON "t10"."n_regionkey" = "t11"."r_regionkey"
    ) AS "t17"
    WHERE
      (
        "t17"."r_name" = 'ASIA'
      )
      AND (
        "t17"."o_orderdate" >= DATEFROMPARTS(1994, 1, 1)
      )
      AND (
        "t17"."o_orderdate" < DATEFROMPARTS(1995, 1, 1)
      )
  ) AS "t18"
  GROUP BY
    1
) AS "t19"
ORDER BY
  "t19"."revenue" DESC NULLS LAST