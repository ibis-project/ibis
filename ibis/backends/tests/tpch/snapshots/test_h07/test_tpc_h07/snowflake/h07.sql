WITH "t9" AS (
  SELECT
    "t4"."N_NATIONKEY" AS "n_nationkey",
    "t4"."N_NAME" AS "n_name",
    "t4"."N_REGIONKEY" AS "n_regionkey",
    "t4"."N_COMMENT" AS "n_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS "t4"
)
SELECT
  "t20"."supp_nation",
  "t20"."cust_nation",
  "t20"."l_year",
  "t20"."revenue"
FROM (
  SELECT
    "t19"."supp_nation",
    "t19"."cust_nation",
    "t19"."l_year",
    SUM("t19"."volume") AS "revenue"
  FROM (
    SELECT
      "t18"."supp_nation",
      "t18"."cust_nation",
      "t18"."l_shipdate",
      "t18"."l_extendedprice",
      "t18"."l_discount",
      "t18"."l_year",
      "t18"."volume"
    FROM (
      SELECT
        "t15"."n_name" AS "supp_nation",
        "t17"."n_name" AS "cust_nation",
        "t11"."l_shipdate",
        "t11"."l_extendedprice",
        "t11"."l_discount",
        DATE_PART(year, "t11"."l_shipdate") AS "l_year",
        "t11"."l_extendedprice" * (
          1 - "t11"."l_discount"
        ) AS "volume"
      FROM (
        SELECT
          "t0"."S_SUPPKEY" AS "s_suppkey",
          "t0"."S_NAME" AS "s_name",
          "t0"."S_ADDRESS" AS "s_address",
          "t0"."S_NATIONKEY" AS "s_nationkey",
          "t0"."S_PHONE" AS "s_phone",
          "t0"."S_ACCTBAL" AS "s_acctbal",
          "t0"."S_COMMENT" AS "s_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS "t0"
      ) AS "t10"
      INNER JOIN (
        SELECT
          "t1"."L_ORDERKEY" AS "l_orderkey",
          "t1"."L_PARTKEY" AS "l_partkey",
          "t1"."L_SUPPKEY" AS "l_suppkey",
          "t1"."L_LINENUMBER" AS "l_linenumber",
          "t1"."L_QUANTITY" AS "l_quantity",
          "t1"."L_EXTENDEDPRICE" AS "l_extendedprice",
          "t1"."L_DISCOUNT" AS "l_discount",
          "t1"."L_TAX" AS "l_tax",
          "t1"."L_RETURNFLAG" AS "l_returnflag",
          "t1"."L_LINESTATUS" AS "l_linestatus",
          "t1"."L_SHIPDATE" AS "l_shipdate",
          "t1"."L_COMMITDATE" AS "l_commitdate",
          "t1"."L_RECEIPTDATE" AS "l_receiptdate",
          "t1"."L_SHIPINSTRUCT" AS "l_shipinstruct",
          "t1"."L_SHIPMODE" AS "l_shipmode",
          "t1"."L_COMMENT" AS "l_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS "t1"
      ) AS "t11"
        ON "t10"."s_suppkey" = "t11"."l_suppkey"
      INNER JOIN (
        SELECT
          "t2"."O_ORDERKEY" AS "o_orderkey",
          "t2"."O_CUSTKEY" AS "o_custkey",
          "t2"."O_ORDERSTATUS" AS "o_orderstatus",
          "t2"."O_TOTALPRICE" AS "o_totalprice",
          "t2"."O_ORDERDATE" AS "o_orderdate",
          "t2"."O_ORDERPRIORITY" AS "o_orderpriority",
          "t2"."O_CLERK" AS "o_clerk",
          "t2"."O_SHIPPRIORITY" AS "o_shippriority",
          "t2"."O_COMMENT" AS "o_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS "t2"
      ) AS "t12"
        ON "t12"."o_orderkey" = "t11"."l_orderkey"
      INNER JOIN (
        SELECT
          "t3"."C_CUSTKEY" AS "c_custkey",
          "t3"."C_NAME" AS "c_name",
          "t3"."C_ADDRESS" AS "c_address",
          "t3"."C_NATIONKEY" AS "c_nationkey",
          "t3"."C_PHONE" AS "c_phone",
          "t3"."C_ACCTBAL" AS "c_acctbal",
          "t3"."C_MKTSEGMENT" AS "c_mktsegment",
          "t3"."C_COMMENT" AS "c_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."CUSTOMER" AS "t3"
      ) AS "t13"
        ON "t13"."c_custkey" = "t12"."o_custkey"
      INNER JOIN "t9" AS "t15"
        ON "t10"."s_nationkey" = "t15"."n_nationkey"
      INNER JOIN "t9" AS "t17"
        ON "t13"."c_nationkey" = "t17"."n_nationkey"
    ) AS "t18"
    WHERE
      (
        (
          (
            "t18"."cust_nation" = 'FRANCE'
          ) AND (
            "t18"."supp_nation" = 'GERMANY'
          )
        )
        OR (
          (
            "t18"."cust_nation" = 'GERMANY'
          ) AND (
            "t18"."supp_nation" = 'FRANCE'
          )
        )
      )
      AND "t18"."l_shipdate" BETWEEN DATE_FROM_PARTS(1995, 1, 1) AND DATE_FROM_PARTS(1996, 12, 31)
  ) AS "t19"
  GROUP BY
    1,
    2,
    3
) AS "t20"
ORDER BY
  "t20"."supp_nation" ASC,
  "t20"."cust_nation" ASC,
  "t20"."l_year" ASC