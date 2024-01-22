WITH "t13" AS (
  SELECT
    "t6"."N_NATIONKEY" AS "n_nationkey",
    "t6"."N_NAME" AS "n_name",
    "t6"."N_REGIONKEY" AS "n_regionkey",
    "t6"."N_COMMENT" AS "n_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS "t6"
)
SELECT
  "t26"."o_year",
  "t26"."mkt_share"
FROM (
  SELECT
    "t25"."o_year",
    SUM("t25"."nation_volume") / SUM("t25"."volume") AS "mkt_share"
  FROM (
    SELECT
      "t24"."o_year",
      "t24"."volume",
      "t24"."nation",
      "t24"."r_name",
      "t24"."o_orderdate",
      "t24"."p_type",
      CASE WHEN "t24"."nation" = 'BRAZIL' THEN "t24"."volume" ELSE 0 END AS "nation_volume"
    FROM (
      SELECT
        DATE_PART(year, "t17"."o_orderdate") AS "o_year",
        "t15"."l_extendedprice" * (
          1 - "t15"."l_discount"
        ) AS "volume",
        "t23"."n_name" AS "nation",
        "t19"."r_name",
        "t17"."o_orderdate",
        "t14"."p_type"
      FROM (
        SELECT
          "t0"."P_PARTKEY" AS "p_partkey",
          "t0"."P_NAME" AS "p_name",
          "t0"."P_MFGR" AS "p_mfgr",
          "t0"."P_BRAND" AS "p_brand",
          "t0"."P_TYPE" AS "p_type",
          "t0"."P_SIZE" AS "p_size",
          "t0"."P_CONTAINER" AS "p_container",
          "t0"."P_RETAILPRICE" AS "p_retailprice",
          "t0"."P_COMMENT" AS "p_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PART" AS "t0"
      ) AS "t14"
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
      ) AS "t15"
        ON "t14"."p_partkey" = "t15"."l_partkey"
      INNER JOIN (
        SELECT
          "t2"."S_SUPPKEY" AS "s_suppkey",
          "t2"."S_NAME" AS "s_name",
          "t2"."S_ADDRESS" AS "s_address",
          "t2"."S_NATIONKEY" AS "s_nationkey",
          "t2"."S_PHONE" AS "s_phone",
          "t2"."S_ACCTBAL" AS "s_acctbal",
          "t2"."S_COMMENT" AS "s_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS "t2"
      ) AS "t16"
        ON "t16"."s_suppkey" = "t15"."l_suppkey"
      INNER JOIN (
        SELECT
          "t3"."O_ORDERKEY" AS "o_orderkey",
          "t3"."O_CUSTKEY" AS "o_custkey",
          "t3"."O_ORDERSTATUS" AS "o_orderstatus",
          "t3"."O_TOTALPRICE" AS "o_totalprice",
          "t3"."O_ORDERDATE" AS "o_orderdate",
          "t3"."O_ORDERPRIORITY" AS "o_orderpriority",
          "t3"."O_CLERK" AS "o_clerk",
          "t3"."O_SHIPPRIORITY" AS "o_shippriority",
          "t3"."O_COMMENT" AS "o_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS "t3"
      ) AS "t17"
        ON "t15"."l_orderkey" = "t17"."o_orderkey"
      INNER JOIN (
        SELECT
          "t4"."C_CUSTKEY" AS "c_custkey",
          "t4"."C_NAME" AS "c_name",
          "t4"."C_ADDRESS" AS "c_address",
          "t4"."C_NATIONKEY" AS "c_nationkey",
          "t4"."C_PHONE" AS "c_phone",
          "t4"."C_ACCTBAL" AS "c_acctbal",
          "t4"."C_MKTSEGMENT" AS "c_mktsegment",
          "t4"."C_COMMENT" AS "c_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."CUSTOMER" AS "t4"
      ) AS "t18"
        ON "t17"."o_custkey" = "t18"."c_custkey"
      INNER JOIN "t13" AS "t21"
        ON "t18"."c_nationkey" = "t21"."n_nationkey"
      INNER JOIN (
        SELECT
          "t5"."R_REGIONKEY" AS "r_regionkey",
          "t5"."R_NAME" AS "r_name",
          "t5"."R_COMMENT" AS "r_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."REGION" AS "t5"
      ) AS "t19"
        ON "t21"."n_regionkey" = "t19"."r_regionkey"
      INNER JOIN "t13" AS "t23"
        ON "t16"."s_nationkey" = "t23"."n_nationkey"
    ) AS "t24"
    WHERE
      "t24"."r_name" = 'AMERICA'
      AND "t24"."o_orderdate" BETWEEN DATE_FROM_PARTS(1995, 1, 1) AND DATE_FROM_PARTS(1996, 12, 31)
      AND "t24"."p_type" = 'ECONOMY ANODIZED STEEL'
  ) AS "t25"
  GROUP BY
    1
) AS "t26"
ORDER BY
  "t26"."o_year" ASC