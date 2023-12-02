SELECT
  *
FROM (
  SELECT
    "t24"."o_year" AS "o_year",
    SUM("t24"."nation_volume") / SUM("t24"."volume") AS "mkt_share"
  FROM (
    SELECT
      "t23"."o_year" AS "o_year",
      "t23"."volume" AS "volume",
      "t23"."nation" AS "nation",
      "t23"."r_name" AS "r_name",
      "t23"."o_orderdate" AS "o_orderdate",
      "t23"."p_type" AS "p_type",
      CASE WHEN "t23"."nation" = 'BRAZIL' THEN "t23"."volume" ELSE 0 END AS "nation_volume"
    FROM (
      SELECT
        *
      FROM (
        SELECT
          DATE_PART('year', "t10"."o_orderdate") AS "o_year",
          "t8"."l_extendedprice" * (
            1 - "t8"."l_discount"
          ) AS "volume",
          "t14"."n_name" AS "nation",
          "t13"."r_name" AS "r_name",
          "t10"."o_orderdate" AS "o_orderdate",
          "t7"."p_type" AS "p_type"
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
          FROM "PART" AS "t0"
        ) AS "t7"
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
          FROM "LINEITEM" AS "t1"
        ) AS "t8"
          ON "t7"."p_partkey" = "t8"."l_partkey"
        INNER JOIN (
          SELECT
            "t2"."S_SUPPKEY" AS "s_suppkey",
            "t2"."S_NAME" AS "s_name",
            "t2"."S_ADDRESS" AS "s_address",
            "t2"."S_NATIONKEY" AS "s_nationkey",
            "t2"."S_PHONE" AS "s_phone",
            "t2"."S_ACCTBAL" AS "s_acctbal",
            "t2"."S_COMMENT" AS "s_comment"
          FROM "SUPPLIER" AS "t2"
        ) AS "t9"
          ON "t9"."s_suppkey" = "t8"."l_suppkey"
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
          FROM "ORDERS" AS "t3"
        ) AS "t10"
          ON "t8"."l_orderkey" = "t10"."o_orderkey"
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
          FROM "CUSTOMER" AS "t4"
        ) AS "t11"
          ON "t10"."o_custkey" = "t11"."c_custkey"
        INNER JOIN (
          SELECT
            "t5"."N_NATIONKEY" AS "n_nationkey",
            "t5"."N_NAME" AS "n_name",
            "t5"."N_REGIONKEY" AS "n_regionkey",
            "t5"."N_COMMENT" AS "n_comment"
          FROM "NATION" AS "t5"
        ) AS "t12"
          ON "t11"."c_nationkey" = "t12"."n_nationkey"
        INNER JOIN (
          SELECT
            "t6"."R_REGIONKEY" AS "r_regionkey",
            "t6"."R_NAME" AS "r_name",
            "t6"."R_COMMENT" AS "r_comment"
          FROM "REGION" AS "t6"
        ) AS "t13"
          ON "t12"."n_regionkey" = "t13"."r_regionkey"
        INNER JOIN (
          SELECT
            "t5"."N_NATIONKEY" AS "n_nationkey",
            "t5"."N_NAME" AS "n_name",
            "t5"."N_REGIONKEY" AS "n_regionkey",
            "t5"."N_COMMENT" AS "n_comment"
          FROM "NATION" AS "t5"
        ) AS "t14"
          ON "t9"."s_nationkey" = "t14"."n_nationkey"
      ) AS "t22"
      WHERE
        (
          "t22"."r_name" = 'AMERICA'
        )
        AND "t22"."o_orderdate" BETWEEN DATEFROMPARTS(1995, 1, 1) AND DATEFROMPARTS(1996, 12, 31)
        AND (
          "t22"."p_type" = 'ECONOMY ANODIZED STEEL'
        )
    ) AS "t23"
  ) AS "t24"
  GROUP BY
    1
) AS "t25"
ORDER BY
  "t25"."o_year" ASC