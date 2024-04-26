SELECT
  "t20"."s_name",
  "t20"."s_address"
FROM (
  SELECT
    "t19"."s_name",
    "t19"."s_address"
  FROM (
    SELECT
      "t14"."s_suppkey",
      "t14"."s_name",
      "t14"."s_address",
      "t14"."s_nationkey",
      "t14"."s_phone",
      "t14"."s_acctbal",
      "t14"."s_comment",
      "t14"."n_nationkey",
      "t14"."n_name",
      "t14"."n_regionkey",
      "t14"."n_comment"
    FROM (
      SELECT
        "t10"."s_suppkey",
        "t10"."s_name",
        "t10"."s_address",
        "t10"."s_nationkey",
        "t10"."s_phone",
        "t10"."s_acctbal",
        "t10"."s_comment",
        "t11"."n_nationkey",
        "t11"."n_name",
        "t11"."n_regionkey",
        "t11"."n_comment"
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
          "t1"."N_NATIONKEY" AS "n_nationkey",
          "t1"."N_NAME" AS "n_name",
          "t1"."N_REGIONKEY" AS "n_regionkey",
          "t1"."N_COMMENT" AS "n_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS "t1"
      ) AS "t11"
        ON "t10"."s_nationkey" = "t11"."n_nationkey"
    ) AS "t14"
    WHERE
      "t14"."n_name" = 'CANADA'
      AND "t14"."s_suppkey" IN (
        SELECT
          "t17"."ps_suppkey"
        FROM (
          SELECT
            "t7"."ps_partkey",
            "t7"."ps_suppkey",
            "t7"."ps_availqty",
            "t7"."ps_supplycost",
            "t7"."ps_comment"
          FROM (
            SELECT
              "t2"."PS_PARTKEY" AS "ps_partkey",
              "t2"."PS_SUPPKEY" AS "ps_suppkey",
              "t2"."PS_AVAILQTY" AS "ps_availqty",
              "t2"."PS_SUPPLYCOST" AS "ps_supplycost",
              "t2"."PS_COMMENT" AS "ps_comment"
            FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PARTSUPP" AS "t2"
          ) AS "t7"
          WHERE
            "t7"."ps_partkey" IN (
              SELECT
                "t12"."p_partkey"
              FROM (
                SELECT
                  "t8"."p_partkey",
                  "t8"."p_name",
                  "t8"."p_mfgr",
                  "t8"."p_brand",
                  "t8"."p_type",
                  "t8"."p_size",
                  "t8"."p_container",
                  "t8"."p_retailprice",
                  "t8"."p_comment"
                FROM (
                  SELECT
                    "t3"."P_PARTKEY" AS "p_partkey",
                    "t3"."P_NAME" AS "p_name",
                    "t3"."P_MFGR" AS "p_mfgr",
                    "t3"."P_BRAND" AS "p_brand",
                    "t3"."P_TYPE" AS "p_type",
                    "t3"."P_SIZE" AS "p_size",
                    "t3"."P_CONTAINER" AS "p_container",
                    "t3"."P_RETAILPRICE" AS "p_retailprice",
                    "t3"."P_COMMENT" AS "p_comment"
                  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PART" AS "t3"
                ) AS "t8"
                WHERE
                  "t8"."p_name" LIKE 'forest%'
              ) AS "t12"
            )
            AND "t7"."ps_availqty" > (
              (
                SELECT
                  SUM("t13"."l_quantity") AS "Sum(l_quantity)"
                FROM (
                  SELECT
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
                      "t4"."L_ORDERKEY" AS "l_orderkey",
                      "t4"."L_PARTKEY" AS "l_partkey",
                      "t4"."L_SUPPKEY" AS "l_suppkey",
                      "t4"."L_LINENUMBER" AS "l_linenumber",
                      "t4"."L_QUANTITY" AS "l_quantity",
                      "t4"."L_EXTENDEDPRICE" AS "l_extendedprice",
                      "t4"."L_DISCOUNT" AS "l_discount",
                      "t4"."L_TAX" AS "l_tax",
                      "t4"."L_RETURNFLAG" AS "l_returnflag",
                      "t4"."L_LINESTATUS" AS "l_linestatus",
                      "t4"."L_SHIPDATE" AS "l_shipdate",
                      "t4"."L_COMMITDATE" AS "l_commitdate",
                      "t4"."L_RECEIPTDATE" AS "l_receiptdate",
                      "t4"."L_SHIPINSTRUCT" AS "l_shipinstruct",
                      "t4"."L_SHIPMODE" AS "l_shipmode",
                      "t4"."L_COMMENT" AS "l_comment"
                    FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS "t4"
                  ) AS "t9"
                  WHERE
                    "t9"."l_partkey" = "t7"."ps_partkey"
                    AND "t9"."l_suppkey" = "t7"."ps_suppkey"
                    AND "t9"."l_shipdate" >= DATE_FROM_PARTS(1994, 1, 1)
                    AND "t9"."l_shipdate" < DATE_FROM_PARTS(1995, 1, 1)
                ) AS "t13"
              ) * 0.5
            )
        ) AS "t17"
      )
  ) AS "t19"
) AS "t20"
ORDER BY
  "t20"."s_name" ASC