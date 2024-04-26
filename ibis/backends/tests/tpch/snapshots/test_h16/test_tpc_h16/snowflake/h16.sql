SELECT
  "t12"."p_brand",
  "t12"."p_type",
  "t12"."p_size",
  "t12"."supplier_cnt"
FROM (
  SELECT
    "t11"."p_brand",
    "t11"."p_type",
    "t11"."p_size",
    COUNT(DISTINCT "t11"."ps_suppkey") AS "supplier_cnt"
  FROM (
    SELECT
      "t9"."ps_partkey",
      "t9"."ps_suppkey",
      "t9"."ps_availqty",
      "t9"."ps_supplycost",
      "t9"."ps_comment",
      "t9"."p_partkey",
      "t9"."p_name",
      "t9"."p_mfgr",
      "t9"."p_brand",
      "t9"."p_type",
      "t9"."p_size",
      "t9"."p_container",
      "t9"."p_retailprice",
      "t9"."p_comment"
    FROM (
      SELECT
        "t6"."ps_partkey",
        "t6"."ps_suppkey",
        "t6"."ps_availqty",
        "t6"."ps_supplycost",
        "t6"."ps_comment",
        "t7"."p_partkey",
        "t7"."p_name",
        "t7"."p_mfgr",
        "t7"."p_brand",
        "t7"."p_type",
        "t7"."p_size",
        "t7"."p_container",
        "t7"."p_retailprice",
        "t7"."p_comment"
      FROM (
        SELECT
          "t0"."PS_PARTKEY" AS "ps_partkey",
          "t0"."PS_SUPPKEY" AS "ps_suppkey",
          "t0"."PS_AVAILQTY" AS "ps_availqty",
          "t0"."PS_SUPPLYCOST" AS "ps_supplycost",
          "t0"."PS_COMMENT" AS "ps_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PARTSUPP" AS "t0"
      ) AS "t6"
      INNER JOIN (
        SELECT
          "t1"."P_PARTKEY" AS "p_partkey",
          "t1"."P_NAME" AS "p_name",
          "t1"."P_MFGR" AS "p_mfgr",
          "t1"."P_BRAND" AS "p_brand",
          "t1"."P_TYPE" AS "p_type",
          "t1"."P_SIZE" AS "p_size",
          "t1"."P_CONTAINER" AS "p_container",
          "t1"."P_RETAILPRICE" AS "p_retailprice",
          "t1"."P_COMMENT" AS "p_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PART" AS "t1"
      ) AS "t7"
        ON "t7"."p_partkey" = "t6"."ps_partkey"
    ) AS "t9"
    WHERE
      "t9"."p_brand" <> 'Brand#45'
      AND NOT (
        "t9"."p_type" LIKE 'MEDIUM POLISHED%'
      )
      AND "t9"."p_size" IN (49, 14, 23, 45, 19, 3, 36, 9)
      AND NOT (
        "t9"."ps_suppkey" IN (
          SELECT
            "t8"."s_suppkey"
          FROM (
            SELECT
              "t5"."s_suppkey",
              "t5"."s_name",
              "t5"."s_address",
              "t5"."s_nationkey",
              "t5"."s_phone",
              "t5"."s_acctbal",
              "t5"."s_comment"
            FROM (
              SELECT
                "t2"."S_SUPPKEY" AS "s_suppkey",
                "t2"."S_NAME" AS "s_name",
                "t2"."S_ADDRESS" AS "s_address",
                "t2"."S_NATIONKEY" AS "s_nationkey",
                "t2"."S_PHONE" AS "s_phone",
                "t2"."S_ACCTBAL" AS "s_acctbal",
                "t2"."S_COMMENT" AS "s_comment"
              FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS "t2"
            ) AS "t5"
            WHERE
              "t5"."s_comment" LIKE '%Customer%Complaints%'
          ) AS "t8"
        )
      )
  ) AS "t11"
  GROUP BY
    1,
    2,
    3
) AS "t12"
ORDER BY
  "t12"."supplier_cnt" DESC NULLS LAST,
  "t12"."p_brand" ASC,
  "t12"."p_type" ASC,
  "t12"."p_size" ASC