SELECT
  "t19"."s_acctbal",
  "t19"."s_name",
  "t19"."n_name",
  "t19"."p_partkey",
  "t19"."p_mfgr",
  "t19"."s_address",
  "t19"."s_phone",
  "t19"."s_comment"
FROM (
  SELECT
    "t10"."p_partkey",
    "t10"."p_name",
    "t10"."p_mfgr",
    "t10"."p_brand",
    "t10"."p_type",
    "t10"."p_size",
    "t10"."p_container",
    "t10"."p_retailprice",
    "t10"."p_comment",
    "t11"."ps_partkey",
    "t11"."ps_suppkey",
    "t11"."ps_availqty",
    "t11"."ps_supplycost",
    "t11"."ps_comment",
    "t13"."s_suppkey",
    "t13"."s_name",
    "t13"."s_address",
    "t13"."s_nationkey",
    "t13"."s_phone",
    "t13"."s_acctbal",
    "t13"."s_comment",
    "t15"."n_nationkey",
    "t15"."n_name",
    "t15"."n_regionkey",
    "t15"."n_comment",
    "t17"."r_regionkey",
    "t17"."r_name",
    "t17"."r_comment"
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
  ) AS "t10"
  INNER JOIN (
    SELECT
      "t1"."PS_PARTKEY" AS "ps_partkey",
      "t1"."PS_SUPPKEY" AS "ps_suppkey",
      "t1"."PS_AVAILQTY" AS "ps_availqty",
      "t1"."PS_SUPPLYCOST" AS "ps_supplycost",
      "t1"."PS_COMMENT" AS "ps_comment"
    FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PARTSUPP" AS "t1"
  ) AS "t11"
    ON "t10"."p_partkey" = "t11"."ps_partkey"
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
  ) AS "t13"
    ON "t13"."s_suppkey" = "t11"."ps_suppkey"
  INNER JOIN (
    SELECT
      "t3"."N_NATIONKEY" AS "n_nationkey",
      "t3"."N_NAME" AS "n_name",
      "t3"."N_REGIONKEY" AS "n_regionkey",
      "t3"."N_COMMENT" AS "n_comment"
    FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS "t3"
  ) AS "t15"
    ON "t13"."s_nationkey" = "t15"."n_nationkey"
  INNER JOIN (
    SELECT
      "t4"."R_REGIONKEY" AS "r_regionkey",
      "t4"."R_NAME" AS "r_name",
      "t4"."R_COMMENT" AS "r_comment"
    FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."REGION" AS "t4"
  ) AS "t17"
    ON "t15"."n_regionkey" = "t17"."r_regionkey"
) AS "t19"
WHERE
  "t19"."p_size" = 15
  AND "t19"."p_type" LIKE '%BRASS'
  AND "t19"."r_name" = 'EUROPE'
  AND "t19"."ps_supplycost" = (
    SELECT
      MIN("t21"."ps_supplycost") AS "Min(ps_supplycost)"
    FROM (
      SELECT
        "t20"."ps_partkey",
        "t20"."ps_suppkey",
        "t20"."ps_availqty",
        "t20"."ps_supplycost",
        "t20"."ps_comment",
        "t20"."s_suppkey",
        "t20"."s_name",
        "t20"."s_address",
        "t20"."s_nationkey",
        "t20"."s_phone",
        "t20"."s_acctbal",
        "t20"."s_comment",
        "t20"."n_nationkey",
        "t20"."n_name",
        "t20"."n_regionkey",
        "t20"."n_comment",
        "t20"."r_regionkey",
        "t20"."r_name",
        "t20"."r_comment"
      FROM (
        SELECT
          "t12"."ps_partkey",
          "t12"."ps_suppkey",
          "t12"."ps_availqty",
          "t12"."ps_supplycost",
          "t12"."ps_comment",
          "t14"."s_suppkey",
          "t14"."s_name",
          "t14"."s_address",
          "t14"."s_nationkey",
          "t14"."s_phone",
          "t14"."s_acctbal",
          "t14"."s_comment",
          "t16"."n_nationkey",
          "t16"."n_name",
          "t16"."n_regionkey",
          "t16"."n_comment",
          "t18"."r_regionkey",
          "t18"."r_name",
          "t18"."r_comment"
        FROM (
          SELECT
            "t1"."PS_PARTKEY" AS "ps_partkey",
            "t1"."PS_SUPPKEY" AS "ps_suppkey",
            "t1"."PS_AVAILQTY" AS "ps_availqty",
            "t1"."PS_SUPPLYCOST" AS "ps_supplycost",
            "t1"."PS_COMMENT" AS "ps_comment"
          FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PARTSUPP" AS "t1"
        ) AS "t12"
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
        ) AS "t14"
          ON "t14"."s_suppkey" = "t12"."ps_suppkey"
        INNER JOIN (
          SELECT
            "t3"."N_NATIONKEY" AS "n_nationkey",
            "t3"."N_NAME" AS "n_name",
            "t3"."N_REGIONKEY" AS "n_regionkey",
            "t3"."N_COMMENT" AS "n_comment"
          FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS "t3"
        ) AS "t16"
          ON "t14"."s_nationkey" = "t16"."n_nationkey"
        INNER JOIN (
          SELECT
            "t4"."R_REGIONKEY" AS "r_regionkey",
            "t4"."R_NAME" AS "r_name",
            "t4"."R_COMMENT" AS "r_comment"
          FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."REGION" AS "t4"
        ) AS "t18"
          ON "t16"."n_regionkey" = "t18"."r_regionkey"
      ) AS "t20"
      WHERE
        "t20"."r_name" = 'EUROPE' AND "t19"."p_partkey" = "t20"."ps_partkey"
    ) AS "t21"
  )
ORDER BY
  "t19"."s_acctbal" DESC NULLS LAST,
  "t19"."n_name" ASC,
  "t19"."s_name" ASC,
  "t19"."p_partkey" ASC
LIMIT 100