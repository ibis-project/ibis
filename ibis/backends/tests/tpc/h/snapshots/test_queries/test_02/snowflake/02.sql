WITH "t9" AS (
  SELECT
    "t4"."R_REGIONKEY" AS "r_regionkey",
    "t4"."R_NAME" AS "r_name",
    "t4"."R_COMMENT" AS "r_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."REGION" AS "t4"
), "t8" AS (
  SELECT
    "t3"."N_NATIONKEY" AS "n_nationkey",
    "t3"."N_NAME" AS "n_name",
    "t3"."N_REGIONKEY" AS "n_regionkey",
    "t3"."N_COMMENT" AS "n_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS "t3"
), "t7" AS (
  SELECT
    "t2"."S_SUPPKEY" AS "s_suppkey",
    "t2"."S_NAME" AS "s_name",
    "t2"."S_ADDRESS" AS "s_address",
    "t2"."S_NATIONKEY" AS "s_nationkey",
    "t2"."S_PHONE" AS "s_phone",
    "t2"."S_ACCTBAL" AS "s_acctbal",
    "t2"."S_COMMENT" AS "s_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS "t2"
), "t6" AS (
  SELECT
    "t1"."PS_PARTKEY" AS "ps_partkey",
    "t1"."PS_SUPPKEY" AS "ps_suppkey",
    "t1"."PS_AVAILQTY" AS "ps_availqty",
    "t1"."PS_SUPPLYCOST" AS "ps_supplycost",
    "t1"."PS_COMMENT" AS "ps_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PARTSUPP" AS "t1"
)
SELECT
  "t23"."s_acctbal",
  "t23"."s_name",
  "t23"."n_name",
  "t23"."p_partkey",
  "t23"."p_mfgr",
  "t23"."s_address",
  "t23"."s_phone",
  "t23"."s_comment"
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
    "t15"."ps_partkey",
    "t15"."ps_suppkey",
    "t15"."ps_availqty",
    "t15"."ps_supplycost",
    "t15"."ps_comment",
    "t17"."s_suppkey",
    "t17"."s_name",
    "t17"."s_address",
    "t17"."s_nationkey",
    "t17"."s_phone",
    "t17"."s_acctbal",
    "t17"."s_comment",
    "t19"."n_nationkey",
    "t19"."n_name",
    "t19"."n_regionkey",
    "t19"."n_comment",
    "t21"."r_regionkey",
    "t21"."r_name",
    "t21"."r_comment"
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
  INNER JOIN "t6" AS "t15"
    ON "t10"."p_partkey" = "t15"."ps_partkey"
  INNER JOIN "t7" AS "t17"
    ON "t17"."s_suppkey" = "t15"."ps_suppkey"
  INNER JOIN "t8" AS "t19"
    ON "t17"."s_nationkey" = "t19"."n_nationkey"
  INNER JOIN "t9" AS "t21"
    ON "t19"."n_regionkey" = "t21"."r_regionkey"
) AS "t23"
WHERE
  "t23"."p_size" = 15
  AND "t23"."p_type" LIKE '%BRASS'
  AND "t23"."r_name" = 'EUROPE'
  AND "t23"."ps_supplycost" = (
    SELECT
      MIN("t25"."ps_supplycost") AS "Min(ps_supplycost)"
    FROM (
      SELECT
        *
      FROM (
        SELECT
          "t16"."ps_partkey",
          "t16"."ps_suppkey",
          "t16"."ps_availqty",
          "t16"."ps_supplycost",
          "t16"."ps_comment",
          "t18"."s_suppkey",
          "t18"."s_name",
          "t18"."s_address",
          "t18"."s_nationkey",
          "t18"."s_phone",
          "t18"."s_acctbal",
          "t18"."s_comment",
          "t20"."n_nationkey",
          "t20"."n_name",
          "t20"."n_regionkey",
          "t20"."n_comment",
          "t22"."r_regionkey",
          "t22"."r_name",
          "t22"."r_comment"
        FROM "t6" AS "t16"
        INNER JOIN "t7" AS "t18"
          ON "t18"."s_suppkey" = "t16"."ps_suppkey"
        INNER JOIN "t8" AS "t20"
          ON "t18"."s_nationkey" = "t20"."n_nationkey"
        INNER JOIN "t9" AS "t22"
          ON "t20"."n_regionkey" = "t22"."r_regionkey"
      ) AS "t24"
      WHERE
        "t24"."r_name" = 'EUROPE' AND "t23"."p_partkey" = "t24"."ps_partkey"
    ) AS "t25"
  )
ORDER BY
  "t23"."s_acctbal" DESC NULLS LAST,
  "t23"."n_name" ASC,
  "t23"."s_name" ASC,
  "t23"."p_partkey" ASC
LIMIT 100