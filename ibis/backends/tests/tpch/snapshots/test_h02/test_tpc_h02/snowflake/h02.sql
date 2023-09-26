WITH t1 AS (
  SELECT
    t7."P_PARTKEY" AS "p_partkey",
    t7."P_NAME" AS "p_name",
    t7."P_MFGR" AS "p_mfgr",
    t7."P_BRAND" AS "p_brand",
    t7."P_TYPE" AS "p_type",
    t7."P_SIZE" AS "p_size",
    t7."P_CONTAINER" AS "p_container",
    t7."P_RETAILPRICE" AS "p_retailprice",
    t7."P_COMMENT" AS "p_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PART" AS t7
), t0 AS (
  SELECT
    t7."PS_PARTKEY" AS "ps_partkey",
    t7."PS_SUPPKEY" AS "ps_suppkey",
    t7."PS_AVAILQTY" AS "ps_availqty",
    t7."PS_SUPPLYCOST" AS "ps_supplycost",
    t7."PS_COMMENT" AS "ps_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PARTSUPP" AS t7
), t2 AS (
  SELECT
    t7."S_SUPPKEY" AS "s_suppkey",
    t7."S_NAME" AS "s_name",
    t7."S_ADDRESS" AS "s_address",
    t7."S_NATIONKEY" AS "s_nationkey",
    t7."S_PHONE" AS "s_phone",
    t7."S_ACCTBAL" AS "s_acctbal",
    t7."S_COMMENT" AS "s_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS t7
), t3 AS (
  SELECT
    t7."N_NATIONKEY" AS "n_nationkey",
    t7."N_NAME" AS "n_name",
    t7."N_REGIONKEY" AS "n_regionkey",
    t7."N_COMMENT" AS "n_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS t7
), t4 AS (
  SELECT
    t7."R_REGIONKEY" AS "r_regionkey",
    t7."R_NAME" AS "r_name",
    t7."R_COMMENT" AS "r_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."REGION" AS t7
), t5 AS (
  SELECT
    t1."p_partkey" AS "p_partkey",
    t1."p_name" AS "p_name",
    t1."p_mfgr" AS "p_mfgr",
    t1."p_brand" AS "p_brand",
    t1."p_type" AS "p_type",
    t1."p_size" AS "p_size",
    t1."p_container" AS "p_container",
    t1."p_retailprice" AS "p_retailprice",
    t1."p_comment" AS "p_comment",
    t0."ps_partkey" AS "ps_partkey",
    t0."ps_suppkey" AS "ps_suppkey",
    t0."ps_availqty" AS "ps_availqty",
    t0."ps_supplycost" AS "ps_supplycost",
    t0."ps_comment" AS "ps_comment",
    t2."s_suppkey" AS "s_suppkey",
    t2."s_name" AS "s_name",
    t2."s_address" AS "s_address",
    t2."s_nationkey" AS "s_nationkey",
    t2."s_phone" AS "s_phone",
    t2."s_acctbal" AS "s_acctbal",
    t2."s_comment" AS "s_comment",
    t3."n_nationkey" AS "n_nationkey",
    t3."n_name" AS "n_name",
    t3."n_regionkey" AS "n_regionkey",
    t3."n_comment" AS "n_comment",
    t4."r_regionkey" AS "r_regionkey",
    t4."r_name" AS "r_name",
    t4."r_comment" AS "r_comment"
  FROM t1
  JOIN t0
    ON t1."p_partkey" = t0."ps_partkey"
  JOIN t2
    ON t2."s_suppkey" = t0."ps_suppkey"
  JOIN t3
    ON t2."s_nationkey" = t3."n_nationkey"
  JOIN t4
    ON t3."n_regionkey" = t4."r_regionkey"
  WHERE
    t1."p_size" = 15
    AND t1."p_type" LIKE '%BRASS'
    AND t4."r_name" = 'EUROPE'
    AND t0."ps_supplycost" = (
      SELECT
        MIN(t0."ps_supplycost") AS "Min(ps_supplycost)"
      FROM t0
      JOIN t2
        ON t2."s_suppkey" = t0."ps_suppkey"
      JOIN t3
        ON t2."s_nationkey" = t3."n_nationkey"
      JOIN t4
        ON t3."n_regionkey" = t4."r_regionkey"
      WHERE
        t4."r_name" = 'EUROPE' AND t1."p_partkey" = t0."ps_partkey"
    )
)
SELECT
  t6."s_acctbal",
  t6."s_name",
  t6."n_name",
  t6."p_partkey",
  t6."p_mfgr",
  t6."s_address",
  t6."s_phone",
  t6."s_comment"
FROM (
  SELECT
    t5."s_acctbal" AS "s_acctbal",
    t5."s_name" AS "s_name",
    t5."n_name" AS "n_name",
    t5."p_partkey" AS "p_partkey",
    t5."p_mfgr" AS "p_mfgr",
    t5."s_address" AS "s_address",
    t5."s_phone" AS "s_phone",
    t5."s_comment" AS "s_comment"
  FROM t5
) AS t6
ORDER BY
  t6."s_acctbal" DESC,
  t6."n_name" ASC,
  t6."s_name" ASC,
  t6."p_partkey" ASC
LIMIT 100