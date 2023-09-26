WITH t1 AS (
  SELECT
    t4."PS_PARTKEY" AS "ps_partkey",
    t4."PS_SUPPKEY" AS "ps_suppkey",
    t4."PS_AVAILQTY" AS "ps_availqty",
    t4."PS_SUPPLYCOST" AS "ps_supplycost",
    t4."PS_COMMENT" AS "ps_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PARTSUPP" AS t4
), t2 AS (
  SELECT
    t4."P_PARTKEY" AS "p_partkey",
    t4."P_NAME" AS "p_name",
    t4."P_MFGR" AS "p_mfgr",
    t4."P_BRAND" AS "p_brand",
    t4."P_TYPE" AS "p_type",
    t4."P_SIZE" AS "p_size",
    t4."P_CONTAINER" AS "p_container",
    t4."P_RETAILPRICE" AS "p_retailprice",
    t4."P_COMMENT" AS "p_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PART" AS t4
), t0 AS (
  SELECT
    t4."S_SUPPKEY" AS "s_suppkey",
    t4."S_NAME" AS "s_name",
    t4."S_ADDRESS" AS "s_address",
    t4."S_NATIONKEY" AS "s_nationkey",
    t4."S_PHONE" AS "s_phone",
    t4."S_ACCTBAL" AS "s_acctbal",
    t4."S_COMMENT" AS "s_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS t4
)
SELECT
  t3."p_brand",
  t3."p_type",
  t3."p_size",
  t3."supplier_cnt"
FROM (
  SELECT
    t2."p_brand" AS "p_brand",
    t2."p_type" AS "p_type",
    t2."p_size" AS "p_size",
    COUNT(DISTINCT t1."ps_suppkey") AS "supplier_cnt"
  FROM t1
  JOIN t2
    ON t2."p_partkey" = t1."ps_partkey"
  WHERE
    t2."p_brand" <> 'Brand#45'
    AND NOT t2."p_type" LIKE 'MEDIUM POLISHED%'
    AND t2."p_size" IN (49, 14, 23, 45, 19, 3, 36, 9)
    AND (
      NOT t1."ps_suppkey" IN (
        SELECT
          t4."s_suppkey"
        FROM (
          SELECT
            t0."s_suppkey" AS "s_suppkey",
            t0."s_name" AS "s_name",
            t0."s_address" AS "s_address",
            t0."s_nationkey" AS "s_nationkey",
            t0."s_phone" AS "s_phone",
            t0."s_acctbal" AS "s_acctbal",
            t0."s_comment" AS "s_comment"
          FROM t0
          WHERE
            t0."s_comment" LIKE '%Customer%Complaints%'
        ) AS t4
      )
    )
  GROUP BY
    1,
    2,
    3
) AS t3
ORDER BY
  t3."supplier_cnt" DESC,
  t3."p_brand" ASC,
  t3."p_type" ASC,
  t3."p_size" ASC