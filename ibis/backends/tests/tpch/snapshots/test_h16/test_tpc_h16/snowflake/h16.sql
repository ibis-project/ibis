SELECT
  "t10"."p_brand",
  "t10"."p_type",
  "t10"."p_size",
  "t10"."supplier_cnt"
FROM (
  SELECT
    "t9"."p_brand",
    "t9"."p_type",
    "t9"."p_size",
    COUNT(DISTINCT "t9"."ps_suppkey") AS "supplier_cnt"
  FROM (
    SELECT
      "t8"."ps_partkey",
      "t8"."ps_suppkey",
      "t8"."ps_availqty",
      "t8"."ps_supplycost",
      "t8"."ps_comment",
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
        "t5"."ps_partkey",
        "t5"."ps_suppkey",
        "t5"."ps_availqty",
        "t5"."ps_supplycost",
        "t5"."ps_comment",
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
      ) AS "t5"
      INNER JOIN (
        SELECT
          "t2"."P_PARTKEY" AS "p_partkey",
          "t2"."P_NAME" AS "p_name",
          "t2"."P_MFGR" AS "p_mfgr",
          "t2"."P_BRAND" AS "p_brand",
          "t2"."P_TYPE" AS "p_type",
          "t2"."P_SIZE" AS "p_size",
          "t2"."P_CONTAINER" AS "p_container",
          "t2"."P_RETAILPRICE" AS "p_retailprice",
          "t2"."P_COMMENT" AS "p_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PART" AS "t2"
      ) AS "t7"
        ON "t7"."p_partkey" = "t5"."ps_partkey"
    ) AS "t8"
    WHERE
      "t8"."p_brand" <> 'Brand#45'
      AND NOT (
        "t8"."p_type" LIKE 'MEDIUM POLISHED%'
      )
      AND "t8"."p_size" IN (49, 14, 23, 45, 19, 3, 36, 9)
      AND NOT (
        "t8"."ps_suppkey" IN (
          SELECT
            "t1"."S_SUPPKEY" AS "s_suppkey"
          FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS "t1"
          WHERE
            "t1"."S_COMMENT" LIKE '%Customer%Complaints%'
        )
      )
  ) AS "t9"
  GROUP BY
    1,
    2,
    3
) AS "t10"
ORDER BY
  "t10"."supplier_cnt" DESC NULLS LAST,
  "t10"."p_brand" ASC,
  "t10"."p_type" ASC,
  "t10"."p_size" ASC