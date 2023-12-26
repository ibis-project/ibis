SELECT
  "t11"."p_brand",
  "t11"."p_type",
  "t11"."p_size",
  "t11"."supplier_cnt"
FROM (
  SELECT
    "t10"."p_brand",
    "t10"."p_type",
    "t10"."p_size",
    COUNT(DISTINCT "t10"."ps_suppkey") AS "supplier_cnt"
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
        FROM "PARTSUPP" AS "t0"
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
        FROM "PART" AS "t2"
      ) AS "t7"
        ON "t7"."p_partkey" = "t5"."ps_partkey"
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
            "t1"."S_SUPPKEY" AS "s_suppkey"
          FROM "SUPPLIER" AS "t1"
          WHERE
            "t1"."S_COMMENT" LIKE '%Customer%Complaints%'
        )
      )
  ) AS "t10"
  GROUP BY
    1,
    2,
    3
) AS "t11"
ORDER BY
  "t11"."supplier_cnt" DESC NULLS LAST,
  "t11"."p_brand" ASC,
  "t11"."p_type" ASC,
  "t11"."p_size" ASC