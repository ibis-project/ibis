SELECT
  "t10"."p_brand" AS "p_brand",
  "t10"."p_type" AS "p_type",
  "t10"."p_size" AS "p_size",
  "t10"."supplier_cnt" AS "supplier_cnt"
FROM (
  SELECT
    "t9"."p_brand" AS "p_brand",
    "t9"."p_type" AS "p_type",
    "t9"."p_size" AS "p_size",
    COUNT(DISTINCT "t9"."ps_suppkey") AS "supplier_cnt"
  FROM (
    SELECT
      "t8"."ps_partkey" AS "ps_partkey",
      "t8"."ps_suppkey" AS "ps_suppkey",
      "t8"."ps_availqty" AS "ps_availqty",
      "t8"."ps_supplycost" AS "ps_supplycost",
      "t8"."ps_comment" AS "ps_comment",
      "t8"."p_partkey" AS "p_partkey",
      "t8"."p_name" AS "p_name",
      "t8"."p_mfgr" AS "p_mfgr",
      "t8"."p_brand" AS "p_brand",
      "t8"."p_type" AS "p_type",
      "t8"."p_size" AS "p_size",
      "t8"."p_container" AS "p_container",
      "t8"."p_retailprice" AS "p_retailprice",
      "t8"."p_comment" AS "p_comment"
    FROM (
      SELECT
        "t3"."ps_partkey" AS "ps_partkey",
        "t3"."ps_suppkey" AS "ps_suppkey",
        "t3"."ps_availqty" AS "ps_availqty",
        "t3"."ps_supplycost" AS "ps_supplycost",
        "t3"."ps_comment" AS "ps_comment",
        "t6"."p_partkey" AS "p_partkey",
        "t6"."p_name" AS "p_name",
        "t6"."p_mfgr" AS "p_mfgr",
        "t6"."p_brand" AS "p_brand",
        "t6"."p_type" AS "p_type",
        "t6"."p_size" AS "p_size",
        "t6"."p_container" AS "p_container",
        "t6"."p_retailprice" AS "p_retailprice",
        "t6"."p_comment" AS "p_comment"
      FROM (
        SELECT
          "t0"."PS_PARTKEY" AS "ps_partkey",
          "t0"."PS_SUPPKEY" AS "ps_suppkey",
          "t0"."PS_AVAILQTY" AS "ps_availqty",
          "t0"."PS_SUPPLYCOST" AS "ps_supplycost",
          "t0"."PS_COMMENT" AS "ps_comment"
        FROM "PARTSUPP" AS "t0"
      ) AS "t3"
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
      ) AS "t6"
        ON "t6"."p_partkey" = "t3"."ps_partkey"
    ) AS "t8"
    WHERE
      "t8"."p_brand" <> 'Brand#45'
      AND NOT (
        "t8"."p_type" LIKE 'MEDIUM POLISHED%'
      )
      AND "t8"."p_size" IN (49, 14, 23, 45, 19, 3, 36, 9)
      AND NOT (
        "t8"."ps_suppkey" IN ((
          SELECT
            "t1"."S_SUPPKEY" AS "s_suppkey"
          FROM "SUPPLIER" AS "t1"
          WHERE
            "t1"."S_COMMENT" LIKE '%Customer%Complaints%'
        ))
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