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
          "t0"."ps_partkey",
          "t0"."ps_suppkey",
          "t0"."ps_availqty",
          CAST("t0"."ps_supplycost" AS DECIMAL(15, 2)) AS "ps_supplycost",
          "t0"."ps_comment"
        FROM "hive"."ibis_sf1"."partsupp" AS "t0"
      ) AS "t6"
      INNER JOIN (
        SELECT
          "t2"."p_partkey",
          "t2"."p_name",
          "t2"."p_mfgr",
          "t2"."p_brand",
          "t2"."p_type",
          "t2"."p_size",
          "t2"."p_container",
          CAST("t2"."p_retailprice" AS DECIMAL(15, 2)) AS "p_retailprice",
          "t2"."p_comment"
        FROM "hive"."ibis_sf1"."part" AS "t2"
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
            "t1"."s_suppkey"
          FROM "hive"."ibis_sf1"."supplier" AS "t1"
          WHERE
            "t1"."s_comment" LIKE '%Customer%Complaints%'
        )
      )
  ) AS "t10"
  GROUP BY
    1,
    2,
    3
) AS "t11"
ORDER BY
  "t11"."supplier_cnt" DESC,
  "t11"."p_brand" ASC,
  "t11"."p_type" ASC,
  "t11"."p_size" ASC