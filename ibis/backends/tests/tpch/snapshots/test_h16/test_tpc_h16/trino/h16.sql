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
            "t1"."s_suppkey"
          FROM "hive"."ibis_sf1"."supplier" AS "t1"
          WHERE
            "t1"."s_comment" LIKE '%Customer%Complaints%'
        )
      )
  ) AS "t9"
  GROUP BY
    1,
    2,
    3
) AS "t10"
ORDER BY
  "t10"."supplier_cnt" DESC,
  "t10"."p_brand" ASC,
  "t10"."p_type" ASC,
  "t10"."p_size" ASC