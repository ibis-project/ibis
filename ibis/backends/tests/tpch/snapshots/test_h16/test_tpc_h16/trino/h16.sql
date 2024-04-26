SELECT
  *
FROM (
  SELECT
    "t11"."p_brand",
    "t11"."p_type",
    "t11"."p_size",
    COUNT(DISTINCT "t11"."ps_suppkey") AS "supplier_cnt"
  FROM (
    SELECT
      *
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
          "t1"."p_partkey",
          "t1"."p_name",
          "t1"."p_mfgr",
          "t1"."p_brand",
          "t1"."p_type",
          "t1"."p_size",
          "t1"."p_container",
          CAST("t1"."p_retailprice" AS DECIMAL(15, 2)) AS "p_retailprice",
          "t1"."p_comment"
        FROM "hive"."ibis_sf1"."part" AS "t1"
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
              *
            FROM (
              SELECT
                "t2"."s_suppkey",
                "t2"."s_name",
                "t2"."s_address",
                "t2"."s_nationkey",
                "t2"."s_phone",
                CAST("t2"."s_acctbal" AS DECIMAL(15, 2)) AS "s_acctbal",
                "t2"."s_comment"
              FROM "hive"."ibis_sf1"."supplier" AS "t2"
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
  "t12"."supplier_cnt" DESC,
  "t12"."p_brand" ASC,
  "t12"."p_type" ASC,
  "t12"."p_size" ASC