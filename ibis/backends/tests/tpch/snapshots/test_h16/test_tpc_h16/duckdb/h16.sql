SELECT
  "t9"."p_brand",
  "t9"."p_type",
  "t9"."p_size",
  "t9"."supplier_cnt"
FROM (
  SELECT
    "t8"."p_brand",
    "t8"."p_type",
    "t8"."p_size",
    COUNT(DISTINCT "t8"."ps_suppkey") AS "supplier_cnt"
  FROM (
    SELECT
      "t6"."ps_partkey",
      "t6"."ps_suppkey",
      "t6"."ps_availqty",
      "t6"."ps_supplycost",
      "t6"."ps_comment",
      "t6"."p_partkey",
      "t6"."p_name",
      "t6"."p_mfgr",
      "t6"."p_brand",
      "t6"."p_type",
      "t6"."p_size",
      "t6"."p_container",
      "t6"."p_retailprice",
      "t6"."p_comment"
    FROM (
      SELECT
        "t3"."ps_partkey",
        "t3"."ps_suppkey",
        "t3"."ps_availqty",
        "t3"."ps_supplycost",
        "t3"."ps_comment",
        "t4"."p_partkey",
        "t4"."p_name",
        "t4"."p_mfgr",
        "t4"."p_brand",
        "t4"."p_type",
        "t4"."p_size",
        "t4"."p_container",
        "t4"."p_retailprice",
        "t4"."p_comment"
      FROM "partsupp" AS "t3"
      INNER JOIN "part" AS "t4"
        ON "t4"."p_partkey" = "t3"."ps_partkey"
    ) AS "t6"
    WHERE
      "t6"."p_brand" <> 'Brand#45'
      AND NOT (
        "t6"."p_type" LIKE 'MEDIUM POLISHED%'
      )
      AND "t6"."p_size" IN (CAST(49 AS TINYINT), CAST(14 AS TINYINT), CAST(23 AS TINYINT), CAST(45 AS TINYINT), CAST(19 AS TINYINT), CAST(3 AS TINYINT), CAST(36 AS TINYINT), CAST(9 AS TINYINT))
      AND NOT (
        "t6"."ps_suppkey" IN (
          SELECT
            "t5"."s_suppkey"
          FROM (
            SELECT
              "t2"."s_suppkey",
              "t2"."s_name",
              "t2"."s_address",
              "t2"."s_nationkey",
              "t2"."s_phone",
              "t2"."s_acctbal",
              "t2"."s_comment"
            FROM "supplier" AS "t2"
            WHERE
              "t2"."s_comment" LIKE '%Customer%Complaints%'
          ) AS "t5"
        )
      )
  ) AS "t8"
  GROUP BY
    1,
    2,
    3
) AS "t9"
ORDER BY
  "t9"."supplier_cnt" DESC,
  "t9"."p_brand" ASC,
  "t9"."p_type" ASC,
  "t9"."p_size" ASC