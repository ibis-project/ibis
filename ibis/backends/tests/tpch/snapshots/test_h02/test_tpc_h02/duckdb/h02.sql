SELECT
  "t14"."s_acctbal",
  "t14"."s_name",
  "t14"."n_name",
  "t14"."p_partkey",
  "t14"."p_mfgr",
  "t14"."s_address",
  "t14"."s_phone",
  "t14"."s_comment"
FROM (
  SELECT
    "t5"."p_partkey",
    "t5"."p_name",
    "t5"."p_mfgr",
    "t5"."p_brand",
    "t5"."p_type",
    "t5"."p_size",
    "t5"."p_container",
    "t5"."p_retailprice",
    "t5"."p_comment",
    "t6"."ps_partkey",
    "t6"."ps_suppkey",
    "t6"."ps_availqty",
    "t6"."ps_supplycost",
    "t6"."ps_comment",
    "t8"."s_suppkey",
    "t8"."s_name",
    "t8"."s_address",
    "t8"."s_nationkey",
    "t8"."s_phone",
    "t8"."s_acctbal",
    "t8"."s_comment",
    "t10"."n_nationkey",
    "t10"."n_name",
    "t10"."n_regionkey",
    "t10"."n_comment",
    "t12"."r_regionkey",
    "t12"."r_name",
    "t12"."r_comment"
  FROM "part" AS "t5"
  INNER JOIN "partsupp" AS "t6"
    ON "t5"."p_partkey" = "t6"."ps_partkey"
  INNER JOIN "supplier" AS "t8"
    ON "t8"."s_suppkey" = "t6"."ps_suppkey"
  INNER JOIN "nation" AS "t10"
    ON "t8"."s_nationkey" = "t10"."n_nationkey"
  INNER JOIN "region" AS "t12"
    ON "t10"."n_regionkey" = "t12"."r_regionkey"
) AS "t14"
WHERE
  "t14"."p_size" = CAST(15 AS TINYINT)
  AND "t14"."p_type" LIKE '%BRASS'
  AND "t14"."r_name" = 'EUROPE'
  AND "t14"."ps_supplycost" = (
    SELECT
      MIN("t16"."ps_supplycost") AS "Min(ps_supplycost)"
    FROM (
      SELECT
        "t15"."ps_partkey",
        "t15"."ps_suppkey",
        "t15"."ps_availqty",
        "t15"."ps_supplycost",
        "t15"."ps_comment",
        "t15"."s_suppkey",
        "t15"."s_name",
        "t15"."s_address",
        "t15"."s_nationkey",
        "t15"."s_phone",
        "t15"."s_acctbal",
        "t15"."s_comment",
        "t15"."n_nationkey",
        "t15"."n_name",
        "t15"."n_regionkey",
        "t15"."n_comment",
        "t15"."r_regionkey",
        "t15"."r_name",
        "t15"."r_comment"
      FROM (
        SELECT
          "t7"."ps_partkey",
          "t7"."ps_suppkey",
          "t7"."ps_availqty",
          "t7"."ps_supplycost",
          "t7"."ps_comment",
          "t9"."s_suppkey",
          "t9"."s_name",
          "t9"."s_address",
          "t9"."s_nationkey",
          "t9"."s_phone",
          "t9"."s_acctbal",
          "t9"."s_comment",
          "t11"."n_nationkey",
          "t11"."n_name",
          "t11"."n_regionkey",
          "t11"."n_comment",
          "t13"."r_regionkey",
          "t13"."r_name",
          "t13"."r_comment"
        FROM "partsupp" AS "t7"
        INNER JOIN "supplier" AS "t9"
          ON "t9"."s_suppkey" = "t7"."ps_suppkey"
        INNER JOIN "nation" AS "t11"
          ON "t9"."s_nationkey" = "t11"."n_nationkey"
        INNER JOIN "region" AS "t13"
          ON "t11"."n_regionkey" = "t13"."r_regionkey"
      ) AS "t15"
      WHERE
        "t15"."r_name" = 'EUROPE' AND "t14"."p_partkey" = "t15"."ps_partkey"
    ) AS "t16"
  )
ORDER BY
  "t14"."s_acctbal" DESC,
  "t14"."n_name" ASC,
  "t14"."s_name" ASC,
  "t14"."p_partkey" ASC
LIMIT 100