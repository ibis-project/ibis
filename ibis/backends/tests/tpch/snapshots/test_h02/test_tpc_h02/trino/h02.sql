WITH "t6" AS (
  SELECT
    *
  FROM "hive"."ibis_sf1"."region" AS "t4"
), "t5" AS (
  SELECT
    *
  FROM "hive"."ibis_sf1"."nation" AS "t3"
), "t9" AS (
  SELECT
    "t2"."s_suppkey",
    "t2"."s_name",
    "t2"."s_address",
    "t2"."s_nationkey",
    "t2"."s_phone",
    CAST("t2"."s_acctbal" AS DECIMAL(15, 2)) AS "s_acctbal",
    "t2"."s_comment"
  FROM "hive"."ibis_sf1"."supplier" AS "t2"
), "t8" AS (
  SELECT
    "t1"."ps_partkey",
    "t1"."ps_suppkey",
    "t1"."ps_availqty",
    CAST("t1"."ps_supplycost" AS DECIMAL(15, 2)) AS "ps_supplycost",
    "t1"."ps_comment"
  FROM "hive"."ibis_sf1"."partsupp" AS "t1"
)
SELECT
  "t23"."s_acctbal",
  "t23"."s_name",
  "t23"."n_name",
  "t23"."p_partkey",
  "t23"."p_mfgr",
  "t23"."s_address",
  "t23"."s_phone",
  "t23"."s_comment"
FROM (
  SELECT
    "t12"."p_partkey",
    "t12"."p_name",
    "t12"."p_mfgr",
    "t12"."p_brand",
    "t12"."p_type",
    "t12"."p_size",
    "t12"."p_container",
    "t12"."p_retailprice",
    "t12"."p_comment",
    "t19"."ps_partkey",
    "t19"."ps_suppkey",
    "t19"."ps_availqty",
    "t19"."ps_supplycost",
    "t19"."ps_comment",
    "t21"."s_suppkey",
    "t21"."s_name",
    "t21"."s_address",
    "t21"."s_nationkey",
    "t21"."s_phone",
    "t21"."s_acctbal",
    "t21"."s_comment",
    "t15"."n_nationkey",
    "t15"."n_name",
    "t15"."n_regionkey",
    "t15"."n_comment",
    "t17"."r_regionkey",
    "t17"."r_name",
    "t17"."r_comment"
  FROM (
    SELECT
      "t0"."p_partkey",
      "t0"."p_name",
      "t0"."p_mfgr",
      "t0"."p_brand",
      "t0"."p_type",
      "t0"."p_size",
      "t0"."p_container",
      CAST("t0"."p_retailprice" AS DECIMAL(15, 2)) AS "p_retailprice",
      "t0"."p_comment"
    FROM "hive"."ibis_sf1"."part" AS "t0"
  ) AS "t12"
  INNER JOIN "t8" AS "t19"
    ON "t12"."p_partkey" = "t19"."ps_partkey"
  INNER JOIN "t9" AS "t21"
    ON "t21"."s_suppkey" = "t19"."ps_suppkey"
  INNER JOIN "t5" AS "t15"
    ON "t21"."s_nationkey" = "t15"."n_nationkey"
  INNER JOIN "t6" AS "t17"
    ON "t15"."n_regionkey" = "t17"."r_regionkey"
) AS "t23"
WHERE
  "t23"."p_size" = 15
  AND "t23"."p_type" LIKE '%BRASS'
  AND "t23"."r_name" = 'EUROPE'
  AND "t23"."ps_supplycost" = (
    SELECT
      MIN("t25"."ps_supplycost") AS "Min(ps_supplycost)"
    FROM (
      SELECT
        *
      FROM (
        SELECT
          "t20"."ps_partkey",
          "t20"."ps_suppkey",
          "t20"."ps_availqty",
          "t20"."ps_supplycost",
          "t20"."ps_comment",
          "t22"."s_suppkey",
          "t22"."s_name",
          "t22"."s_address",
          "t22"."s_nationkey",
          "t22"."s_phone",
          "t22"."s_acctbal",
          "t22"."s_comment",
          "t16"."n_nationkey",
          "t16"."n_name",
          "t16"."n_regionkey",
          "t16"."n_comment",
          "t18"."r_regionkey",
          "t18"."r_name",
          "t18"."r_comment"
        FROM "t8" AS "t20"
        INNER JOIN "t9" AS "t22"
          ON "t22"."s_suppkey" = "t20"."ps_suppkey"
        INNER JOIN "t5" AS "t16"
          ON "t22"."s_nationkey" = "t16"."n_nationkey"
        INNER JOIN "t6" AS "t18"
          ON "t16"."n_regionkey" = "t18"."r_regionkey"
      ) AS "t24"
      WHERE
        "t24"."r_name" = 'EUROPE' AND "t23"."p_partkey" = "t24"."ps_partkey"
    ) AS "t25"
  )
ORDER BY
  "t23"."s_acctbal" DESC,
  "t23"."n_name" ASC,
  "t23"."s_name" ASC,
  "t23"."p_partkey" ASC
LIMIT 100