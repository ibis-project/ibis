SELECT
  "t26"."s_acctbal",
  "t26"."s_name",
  "t26"."n_name",
  "t26"."p_partkey",
  "t26"."p_mfgr",
  "t26"."s_address",
  "t26"."s_phone",
  "t26"."s_comment"
FROM (
  SELECT
    "t14"."p_partkey",
    "t14"."p_name",
    "t14"."p_mfgr",
    "t14"."p_brand",
    "t14"."p_type",
    "t14"."p_size",
    "t14"."p_container",
    "t14"."p_retailprice",
    "t14"."p_comment",
    "t15"."ps_partkey",
    "t15"."ps_suppkey",
    "t15"."ps_availqty",
    "t15"."ps_supplycost",
    "t15"."ps_comment",
    "t17"."s_suppkey",
    "t17"."s_name",
    "t17"."s_address",
    "t17"."s_nationkey",
    "t17"."s_phone",
    "t17"."s_acctbal",
    "t17"."s_comment",
    "t10"."n_nationkey",
    "t10"."n_name",
    "t10"."n_regionkey",
    "t10"."n_comment",
    "t12"."r_regionkey",
    "t12"."r_name",
    "t12"."r_comment"
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
  ) AS "t14"
  INNER JOIN (
    SELECT
      "t1"."ps_partkey",
      "t1"."ps_suppkey",
      "t1"."ps_availqty",
      CAST("t1"."ps_supplycost" AS DECIMAL(15, 2)) AS "ps_supplycost",
      "t1"."ps_comment"
    FROM "hive"."ibis_sf1"."partsupp" AS "t1"
  ) AS "t15"
    ON "t14"."p_partkey" = "t15"."ps_partkey"
  INNER JOIN (
    SELECT
      "t2"."s_suppkey",
      "t2"."s_name",
      "t2"."s_address",
      "t2"."s_nationkey",
      "t2"."s_phone",
      CAST("t2"."s_acctbal" AS DECIMAL(15, 2)) AS "s_acctbal",
      "t2"."s_comment"
    FROM "hive"."ibis_sf1"."supplier" AS "t2"
  ) AS "t17"
    ON "t17"."s_suppkey" = "t15"."ps_suppkey"
  INNER JOIN (
    SELECT
      "t3"."n_nationkey",
      "t3"."n_name",
      "t3"."n_regionkey",
      "t3"."n_comment"
    FROM "hive"."ibis_sf1"."nation" AS "t3"
  ) AS "t10"
    ON "t17"."s_nationkey" = "t10"."n_nationkey"
  INNER JOIN (
    SELECT
      "t4"."r_regionkey",
      "t4"."r_name",
      "t4"."r_comment"
    FROM "hive"."ibis_sf1"."region" AS "t4"
  ) AS "t12"
    ON "t10"."n_regionkey" = "t12"."r_regionkey"
) AS "t26"
WHERE
  "t26"."p_size" = 15
  AND "t26"."p_type" LIKE '%BRASS'
  AND "t26"."r_name" = 'EUROPE'
  AND "t26"."ps_supplycost" = (
    SELECT
      MIN("t28"."ps_supplycost") AS "Min(ps_supplycost)"
    FROM (
      SELECT
        "t27"."ps_partkey",
        "t27"."ps_suppkey",
        "t27"."ps_availqty",
        "t27"."ps_supplycost",
        "t27"."ps_comment",
        "t27"."s_suppkey",
        "t27"."s_name",
        "t27"."s_address",
        "t27"."s_nationkey",
        "t27"."s_phone",
        "t27"."s_acctbal",
        "t27"."s_comment",
        "t27"."n_nationkey",
        "t27"."n_name",
        "t27"."n_regionkey",
        "t27"."n_comment",
        "t27"."r_regionkey",
        "t27"."r_name",
        "t27"."r_comment"
      FROM (
        SELECT
          "t16"."ps_partkey",
          "t16"."ps_suppkey",
          "t16"."ps_availqty",
          "t16"."ps_supplycost",
          "t16"."ps_comment",
          "t18"."s_suppkey",
          "t18"."s_name",
          "t18"."s_address",
          "t18"."s_nationkey",
          "t18"."s_phone",
          "t18"."s_acctbal",
          "t18"."s_comment",
          "t11"."n_nationkey",
          "t11"."n_name",
          "t11"."n_regionkey",
          "t11"."n_comment",
          "t13"."r_regionkey",
          "t13"."r_name",
          "t13"."r_comment"
        FROM (
          SELECT
            "t1"."ps_partkey",
            "t1"."ps_suppkey",
            "t1"."ps_availqty",
            CAST("t1"."ps_supplycost" AS DECIMAL(15, 2)) AS "ps_supplycost",
            "t1"."ps_comment"
          FROM "hive"."ibis_sf1"."partsupp" AS "t1"
        ) AS "t16"
        INNER JOIN (
          SELECT
            "t2"."s_suppkey",
            "t2"."s_name",
            "t2"."s_address",
            "t2"."s_nationkey",
            "t2"."s_phone",
            CAST("t2"."s_acctbal" AS DECIMAL(15, 2)) AS "s_acctbal",
            "t2"."s_comment"
          FROM "hive"."ibis_sf1"."supplier" AS "t2"
        ) AS "t18"
          ON "t18"."s_suppkey" = "t16"."ps_suppkey"
        INNER JOIN (
          SELECT
            "t3"."n_nationkey",
            "t3"."n_name",
            "t3"."n_regionkey",
            "t3"."n_comment"
          FROM "hive"."ibis_sf1"."nation" AS "t3"
        ) AS "t11"
          ON "t18"."s_nationkey" = "t11"."n_nationkey"
        INNER JOIN (
          SELECT
            "t4"."r_regionkey",
            "t4"."r_name",
            "t4"."r_comment"
          FROM "hive"."ibis_sf1"."region" AS "t4"
        ) AS "t13"
          ON "t11"."n_regionkey" = "t13"."r_regionkey"
      ) AS "t27"
      WHERE
        "t27"."r_name" = 'EUROPE' AND "t26"."p_partkey" = "t27"."ps_partkey"
    ) AS "t28"
  )
ORDER BY
  "t26"."s_acctbal" DESC,
  "t26"."n_name" ASC,
  "t26"."s_name" ASC,
  "t26"."p_partkey" ASC
LIMIT 100