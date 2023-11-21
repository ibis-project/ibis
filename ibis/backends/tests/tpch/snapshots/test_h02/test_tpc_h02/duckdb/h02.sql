SELECT
  *
FROM (
  SELECT
    *
  FROM (
    SELECT
      t13.s_acctbal AS s_acctbal,
      t13.s_name AS s_name,
      t13.n_name AS n_name,
      t13.p_partkey AS p_partkey,
      t13.p_mfgr AS p_mfgr,
      t13.s_address AS s_address,
      t13.s_phone AS s_phone,
      t13.s_comment AS s_comment
    FROM (
      SELECT
        *
      FROM (
        SELECT
          t0.p_partkey AS p_partkey,
          t0.p_name AS p_name,
          t0.p_mfgr AS p_mfgr,
          t0.p_brand AS p_brand,
          t0.p_type AS p_type,
          t0.p_size AS p_size,
          t0.p_container AS p_container,
          t0.p_retailprice AS p_retailprice,
          t0.p_comment AS p_comment,
          t1.ps_partkey AS ps_partkey,
          t1.ps_suppkey AS ps_suppkey,
          t1.ps_availqty AS ps_availqty,
          t1.ps_supplycost AS ps_supplycost,
          t1.ps_comment AS ps_comment,
          t2.s_suppkey AS s_suppkey,
          t2.s_name AS s_name,
          t2.s_address AS s_address,
          t2.s_nationkey AS s_nationkey,
          t2.s_phone AS s_phone,
          t2.s_acctbal AS s_acctbal,
          t2.s_comment AS s_comment,
          t3.n_nationkey AS n_nationkey,
          t3.n_name AS n_name,
          t3.n_regionkey AS n_regionkey,
          t3.n_comment AS n_comment,
          t4.r_regionkey AS r_regionkey,
          t4.r_name AS r_name,
          t4.r_comment AS r_comment
        FROM "part" AS t0
        INNER JOIN "partsupp" AS t1
          ON (
            t0.p_partkey = t1.ps_partkey
          )
        INNER JOIN "supplier" AS t2
          ON (
            t2.s_suppkey = t1.ps_suppkey
          )
        INNER JOIN "nation" AS t3
          ON (
            t2.s_nationkey = t3.n_nationkey
          )
        INNER JOIN "region" AS t4
          ON (
            t3.n_regionkey = t4.r_regionkey
          )
      ) AS t9
      WHERE
        (
          t9.p_size = CAST(15 AS TINYINT)
        )
        AND t9.p_type LIKE '%BRASS'
        AND (
          t9.r_name = 'EUROPE'
        )
        AND (
          t9.ps_supplycost = (
            SELECT
              MIN(t11.ps_supplycost) AS "Min(ps_supplycost)"
            FROM (
              SELECT
                *
              FROM (
                SELECT
                  t1.ps_partkey AS ps_partkey,
                  t1.ps_suppkey AS ps_suppkey,
                  t1.ps_availqty AS ps_availqty,
                  t1.ps_supplycost AS ps_supplycost,
                  t1.ps_comment AS ps_comment,
                  t2.s_suppkey AS s_suppkey,
                  t2.s_name AS s_name,
                  t2.s_address AS s_address,
                  t2.s_nationkey AS s_nationkey,
                  t2.s_phone AS s_phone,
                  t2.s_acctbal AS s_acctbal,
                  t2.s_comment AS s_comment,
                  t3.n_nationkey AS n_nationkey,
                  t3.n_name AS n_name,
                  t3.n_regionkey AS n_regionkey,
                  t3.n_comment AS n_comment,
                  t4.r_regionkey AS r_regionkey,
                  t4.r_name AS r_name,
                  t4.r_comment AS r_comment
                FROM "partsupp" AS t1
                INNER JOIN "supplier" AS t2
                  ON (
                    t2.s_suppkey = t1.ps_suppkey
                  )
                INNER JOIN "nation" AS t3
                  ON (
                    t2.s_nationkey = t3.n_nationkey
                  )
                INNER JOIN "region" AS t4
                  ON (
                    t3.n_regionkey = t4.r_regionkey
                  )
              ) AS t10
              WHERE
                (
                  t10.r_name = 'EUROPE'
                ) AND (
                  t9.p_partkey = t10.ps_partkey
                )
            ) AS t11
          )
        )
    ) AS t13
  ) AS t14
  ORDER BY
    t14.s_acctbal DESC,
    t14.n_name ASC,
    t14.s_name ASC,
    t14.p_partkey ASC
) AS t15
LIMIT 100