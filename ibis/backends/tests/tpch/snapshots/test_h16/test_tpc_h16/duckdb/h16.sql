SELECT
  *
FROM (
  SELECT
    t7.p_brand AS p_brand,
    t7.p_type AS p_type,
    t7.p_size AS p_size,
    COUNT(DISTINCT t7.ps_suppkey) AS supplier_cnt
  FROM (
    SELECT
      *
    FROM (
      SELECT
        t0.ps_partkey AS ps_partkey,
        t0.ps_suppkey AS ps_suppkey,
        t0.ps_availqty AS ps_availqty,
        t0.ps_supplycost AS ps_supplycost,
        t0.ps_comment AS ps_comment,
        t1.p_partkey AS p_partkey,
        t1.p_name AS p_name,
        t1.p_mfgr AS p_mfgr,
        t1.p_brand AS p_brand,
        t1.p_type AS p_type,
        t1.p_size AS p_size,
        t1.p_container AS p_container,
        t1.p_retailprice AS p_retailprice,
        t1.p_comment AS p_comment
      FROM "partsupp" AS t0
      INNER JOIN "part" AS t1
        ON (
          t1.p_partkey = t0.ps_partkey
        )
    ) AS t5
    WHERE
      (
        t5.p_brand <> 'Brand#45'
      )
      AND NOT t5.p_type LIKE 'MEDIUM POLISHED%'
      AND t5.p_size IN (CAST(49 AS TINYINT), CAST(14 AS TINYINT), CAST(23 AS TINYINT), CAST(45 AS TINYINT), CAST(19 AS TINYINT), CAST(3 AS TINYINT), CAST(36 AS TINYINT), CAST(9 AS TINYINT))
      AND NOT t5.ps_suppkey IN ((
        SELECT
          t4.s_suppkey AS s_suppkey
        FROM (
          SELECT
            *
          FROM "supplier" AS t2
          WHERE
            t2.s_comment LIKE '%Customer%Complaints%'
        ) AS t4
      ))
  ) AS t7
  GROUP BY
    1,
    2,
    3
) AS t8
ORDER BY
  t8.supplier_cnt DESC,
  t8.p_brand ASC,
  t8.p_type ASC,
  t8.p_size ASC