SELECT
  *
FROM (
  SELECT
    t3.p_brand,
    t3.p_type,
    t3.p_size,
    COUNT(DISTINCT t3.ps_suppkey) AS supplier_cnt
  FROM (
    SELECT
      t0.*,
      t1.*
    FROM "partsupp" AS t0
    INNER JOIN "part" AS t1
      ON (
        t1.p_partkey = t0.ps_partkey
      )
  ) AS t3
  WHERE
    (
      t3.p_brand <> 'Brand#45'
    )
    AND NOT t3.p_type LIKE 'MEDIUM POLISHED%'
    AND t3.p_size IN (CAST(49 AS TINYINT), CAST(14 AS TINYINT), CAST(23 AS TINYINT), CAST(45 AS TINYINT), CAST(19 AS TINYINT), CAST(3 AS TINYINT), CAST(36 AS TINYINT), CAST(9 AS TINYINT))
    AND NOT t3.ps_suppkey IN (
      SELECT
        t4.s_suppkey
      FROM (
        SELECT
          *
        FROM "supplier" AS t2
        WHERE
          t2.s_comment LIKE '%Customer%Complaints%'
      ) AS t4
    )
  GROUP BY
    1,
    2,
    3
) AS t6
ORDER BY
  t6.supplier_cnt DESC,
  t6.p_brand ASC,
  t6.p_type ASC,
  t6.p_size ASC