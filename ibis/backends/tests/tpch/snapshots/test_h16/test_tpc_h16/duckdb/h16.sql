SELECT
  t0.p_brand,
  t0.p_type,
  t0.p_size,
  t0.supplier_cnt
FROM (
  SELECT
    t2.p_brand AS p_brand,
    t2.p_type AS p_type,
    t2.p_size AS p_size,
    COUNT(DISTINCT t1.ps_suppkey) AS supplier_cnt
  FROM main.partsupp AS t1
  JOIN main.part AS t2
    ON t2.p_partkey = t1.ps_partkey
  WHERE
    t2.p_brand <> 'Brand#45'
    AND NOT t2.p_type LIKE 'MEDIUM POLISHED%'
    AND t2.p_size IN (CAST(49 AS TINYINT), CAST(14 AS TINYINT), CAST(23 AS TINYINT), CAST(45 AS TINYINT), CAST(19 AS TINYINT), CAST(3 AS TINYINT), CAST(36 AS TINYINT), CAST(9 AS TINYINT))
    AND (
      NOT t1.ps_suppkey IN (
        SELECT
          t3.s_suppkey
        FROM (
          SELECT
            t4.s_suppkey AS s_suppkey,
            t4.s_name AS s_name,
            t4.s_address AS s_address,
            t4.s_nationkey AS s_nationkey,
            t4.s_phone AS s_phone,
            t4.s_acctbal AS s_acctbal,
            t4.s_comment AS s_comment
          FROM main.supplier AS t4
          WHERE
            t4.s_comment LIKE '%Customer%Complaints%'
        ) AS t3
      )
    )
  GROUP BY
    1,
    2,
    3
) AS t0
ORDER BY
  t0.supplier_cnt DESC,
  t0.p_brand ASC,
  t0.p_type ASC,
  t0.p_size ASC