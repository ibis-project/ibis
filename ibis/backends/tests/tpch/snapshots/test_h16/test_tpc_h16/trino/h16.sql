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
  FROM hive.ibis_sf1.partsupp AS t1
  JOIN hive.ibis_sf1.part AS t2
    ON t2.p_partkey = t1.ps_partkey
  WHERE
    t2.p_brand <> 'Brand#45'
    AND NOT t2.p_type LIKE 'MEDIUM POLISHED%'
    AND t2.p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
    AND (
      NOT t1.ps_suppkey IN (
        SELECT
          t3.s_suppkey
        FROM hive.ibis_sf1.supplier AS t3
        WHERE
          t3.s_comment LIKE '%Customer%Complaints%'
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