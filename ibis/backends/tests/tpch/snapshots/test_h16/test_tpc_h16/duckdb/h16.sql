SELECT
  t9.p_brand,
  t9.p_type,
  t9.p_size,
  t9.supplier_cnt
FROM (
  SELECT
    t8.p_brand,
    t8.p_type,
    t8.p_size,
    COUNT(DISTINCT t8.ps_suppkey) AS supplier_cnt
  FROM (
    SELECT
      t7.ps_partkey,
      t7.ps_suppkey,
      t7.ps_availqty,
      t7.ps_supplycost,
      t7.ps_comment,
      t7.p_partkey,
      t7.p_name,
      t7.p_mfgr,
      t7.p_brand,
      t7.p_type,
      t7.p_size,
      t7.p_container,
      t7.p_retailprice,
      t7.p_comment
    FROM (
      SELECT
        t3.ps_partkey,
        t3.ps_suppkey,
        t3.ps_availqty,
        t3.ps_supplycost,
        t3.ps_comment,
        t4.p_partkey,
        t4.p_name,
        t4.p_mfgr,
        t4.p_brand,
        t4.p_type,
        t4.p_size,
        t4.p_container,
        t4.p_retailprice,
        t4.p_comment
      FROM partsupp AS t3
      INNER JOIN part AS t4
        ON t4.p_partkey = t3.ps_partkey
    ) AS t7
    WHERE
      t7.p_brand <> 'Brand#45'
      AND NOT (
        t7.p_type LIKE 'MEDIUM POLISHED%'
      )
      AND t7.p_size IN (CAST(49 AS TINYINT), CAST(14 AS TINYINT), CAST(23 AS TINYINT), CAST(45 AS TINYINT), CAST(19 AS TINYINT), CAST(3 AS TINYINT), CAST(36 AS TINYINT), CAST(9 AS TINYINT))
      AND NOT (
        t7.ps_suppkey IN (
          SELECT
            t2.s_suppkey
          FROM supplier AS t2
          WHERE
            t2.s_comment LIKE '%Customer%Complaints%'
        )
      )
  ) AS t8
  GROUP BY
    1,
    2,
    3
) AS t9
ORDER BY
  t9.supplier_cnt DESC,
  t9.p_brand ASC,
  t9.p_type ASC,
  t9.p_size ASC