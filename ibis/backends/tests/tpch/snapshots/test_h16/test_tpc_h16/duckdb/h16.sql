SELECT
  t8.p_brand AS p_brand,
  t8.p_type AS p_type,
  t8.p_size AS p_size,
  t8.supplier_cnt AS supplier_cnt
FROM (
  SELECT
    t7.p_brand AS p_brand,
    t7.p_type AS p_type,
    t7.p_size AS p_size,
    COUNT(DISTINCT t7.ps_suppkey) AS supplier_cnt
  FROM (
    SELECT
      t6.ps_partkey AS ps_partkey,
      t6.ps_suppkey AS ps_suppkey,
      t6.ps_availqty AS ps_availqty,
      t6.ps_supplycost AS ps_supplycost,
      t6.ps_comment AS ps_comment,
      t6.p_partkey AS p_partkey,
      t6.p_name AS p_name,
      t6.p_mfgr AS p_mfgr,
      t6.p_brand AS p_brand,
      t6.p_type AS p_type,
      t6.p_size AS p_size,
      t6.p_container AS p_container,
      t6.p_retailprice AS p_retailprice,
      t6.p_comment AS p_comment
    FROM (
      SELECT
        t0.ps_partkey AS ps_partkey,
        t0.ps_suppkey AS ps_suppkey,
        t0.ps_availqty AS ps_availqty,
        t0.ps_supplycost AS ps_supplycost,
        t0.ps_comment AS ps_comment,
        t3.p_partkey AS p_partkey,
        t3.p_name AS p_name,
        t3.p_mfgr AS p_mfgr,
        t3.p_brand AS p_brand,
        t3.p_type AS p_type,
        t3.p_size AS p_size,
        t3.p_container AS p_container,
        t3.p_retailprice AS p_retailprice,
        t3.p_comment AS p_comment
      FROM partsupp AS t0
      INNER JOIN part AS t3
        ON t3.p_partkey = t0.ps_partkey
    ) AS t6
    WHERE
      t6.p_brand <> 'Brand#45'
      AND NOT (
        t6.p_type LIKE 'MEDIUM POLISHED%'
      )
      AND t6.p_size IN (CAST(49 AS TINYINT), CAST(14 AS TINYINT), CAST(23 AS TINYINT), CAST(45 AS TINYINT), CAST(19 AS TINYINT), CAST(3 AS TINYINT), CAST(36 AS TINYINT), CAST(9 AS TINYINT))
      AND NOT (
        t6.ps_suppkey IN ((
          SELECT
            t2.s_suppkey AS s_suppkey
          FROM supplier AS t2
          WHERE
            t2.s_comment LIKE '%Customer%Complaints%'
        ))
      )
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