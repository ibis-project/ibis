WITH t0 AS (
  SELECT
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
    t3.n_comment AS n_comment
  FROM hive.ibis_sf1.supplier AS t2
  JOIN hive.ibis_sf1.nation AS t3
    ON t2.s_nationkey = t3.n_nationkey
  WHERE
    t3.n_name = 'CANADA'
    AND t2.s_suppkey IN (
      SELECT
        t4.ps_suppkey
      FROM (
        SELECT
          t5.ps_partkey AS ps_partkey,
          t5.ps_suppkey AS ps_suppkey,
          t5.ps_availqty AS ps_availqty,
          t5.ps_supplycost AS ps_supplycost,
          t5.ps_comment AS ps_comment
        FROM hive.ibis_sf1.partsupp AS t5
        WHERE
          t5.ps_partkey IN (
            SELECT
              t6.p_partkey
            FROM (
              SELECT
                t7.p_partkey AS p_partkey,
                t7.p_name AS p_name,
                t7.p_mfgr AS p_mfgr,
                t7.p_brand AS p_brand,
                t7.p_type AS p_type,
                t7.p_size AS p_size,
                t7.p_container AS p_container,
                t7.p_retailprice AS p_retailprice,
                t7.p_comment AS p_comment
              FROM hive.ibis_sf1.part AS t7
              WHERE
                t7.p_name LIKE 'forest%'
            ) AS t6
          )
          AND t5.ps_availqty > (
            SELECT
              SUM(t6.l_quantity) AS "Sum(l_quantity)"
            FROM hive.ibis_sf1.lineitem AS t6
            WHERE
              t6.l_partkey = t5.ps_partkey
              AND t6.l_suppkey = t5.ps_suppkey
              AND t6.l_shipdate >= FROM_ISO8601_DATE('1994-01-01')
              AND t6.l_shipdate < FROM_ISO8601_DATE('1995-01-01')
          ) * 0.5
      ) AS t4
    )
)
SELECT
  t1.s_name,
  t1.s_address
FROM (
  SELECT
    t0.s_name AS s_name,
    t0.s_address AS s_address
  FROM t0
) AS t1
ORDER BY
  t1.s_name ASC