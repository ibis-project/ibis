SELECT
  t15.ps_partkey AS ps_partkey,
  t15.value AS value
FROM (
  SELECT
    t13.ps_partkey AS ps_partkey,
    SUM(t13.ps_supplycost * t13.ps_availqty) AS value
  FROM (
    SELECT
      t11.ps_partkey AS ps_partkey,
      t11.ps_suppkey AS ps_suppkey,
      t11.ps_availqty AS ps_availqty,
      t11.ps_supplycost AS ps_supplycost,
      t11.ps_comment AS ps_comment,
      t11.s_suppkey AS s_suppkey,
      t11.s_name AS s_name,
      t11.s_address AS s_address,
      t11.s_nationkey AS s_nationkey,
      t11.s_phone AS s_phone,
      t11.s_acctbal AS s_acctbal,
      t11.s_comment AS s_comment,
      t11.n_nationkey AS n_nationkey,
      t11.n_name AS n_name,
      t11.n_regionkey AS n_regionkey,
      t11.n_comment AS n_comment
    FROM (
      SELECT
        t0.ps_partkey AS ps_partkey,
        t0.ps_suppkey AS ps_suppkey,
        t0.ps_availqty AS ps_availqty,
        t0.ps_supplycost AS ps_supplycost,
        t0.ps_comment AS ps_comment,
        t3.s_suppkey AS s_suppkey,
        t3.s_name AS s_name,
        t3.s_address AS s_address,
        t3.s_nationkey AS s_nationkey,
        t3.s_phone AS s_phone,
        t3.s_acctbal AS s_acctbal,
        t3.s_comment AS s_comment,
        t5.n_nationkey AS n_nationkey,
        t5.n_name AS n_name,
        t5.n_regionkey AS n_regionkey,
        t5.n_comment AS n_comment
      FROM partsupp AS t0
      INNER JOIN supplier AS t3
        ON t0.ps_suppkey = t3.s_suppkey
      INNER JOIN nation AS t5
        ON t5.n_nationkey = t3.s_nationkey
    ) AS t11
    WHERE
      t11.n_name = 'GERMANY'
  ) AS t13
  GROUP BY
    1
) AS t15
WHERE
  t15.value > (
    (
      SELECT
        SUM(t14.ps_supplycost * t14.ps_availqty) AS "Sum(Multiply(ps_supplycost, ps_availqty))"
      FROM (
        SELECT
          t12.ps_partkey AS ps_partkey,
          t12.ps_suppkey AS ps_suppkey,
          t12.ps_availqty AS ps_availqty,
          t12.ps_supplycost AS ps_supplycost,
          t12.ps_comment AS ps_comment,
          t12.s_suppkey AS s_suppkey,
          t12.s_name AS s_name,
          t12.s_address AS s_address,
          t12.s_nationkey AS s_nationkey,
          t12.s_phone AS s_phone,
          t12.s_acctbal AS s_acctbal,
          t12.s_comment AS s_comment,
          t12.n_nationkey AS n_nationkey,
          t12.n_name AS n_name,
          t12.n_regionkey AS n_regionkey,
          t12.n_comment AS n_comment
        FROM (
          SELECT
            t0.ps_partkey AS ps_partkey,
            t0.ps_suppkey AS ps_suppkey,
            t0.ps_availqty AS ps_availqty,
            t0.ps_supplycost AS ps_supplycost,
            t0.ps_comment AS ps_comment,
            t4.s_suppkey AS s_suppkey,
            t4.s_name AS s_name,
            t4.s_address AS s_address,
            t4.s_nationkey AS s_nationkey,
            t4.s_phone AS s_phone,
            t4.s_acctbal AS s_acctbal,
            t4.s_comment AS s_comment,
            t6.n_nationkey AS n_nationkey,
            t6.n_name AS n_name,
            t6.n_regionkey AS n_regionkey,
            t6.n_comment AS n_comment
          FROM partsupp AS t0
          INNER JOIN supplier AS t4
            ON t0.ps_suppkey = t4.s_suppkey
          INNER JOIN nation AS t6
            ON t6.n_nationkey = t4.s_nationkey
        ) AS t12
        WHERE
          t12.n_name = 'GERMANY'
      ) AS t14
    ) * CAST(0.0001 AS DOUBLE)
  )
ORDER BY
  t15.value DESC
