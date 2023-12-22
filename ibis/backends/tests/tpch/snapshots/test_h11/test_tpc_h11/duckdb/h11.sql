SELECT
  t10.ps_partkey,
  t10.value
FROM (
  SELECT
    t9.ps_partkey,
    SUM(t9.ps_supplycost * t9.ps_availqty) AS value
  FROM (
    SELECT
      t8.ps_partkey,
      t8.ps_suppkey,
      t8.ps_availqty,
      t8.ps_supplycost,
      t8.ps_comment,
      t8.s_suppkey,
      t8.s_name,
      t8.s_address,
      t8.s_nationkey,
      t8.s_phone,
      t8.s_acctbal,
      t8.s_comment,
      t8.n_nationkey,
      t8.n_name,
      t8.n_regionkey,
      t8.n_comment
    FROM (
      SELECT
        t3.ps_partkey,
        t3.ps_suppkey,
        t3.ps_availqty,
        t3.ps_supplycost,
        t3.ps_comment,
        t4.s_suppkey,
        t4.s_name,
        t4.s_address,
        t4.s_nationkey,
        t4.s_phone,
        t4.s_acctbal,
        t4.s_comment,
        t5.n_nationkey,
        t5.n_name,
        t5.n_regionkey,
        t5.n_comment
      FROM partsupp AS t3
      INNER JOIN supplier AS t4
        ON t3.ps_suppkey = t4.s_suppkey
      INNER JOIN nation AS t5
        ON t5.n_nationkey = t4.s_nationkey
    ) AS t8
    WHERE
      t8.n_name = 'GERMANY'
  ) AS t9
  GROUP BY
    1
) AS t10
WHERE
  t10.value > (
    (
      SELECT
        SUM(t9.ps_supplycost * t9.ps_availqty) AS "Sum(Multiply(ps_supplycost, ps_availqty))"
      FROM (
        SELECT
          t8.ps_partkey,
          t8.ps_suppkey,
          t8.ps_availqty,
          t8.ps_supplycost,
          t8.ps_comment,
          t8.s_suppkey,
          t8.s_name,
          t8.s_address,
          t8.s_nationkey,
          t8.s_phone,
          t8.s_acctbal,
          t8.s_comment,
          t8.n_nationkey,
          t8.n_name,
          t8.n_regionkey,
          t8.n_comment
        FROM (
          SELECT
            t3.ps_partkey,
            t3.ps_suppkey,
            t3.ps_availqty,
            t3.ps_supplycost,
            t3.ps_comment,
            t4.s_suppkey,
            t4.s_name,
            t4.s_address,
            t4.s_nationkey,
            t4.s_phone,
            t4.s_acctbal,
            t4.s_comment,
            t5.n_nationkey,
            t5.n_name,
            t5.n_regionkey,
            t5.n_comment
          FROM partsupp AS t3
          INNER JOIN supplier AS t4
            ON t3.ps_suppkey = t4.s_suppkey
          INNER JOIN nation AS t5
            ON t5.n_nationkey = t4.s_nationkey
        ) AS t8
        WHERE
          t8.n_name = 'GERMANY'
      ) AS t9
    ) * CAST(0.0001 AS DOUBLE)
  )
ORDER BY
  t10.value DESC