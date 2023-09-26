WITH t0 AS (
  SELECT
    t3.l_orderkey AS l1_orderkey,
    t4.o_orderstatus AS o_orderstatus,
    t3.l_receiptdate AS l_receiptdate,
    t3.l_commitdate AS l_commitdate,
    t3.l_suppkey AS l1_suppkey,
    t2.s_name AS s_name,
    t5.n_name AS n_name
  FROM main.supplier AS t2
  JOIN main.lineitem AS t3
    ON t2.s_suppkey = t3.l_suppkey
  JOIN main.orders AS t4
    ON t4.o_orderkey = t3.l_orderkey
  JOIN main.nation AS t5
    ON t2.s_nationkey = t5.n_nationkey
)
SELECT
  t1.s_name,
  t1.numwait
FROM (
  SELECT
    t0.s_name AS s_name,
    COUNT(*) AS numwait
  FROM t0
  WHERE
    t0.o_orderstatus = 'F'
    AND t0.l_receiptdate > t0.l_commitdate
    AND t0.n_name = 'SAUDI ARABIA'
    AND (
      EXISTS(
        SELECT
          CAST(1 AS TINYINT) AS anon_1
        FROM main.lineitem AS t2
        WHERE
          t2.l_orderkey = t0.l1_orderkey AND t2.l_suppkey <> t0.l1_suppkey
      )
    )
    AND NOT (
      EXISTS(
        SELECT
          CAST(1 AS TINYINT) AS anon_2
        FROM main.lineitem AS t2
        WHERE
          t2.l_orderkey = t0.l1_orderkey
          AND t2.l_suppkey <> t0.l1_suppkey
          AND t2.l_receiptdate > t2.l_commitdate
      )
    )
  GROUP BY
    1
) AS t1
ORDER BY
  t1.numwait DESC,
  t1.s_name ASC
LIMIT 100