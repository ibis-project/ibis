SELECT
  *
FROM (
  SELECT
    t3.c_count,
    COUNT(*) AS custdist
  FROM (
    SELECT
      t2.c_custkey,
      COUNT(t2.o_orderkey) AS c_count
    FROM (
      SELECT
        t0.*,
        t1.*
      FROM "customer" AS t0
      LEFT JOIN "orders" AS t1
        ON (
          t0.c_custkey = t1.o_custkey
        ) AND NOT t1.o_comment LIKE '%special%requests%'
    ) AS t2
    GROUP BY
      1
  ) AS t3
  GROUP BY
    1
) AS t4
ORDER BY
  t4.custdist DESC,
  t4.c_count DESC