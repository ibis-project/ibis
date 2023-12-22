SELECT
  t8.origin,
  COUNT(*) AS "CountStar()"
FROM (
  SELECT
    t2.dest,
    t2.origin,
    t2.arrdelay
  FROM (
    SELECT
      t0.dest,
      t0.origin,
      t0.arrdelay
    FROM airlines AS t0
    WHERE
      t0.dest IN ('ORD', 'JFK', 'SFO')
  ) AS t2
  SEMI JOIN (
    SELECT
      t3.dest,
      t3."Mean(arrdelay)"
    FROM (
      SELECT
        t1.dest,
        AVG(t1.arrdelay) AS "Mean(arrdelay)"
      FROM (
        SELECT
          t0.dest,
          t0.origin,
          t0.arrdelay
        FROM airlines AS t0
        WHERE
          t0.dest IN ('ORD', 'JFK', 'SFO')
      ) AS t1
      GROUP BY
        1
    ) AS t3
    ORDER BY
      t3."Mean(arrdelay)" DESC
    LIMIT 10
  ) AS t6
    ON t2.dest = t6.dest
) AS t8
GROUP BY
  1