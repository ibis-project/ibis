SELECT
  t6.origin AS origin,
  COUNT(*) AS "CountStar()"
FROM (
  SELECT
    t1.dest AS dest,
    t1.origin AS origin,
    t1.arrdelay AS arrdelay
  FROM (
    SELECT
      *
    FROM airlines AS t0
    WHERE
      t0.dest IN ('ORD', 'JFK', 'SFO')
  ) AS t1
  SEMI JOIN (
    SELECT
      *
    FROM (
      SELECT
        t1.dest AS dest,
        AVG(t1.arrdelay) AS "Mean(arrdelay)"
      FROM (
        SELECT
          *
        FROM airlines AS t0
        WHERE
          t0.dest IN ('ORD', 'JFK', 'SFO')
      ) AS t1
      GROUP BY
        1
    ) AS t2
    ORDER BY
      t2."Mean(arrdelay)" DESC
    LIMIT 10
  ) AS t4
    ON t1.dest = t4.dest
) AS t6
GROUP BY
  1