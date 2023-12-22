SELECT
  t6.on,
  t6.by,
  t6.on_right,
  t6.by_right,
  t6.val
FROM (
  SELECT
    t2.on,
    t2.by,
    t3.on AS on_right,
    t3.by AS by_right,
    t3.val
  FROM left AS t2
  LEFT OUTER JOIN right AS t3
    ON t2.by = t3.by
) AS t6
WHERE
  t6.on_right = (
    SELECT
      MAX(t4.on) AS "Max(on)"
    FROM (
      SELECT
        t1.on,
        t1.by,
        t1.val
      FROM right AS t1
      WHERE
        t1.by = t0.by AND t1.on <= t0.on
    ) AS t4
  )