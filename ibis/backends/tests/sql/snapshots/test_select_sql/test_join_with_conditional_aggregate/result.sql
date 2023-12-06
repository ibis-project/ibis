SELECT
  t0.on AS on,
  t0.by AS by,
  t1.on AS on_right,
  t1.by AS by_right,
  t1.val AS val
FROM left AS t0
LEFT OUTER JOIN right AS t1
  ON t0.by = t1.by
WHERE
  (
    t1.on_right = (
      SELECT
        MAX(t1.on) AS "Max(on)"
      FROM right AS t1
      WHERE
        (
          t1.by = t0.by
        ) AND (
          t1.on <= t0.on
        )
    )
  )
