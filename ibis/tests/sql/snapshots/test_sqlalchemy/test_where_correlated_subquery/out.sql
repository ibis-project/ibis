SELECT
  t0.job,
  t0.dept_id,
  t0.year,
  t0.y
FROM foo AS t0
WHERE
  t0.y > (
    SELECT
      AVG(t1.y) AS "Mean(y)"
    FROM foo AS t1
    WHERE
      t0.dept_id = t1.dept_id
  )