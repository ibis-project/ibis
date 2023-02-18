WITH t0 AS (
  SELECT t2.*
  FROM t t2
  WHERE (lower(t2.`color`) LIKE '%de%') AND
        (locate('de', lower(t2.`color`)) - 1 >= 0)
)
SELECT *
FROM (
  SELECT *
  FROM t0
  WHERE regexp_like(lower(t0.`color`), '.*ge.*')
) t1