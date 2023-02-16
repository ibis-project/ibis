SELECT *
FROM (
  WITH t2 AS (
    SELECT t2.*
    FROM t t2
    WHERE (lower(t2.`color`) LIKE '%de%') AND
          (locate('de', lower(t2.`color`)) - 1 >= 0)
  )
  SELECT *
  FROM (
    SELECT t2.*
    FROM t t2
    WHERE (lower(t2.`color`) LIKE '%de%') AND
          (locate('de', lower(t2.`color`)) - 1 >= 0)
  ) t2
  WHERE regexp_like(lower(t2.`color`), '.*ge.*')
) t0