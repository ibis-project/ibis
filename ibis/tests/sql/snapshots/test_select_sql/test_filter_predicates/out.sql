SELECT *
FROM (
  SELECT *
  FROM (
    SELECT t2.*
    FROM t t2
    WHERE (lower(t2.`color`) LIKE '%de%') AND
          (locate('de', lower(t2.`color`)) - 1 >= 0)
  ) t1
  WHERE regexp_like(lower(t1.`color`), '.*ge.*')
) t0