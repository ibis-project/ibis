SELECT *
FROM (
  SELECT *
  FROM (
    SELECT *
    FROM t
    WHERE (lower(`color`) LIKE '%de%') AND
          (locate('de', lower(`color`)) - 1 >= 0)
  ) t1
  WHERE regexp_like(lower(`color`), '.*ge.*')
) t0
