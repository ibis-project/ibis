SELECT t0.*
FROM t t0
WHERE (lower(t0.`color`) LIKE '%de%') AND
      (locate('de', lower(t0.`color`)) - 1 >= 0) AND
      (regexp_like(lower(t0.`color`), '.*ge.*'))