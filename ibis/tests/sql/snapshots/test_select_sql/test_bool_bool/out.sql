SELECT *
FROM airlines
WHERE (CAST(`dest` AS bigint) = 0) = TRUE