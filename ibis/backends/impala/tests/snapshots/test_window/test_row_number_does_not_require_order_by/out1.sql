SELECT *, (row_number() OVER (PARTITION BY `g`) - 1) AS `foo`
FROM alltypes