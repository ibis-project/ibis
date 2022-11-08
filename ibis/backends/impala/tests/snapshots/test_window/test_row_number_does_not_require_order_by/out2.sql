SELECT *, (row_number() OVER (PARTITION BY `g` ORDER BY `f`) - 1) AS `foo`
FROM alltypes