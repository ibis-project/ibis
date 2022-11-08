SELECT *, (row_number() OVER (ORDER BY `f`) - 1) / 2 AS `new`
FROM alltypes