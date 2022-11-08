SELECT `g`, (rank() OVER (ORDER BY `f`) - 1) AS `minr`,
       (dense_rank() OVER (ORDER BY `f`) - 1) AS `denser`
FROM alltypes