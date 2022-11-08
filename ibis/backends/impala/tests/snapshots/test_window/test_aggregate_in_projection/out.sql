SELECT *, `f` / sum(`f`) OVER () AS `normed_f`
FROM alltypes