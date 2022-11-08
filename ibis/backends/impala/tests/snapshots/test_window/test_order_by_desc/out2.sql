SELECT lag(`d`) OVER (PARTITION BY `g` ORDER BY `f` DESC) AS `foo`,
       max(`a`) OVER (PARTITION BY `g` ORDER BY `f` DESC) AS `max`
FROM alltypes