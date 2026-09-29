DROP TABLE IF EXISTS diamonds;

CREATE TABLE diamonds (
    carat FLOAT,
    cut STRING,
    color STRING,
    clarity STRING,
    depth FLOAT,
    `table` FLOAT,
    price BIGINT,
    x FLOAT,
    y FLOAT,
    z FLOAT
)
ENGINE=OLAP
DISTRIBUTED BY RANDOM
PROPERTIES ("replication_num" = "1");

DROP TABLE IF EXISTS astronauts;

CREATE TABLE astronauts (
    `id` BIGINT,
    `number` BIGINT,
    `nationwide_number` BIGINT,
    `name` STRING,
    `original_name` STRING,
    `sex` STRING,
    `year_of_birth` BIGINT,
    `nationality` STRING,
    `military_civilian` STRING,
    `selection` STRING,
    `year_of_selection` BIGINT,
    `mission_number` BIGINT,
    `total_number_of_missions` BIGINT,
    `occupation` STRING,
    `year_of_mission` BIGINT,
    `mission_title` STRING,
    `ascend_shuttle` STRING,
    `in_orbit` STRING,
    `descend_shuttle` STRING,
    `hours_mission` FLOAT,
    `total_hrs_sum` FLOAT,
    `field21` BIGINT,
    `eva_hrs_mission` FLOAT,
    `total_eva_hrs` FLOAT
)
ENGINE=OLAP
DISTRIBUTED BY RANDOM
PROPERTIES ("replication_num" = "1");

DROP TABLE IF EXISTS batting;

CREATE TABLE batting (
    `playerID` STRING,
    `yearID` BIGINT,
    stint BIGINT,
    `teamID` STRING,
    `lgID` STRING,
    `G` BIGINT,
    `AB` BIGINT,
    `R` BIGINT,
    `H` BIGINT,
    `X2B` BIGINT,
    `X3B` BIGINT,
    `HR` BIGINT,
    `RBI` BIGINT,
    `SB` BIGINT,
    `CS` BIGINT,
    `BB` BIGINT,
    `SO` BIGINT,
    `IBB` BIGINT NULL,
    `HBP` BIGINT NULL,
    `SH` BIGINT NULL,
    `SF` BIGINT NULL,
    `GIDP` BIGINT NULL
)
ENGINE=OLAP
DISTRIBUTED BY RANDOM
PROPERTIES ("replication_num" = "1");

DROP TABLE IF EXISTS awards_players;

CREATE TABLE awards_players (
    `playerID` STRING,
    `awardID` STRING,
    `yearID` BIGINT,
    `lgID` STRING,
    tie STRING,
    notes STRING
)
ENGINE=OLAP
DISTRIBUTED BY RANDOM
PROPERTIES ("replication_num" = "1");

DROP TABLE IF EXISTS functional_alltypes;

CREATE TABLE functional_alltypes (
    id INTEGER,
    bool_col BOOLEAN,
    tinyint_col TINYINT,
    smallint_col SMALLINT,
    int_col INTEGER,
    bigint_col BIGINT,
    float_col FLOAT,
    double_col DOUBLE,
    date_string_col STRING,
    string_col STRING,
    timestamp_col DATETIME,
    year INTEGER,
    month INTEGER
)
ENGINE=OLAP
DISTRIBUTED BY RANDOM
PROPERTIES ("replication_num" = "1");

DROP TABLE IF EXISTS json_t;

CREATE TABLE IF NOT EXISTS json_t (
    rowid BIGINT,
    js JSON
)
ENGINE=OLAP
DISTRIBUTED BY RANDOM
PROPERTIES ("replication_num" = "1");

INSERT INTO json_t VALUES
    (1, parse_json('{"a": [1,2,3,4], "b": 1}')),
    (2, parse_json('{"a":null,"b":2}')),
    (3, parse_json('{"a":"foo", "c":null}')),
    (4, parse_json('null')),
    (5, parse_json('[42,47,55]')),
    (6, parse_json('[]')),
    (7, parse_json('"a"')),
    (8, parse_json('""')),
    (9, parse_json('"b"')),
    (10, NULL),
    (11, parse_json('true')),
    (12, parse_json('false')),
    (13, parse_json('42')),
    (14, parse_json('37.37'));

DROP TABLE IF EXISTS win;

CREATE TABLE win (
    g STRING,
    x BIGINT NOT NULL,
    y BIGINT
)
ENGINE=OLAP
DISTRIBUTED BY RANDOM
PROPERTIES ("replication_num" = "1");

INSERT INTO win VALUES
    ('a', 0, 3),
    ('a', 1, 2),
    ('a', 2, 0),
    ('a', 3, 1),
    ('a', 4, 1);

DROP TABLE IF EXISTS topk;

CREATE TABLE topk (
    x BIGINT
)
ENGINE=OLAP
DISTRIBUTED BY RANDOM
PROPERTIES ("replication_num" = "1");

INSERT INTO topk VALUES (1), (1), (NULL);
