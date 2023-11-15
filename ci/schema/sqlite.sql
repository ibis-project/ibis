DROP TABLE IF EXISTS functional_alltypes;

CREATE TABLE functional_alltypes (
    id BIGINT,
    bool_col BOOLEAN,
    tinyint_col BIGINT,
    smallint_col BIGINT,
    int_col BIGINT,
    bigint_col BIGINT,
    float_col FLOAT,
    double_col REAL,
    date_string_col TEXT,
    string_col TEXT,
    timestamp_col TIMESTAMP,
    year BIGINT,
    month BIGINT,
    CHECK (bool_col IN (0, 1))
);

DROP TABLE IF EXISTS astronauts;

CREATE TABLE astronauts (
    "id" BIGINT,
    "number" BIGINT,
    "nationwide_number" BIGINT,
    "name" TEXT,
    "original_name" TEXT,
    "sex" TEXT,
    "year_of_birth" BIGINT,
    "nationality" TEXT,
    "military_civilian" TEXT,
    "selection" TEXT,
    "year_of_selection" BIGINT,
    "mission_number" BIGINT,
    "total_number_of_missions" BIGINT,
    "occupation" TEXT,
    "year_of_mission" BIGINT,
    "mission_title" TEXT,
    "ascend_shuttle" TEXT,
    "in_orbit" TEXT,
    "descend_shuttle" TEXT,
    "hours_mission" FLOAT,
    "total_hrs_sum" FLOAT,
    "field21" BIGINT,
    "eva_hrs_mission" FLOAT,
    "total_eva_hrs" FLOAT
);

DROP TABLE IF EXISTS awards_players;

CREATE TABLE awards_players (
    "playerID" TEXT,
    "awardID" TEXT,
    "yearID" BIGINT,
    "lgID" TEXT,
    tie TEXT,
    notes TEXT
);

DROP TABLE IF EXISTS batting;

CREATE TABLE batting (
    "playerID" TEXT,
    "yearID" BIGINT,
    stint BIGINT,
    "teamID" TEXT,
    "lgID" TEXT,
    "G" BIGINT,
    "AB" BIGINT,
    "R" BIGINT,
    "H" BIGINT,
    "X2B" BIGINT,
    "X3B" BIGINT,
    "HR" BIGINT,
    "RBI" BIGINT,
    "SB" BIGINT,
    "CS" BIGINT,
    "BB" BIGINT,
    "SO" BIGINT,
    "IBB" BIGINT,
    "HBP" BIGINT,
    "SH" BIGINT,
    "SF" BIGINT,
    "GIDP" BIGINT
);

DROP TABLE IF EXISTS diamonds;

CREATE TABLE diamonds (
    carat FLOAT,
    cut TEXT,
    color TEXT,
    clarity TEXT,
    depth FLOAT,
    "table" FLOAT,
    price BIGINT,
    x FLOAT,
    y FLOAT,
    z FLOAT
);

DROP TABLE IF EXISTS json_t;

CREATE TABLE json_t (js JSON);

INSERT INTO json_t VALUES
    ('{"a": [1,2,3,4], "b": 1}'),
    ('{"a":null,"b":2}'),
    ('{"a":"foo", "c":null}'),
    ('null'),
    ('[42,47,55]'),
    ('[]');

DROP TABLE IF EXISTS win;
CREATE TABLE win (g TEXT, x BIGINT NOT NULL, y BIGINT);
INSERT INTO win VALUES
    ('a', 0, 3),
    ('a', 1, 2),
    ('a', 2, 0),
    ('a', 3, 1),
    ('a', 4, 1);
