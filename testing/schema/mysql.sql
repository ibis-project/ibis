DROP TABLE IF EXISTS diamonds;

CREATE TABLE diamonds (
    carat FLOAT,
    cut TEXT,
    color TEXT,
    clarity TEXT,
    depth FLOAT,
    `table` FLOAT,
    price BIGINT,
    x FLOAT,
    y FLOAT,
    z FLOAT
);

DROP TABLE IF EXISTS batting;

CREATE TABLE batting (
    `playerID` TEXT,
    `yearID` BIGINT,
    stint BIGINT,
    `teamID` TEXT,
    `lgID` TEXT,
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
    `IBB` BIGINT,
    `HBP` BIGINT,
    `SH` BIGINT,
    `SF` BIGINT,
    `GIDP` BIGINT
);

DROP TABLE IF EXISTS awards_players;

CREATE TABLE awards_players (
    `playerID` TEXT,
    `awardID` TEXT,
    `yearID` BIGINT,
    `lgID` TEXT,
    tie TEXT,
    notes TEXT
);

DROP TABLE IF EXISTS functional_alltypes;

CREATE TABLE functional_alltypes (
    `index` BIGINT,
    `Unnamed: 0` BIGINT,
    id INTEGER,
    bool_col BOOLEAN,
    tinyint_col TINYINT,
    smallint_col SMALLINT,
    int_col INTEGER,
    bigint_col BIGINT,
    float_col FLOAT,
    double_col DOUBLE,
    date_string_col TEXT,
    string_col TEXT,
    timestamp_col TIMESTAMP,
    year INTEGER,
    month INTEGER
);

CREATE INDEX `ix_functional_alltypes_index` ON functional_alltypes (`index`);
