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
) -- mssql equivalent? -- DEFAULT CHARACTER SET = utf8;

DROP TABLE IF EXISTS batting;

CREATE TABLE batting (
    `playerID` VARCHAR(255),
    `yearID` BIGINT,
    stint BIGINT,
    `teamID` VARCHAR(7),
    `lgID` VARCHAR(7),
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
) -- mssql equivalent? -- DEFAULT CHARACTER SET = utf8;

DROP TABLE IF EXISTS awards_players;

CREATE TABLE awards_players (
    `playerID` VARCHAR(255),
    `awardID` VARCHAR(255),
    `yearID` BIGINT,
    `lgID` VARCHAR(7),
    tie VARCHAR(7),
    notes VARCHAR(255)
) -- DEFAULT CHARACTER SET = utf8;

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
) -- mssql equivalent? -- DEFAULT CHARACTER SET = utf8;

CREATE INDEX `ix_functional_alltypes_index` ON functional_alltypes (`index`);
