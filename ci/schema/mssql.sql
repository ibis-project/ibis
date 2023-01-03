DROP TABLE IF EXISTS diamonds;

CREATE TABLE diamonds (
    carat FLOAT,
    cut VARCHAR(MAX),
    color VARCHAR(MAX),
    clarity VARCHAR(MAX),
    depth FLOAT,
    "table" FLOAT,
    price BIGINT,
    x FLOAT,
    y FLOAT,
    z FLOAT
);

DROP TABLE IF EXISTS batting;

CREATE TABLE batting (
    "playerID" VARCHAR(MAX),
    "yearID" BIGINT,
    stint BIGINT,
    "teamID" VARCHAR(MAX),
    "lgID" VARCHAR(MAX),
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

DROP TABLE IF EXISTS awards_players;

CREATE TABLE awards_players (
    "playerID" VARCHAR(MAX),
    "awardID" VARCHAR(MAX),
    "yearID" BIGINT,
    "lgID" VARCHAR(MAX),
    tie VARCHAR(MAX),
    notes VARCHAR(MAX)
);

DROP TABLE IF EXISTS functional_alltypes;

CREATE TABLE functional_alltypes (
    "index" BIGINT,
    "Unnamed: 0" BIGINT,
    id INTEGER,
    bool_col BIT,
    tinyint_col SMALLINT,
    smallint_col SMALLINT,
    int_col INTEGER,
    bigint_col BIGINT,
    float_col REAL,
    double_col DOUBLE PRECISION,
    date_string_col VARCHAR(MAX),
    string_col VARCHAR(MAX),
    timestamp_col DATETIME2,
    year INTEGER,
    month INTEGER
);

CREATE INDEX "ix_functional_alltypes_index" ON functional_alltypes ("index");

DROP TABLE IF EXISTS win;

CREATE TABLE win (g VARCHAR(MAX), x BIGINT, y BIGINT);
INSERT INTO win VALUES
    ('a', 0, 3),
    ('a', 1, 2),
    ('a', 2, 0),
    ('a', 3, 1),
    ('a', 4, 1);
