.. _omnisci:

*************************
Using Ibis with OmniSciDB
*************************

To get started with the OmniSci client, check the :ref:`OmniSci quick start <install.omniscidb>`.

For the API documentation, visit the :ref:`OmniSci API <api.omniscidb>`.

Backend internals
=================

In this document it would be explained the main aspects of `ibis` and
`omniscidb` backend implementation.

Modules
-------

omniscidb backend has 7 modules, the main three are:

- `api`
- `client`
- `compiler`

The `identifiers` and `operations` modules complement the `compiler` module.

api
---

`api` module is the central key to access the backend. Ensure to include
the follow code into `ibis.__init__`:

.. code-block:: python

    with suppress(ImportError):
        # pip install ibis-framework[omniscidb]
        import ibis.omniscidb.api as omniscidb

There are three functions in the `api` module:

- :func:`~ibis.omnisci.api.compile`
- :func:`~ibis.omnisci.api.connect`
- :func:`~ibis.omnisci.api.verify`

:func:`~ibis.omnisci.api.compile` method compiles an `ibis` expression into `SQL`:

.. code-block:: python

    t = omniscidb_cli.table('flights_2008_10k')
    proj = t['arrtime', 'arrdelay']
    print(ibis.omniscidb.compile(proj))

:func:`~ibis.omnisci.api.connect` method instantiates a :class:`~ibis.omnisci.api.OmniSciDBClient` object that connect to the specified
`omniscidb` database:

.. code-block:: python

    omniscidb_cli = ibis.omniscidb.connect(
        host='localhost', user='admin', password='HyperInteractive',
        port=6274, database='omnisci'
    )

:func:`~ibis.omnisci.api.verify` method checks if the `ibis` expression can be compiled:

.. code-block:: python

    t = omniscidb_cli.table('flights_2008_10k')
    proj = t['arrtime', 'arrdelay']
    assert ibis.omniscidb.verify(proj) == True

client
------

`client` module contains the main classes to handle the connection to an `omniscidb`
database.

The main classes are:

- :class:`~ibis.omnisci.api.OmniSciDBClient`
- `OmniSciDBQuery`
- `OmniSciDBDataType`
- `OmniSciDBDefaultCursor`

`OmniSciDBDataType` class is used to translate data type from `ibis` and to `ibis`.
Its main methods are:

- `parse`
- `to_ibis`
- `from_ibis`

:class:`~ibis.omnisci.api.OmniSciDBClient` class is used to connect to an `omniscidb` database and manipulate data
expressions. Its main methods are:

- `__init__`
- `_build_ast`
- `_execute`
- `_fully_qualified_name`
- `_get_table_schema`
- `_table_expr_klass`
- :func:`~ibis.omnisci.api.OmniSciDBClient.log`
- :func:`~ibis.omnisci.api.OmniSciDBClient.close`
- :func:`~ibis.omnisci.api.OmniSciDBClient.database`
- `current_database`
- :func:`~ibis.omnisci.api.OmniSciDBClient.set_database`
- `exists_database`
- `list_databases`
- :func:`~ibis.omnisci.api.OmniSciDBClient.exists_table`
- :func:`~ibis.omnisci.api.OmniSciDBClient.list_tables`
- :func:`~ibis.omnisci.api.OmniSciDBClient.get_schema`
- :func:`~ibis.omnisci.api.OmniSciDBClient.version`

`_build_ast` method is required.

`OmniSciDBQuery` class should define at least the `_fetch` method. If `Query`
class is used when the `OmniSciDBClient.execute` method is called, an exception
is raised.

    (...) once the data arrives from the database we need to convert that data
    to a pandas DataFrame.

    The Query class, with its _fetch() method, provides a way for ibis
    SQLClient objects to do any additional processing necessary after
    the database returns results to the client.
    (http://docs.ibis-project.org/design.html#execution)

Under the hood the `execute` method, uses a cursor class that will fetch the
result from the database and load it to a dataframe format (e.g. pandas, GeoPandas, cuDF).

compiler
--------

The main classes inside `compiler` module are:

- `OmniSciDBDialect`
- `OmniSciDBExprTranslator`
- `OmniSciDBQueryBuilder`
- `OmniSciDBSelect`
- `OmniSciDBSelectBuilder`
- `OmniSciDBTableSetFormatter`

operations
----------

    `Node` subclasses make up the core set of operations of ibis.
    Each node corresponds to a particular operation.
    Most nodes are defined in the `operations` module.
    (http://docs.ibis-project.org/design.html#the-node-class).


Creating new expression: To create new expressions it is necessary to do these
steps:

1. create a new class
2. create a new function and assign it to a DataType
3. create a compiler function to this new function and assign it to the compiler translator

A new Class database function would be like this (`my_backend_operations.py`):

.. code-block:: python

    class MyNewFunction(ops.UnaryOp):
        """My new class function"""
        output_type = rlz.shape_like('arg', 'float')

After creating the new class database function, the follow step is to create a
function and assign it to the DataTypes allowed to use it:

.. code-block:: python

    def my_new_function(numeric_value):
        return MyNewFunction(numeric_value).to_expr()


    NumericValue.sin = sin

Also, it is necessary to register the new function:

.. code-block:: python

    # if it necessary define the fixed_arity function
    def fixed_arity(func_name, arity):
        def formatter(translator, expr):
            op = expr.op()
            arg_count = len(op.args)
            if arity != arg_count:
                msg = 'Incorrect number of args {0} instead of {1}'
                raise com.UnsupportedOperationError(
                    msg.format(arg_count, arity)
                )
            return _call(translator, func_name, *op.args)
        return formatter

    _operation_registry.update({
        MyNewFunction: fixed_arity('my_new_function', 1)
    })

Now, it just needs a compiler function to translate the function to a SQL code
(my_backend/compiler.py):

.. code-block:: python

    compiles = MyBackendExprTranslator.compiles

    @compiles(MyNewFunction)
    def compile_my_new_function(translator, expr):
        # pull out the arguments to the expression
        arg, = expr.op().args

        # compile the argument
        compiled_arg = translator.translate(arg)
        return 'my_new_function(%s)' % compiled_arg


identifiers
-----------

`identifiers` module keep a set of identifiers (`_identifiers`) to be used
inside `quote_identifier` function (inside the same module). `_identifiers` is
a set of reserved words from `omniscidb` language.

`quote_identifiers` is used to put quotes around the string sent if the string
match to specific criteria.

Timestamp/Date operations
-------------------------

**Interval:**

omniscidb Interval statement allows just the following date/time attribute: YEAR, DAY,
MONTH, HOUR, MINUTE, SECOND

To use the interval statement, it is necessary to use a `integer literal/constant`
and use the `to_interval` method:

.. code-block:: python

    >>> t['arr_timestamp'] + ibis.literal(1).to_interval('Y')

.. code-block:: sql

    SELECT TIMESTAMPADD(YEAR, 1, "arr_timestamp") AS tmp
    FROM omniscidb.flights_2008_10k

Another way to use intervals is using `ibis.interval(years=1)`

**Extract date/time**

To extract a date part information from a timestamp, `extract` would be used:

.. code-block:: python

    >>> t['arr_timestamp'].extract('YEAR')

The `extract` method is just available on `ibis.omniscidb` backend.

The operators allowed are: YEAR, QUARTER, MONTH, DAY, HOUR, MINUTE, SECOND,
DOW, ISODOW, DOY, EPOCH, QUARTERDAY, WEEK

**Direct functions to extract date/time**

There are some direct functions to extract date/time, the following shows how
to use them:

.. code-block:: python

    >>> t['arr_timestamp'].year()
    >>> t['arr_timestamp'].month()
    >>> t['arr_timestamp'].day()
    >>> t['arr_timestamp'].hour()
    >>> t['arr_timestamp'].minute()
    >>> t['arr_timestamp'].second()

The result will be:

.. code-block:: sql

    SELECT EXTRACT(YEAR FROM "arr_timestamp") AS tmp
    FROM omniscidb.flights_2008_10k

    SELECT EXTRACT(MONTH FROM "arr_timestamp") AS tmp
    FROM omniscidb.flights_2008_10k

    SELECT EXTRACT(DAY FROM "arr_timestamp") AS tmp
    FROM omniscidb.flights_2008_10k

    SELECT EXTRACT(HOUR FROM "arr_timestamp") AS tmp
    FROM omniscidb.flights_2008_10k

    SELECT EXTRACT(MINUTE FROM "arr_timestamp") AS tmp
    FROM omniscidb.flights_2008_10k

    SELECT EXTRACT(SECOND FROM "arr_timestamp") AS tmp
    FROM omniscidb.flights_2008_10k

**Timestap/Date Truncate**

A truncate timestamp/data value function is available as `truncate`:

.. code-block:: python

    >>> t['arr_timestamp'].truncate(date_part)

The date part operators allowed are: YEAR or Y, QUARTER or Q, MONTH or M,
DAY or D, HOUR or h, MINUTE or m, SECOND or s, WEEK, MILLENNIUM, CENTURY,
DECADE, QUARTERDAY


String operations
-----------------

- `byte_length` is not part of `ibis` `string` operations, it will work just for `omniscidb` backend.

`Not` operation can be done using `~` operator:

.. code-block:: python

    >>> ~t['dest_name'].like('L%')

`regexp` and `regexp_like` operations can be done using `re_search` operation:

.. code-block:: python

    >>> t['dest_name'].re_search('L%')


Aggregate operations
====================

The aggregation operations available are: max, min, mean, count, distinct and count, nunique, approx_nunique.

The following examples show how to use count operations:

- get the row count of the table: `t['taxiin'].count()`
- get the distinct count of a field: `t['taxiin'].distinct().count()` or `t['taxiin'].nunique().name('v')`
- get the approximate distinct count of a field: `t['taxiin'].approx_nunique(10)`


Best practices
--------------

- Use `Numpy` standard for docstrings: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
- Use `format` string function to format a string instead of `%` statement.


History
-------

New implementations on `ibis` core:

- Trigonometric operations (https://github.com/ibis-project/ibis/issues/893 );
- Radians and Degrees operations (https://github.com/ibis-project/ibis/issues/1431 );
- PI constant (https://github.com/ibis-project/ibis/issues/1418 );
- Correlation and Covariation operations added (https://github.com/ibis-project/ibis/issues/1432 );
- ILIKE operation (https://github.com/ibis-project/ibis/issues/1433 );
- Distance operation (https://github.com/ibis-project/ibis/issues/1434 );

Issues appointed:

- `Ibis` `CASE` statement wasn't allowing input and output with different types (https://github.com/ibis-project/ibis/issues/93 )
- At this time, not all omniscidb `date parts` are available on `ibis` (https://github.com/ibis-project/ibis/issues/1430 )


Pull Requests:

- https://github.com/ibis-project/ibis/pull/1419

References
----------

- ibis API: http://docs.ibis-project.org/api.html
