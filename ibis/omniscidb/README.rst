OmniSciDB IBIS backend
=================

In this document it would be explained the main aspects of `ibis` and
`omniscidb` backend implementation.

Modules
-------

omniscidb backend has 5 modules, which 3 of them are the main modules:

- `api`
- `client`
- `compiler`

The `identifiers` and `operations` modules were created to organize `compiler`
method.

api
---

`api` module is the central key to access the backend. Ensure to include
the follow code into `ibi.__init__`:

.. code-block:: python

    with suppress(ImportError):
        # pip install ibis-framework[omniscidb]
        import ibis.omniscidb.api as omniscidb

Basically, there is 3 function on `api` module>

- `compile`
- `connect`
- `verify`

`compile` method compiles a `ibis` expression to `SQL`:

.. code-block:: python

    t = omniscidb_cli.table('flights_2008_10k')
    proj = t['arrtime', 'arrdelay']
    print(ibis.omniscidb.compile(proj))

`connect` method instantiates a `MapDClient` object that connect to the `omniscidb`
database specified:

.. code-block:: python

    omniscidb_cli = ibis.omniscidb.connect(
        host='localhost', user='admin', password='HyperInteractive',
        port=6274, database='omnisci'
    )

`verify` method checks if the `ibis` expression can be compiled.

.. code-block:: python

    t = omniscidb_cli.table('flights_2008_10k')
    proj = t['arrtime', 'arrdelay']
    assert ibis.omniscidb.verify(proj) == True

client
------

`client` module has the main classes to handle connection to the `omniscidb`
database.

The main classes are:

- `MapDClient`
- `MapDQuery`
- `MapDDataType`
- `MapDCursor`

`MapDDataType` class is used to translate data type from `ibis` and to `ibis`.
Its main methods are:

- `parse`
- `to_ibis`
- `from_ibis`

`parse` method ... # @TODO

`to_ibis` method ... # @TODO

`from_ibis` method ... # @TODO

`MapDClient` class is used to connect to `omniscidb` database and manipulation data
expression. Its main methods are:

- __init__
- _build_ast
- _execute
- _fully_qualified_name
- _get_table_schema
- _table_expr_klass
- log
- close
- database
- current_database
- set_database
- exists_database
- list_databases
- exists_table
- list_tables
- get_schema
- version

`_build_ast` method is required.

`MapDQuery` class should be used redefine at least `_fetch` method. If `Query`
class is used instead, when `MapDClient.execute` method is called, a exception
is raised.

    (...) once the data arrives from the database we need to convert that data
    to a pandas DataFrame.

    The Query class, with its _fetch() method, provides a way for ibis
    SQLClient objects to do any additional processing necessary after
    the database returns results to the client.
    (http://docs.ibis-project.org/design.html#execution)

`MapDCursor` class was created just to allow `ibis.client.Query.execute`
useful automatically, because it uses `with` statement:

.. code-block:: Python
    with self.client._execute(self.compiled_ddl, results=True) as cur:
       ...

Otherwise, `MapDQuery` should rewrites `execute` method with no `with`
statement.

compiler
--------

The main classes inside `compiler` module are:

- MapDDialect
- MapDExprTranslator
- MapDQueryBuilder
- MapDSelect
- MapDSelectBuilder
- MapDTableSetFormatter

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

A new Class database function seems like this (`my_backend_operations.py`):

.. code-block:: Python

    class MyNewFunction(ops.UnaryOp):
        """My new class function"""
        output_type = rlz.shape_like('arg', 'float')

After create the new class database function, the follow step is create a
function and assign it to the DataTypes allowed to use it:

.. code-block:: Python

    def my_new_function(numeric_value):
        return MyNewFunction(numeric_value).to_expr()


    NumericValue.sin = sin

Also, it is necessary register the new function:

.. code-block:: Python
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

Now, it just need a compiler function to translate the function to a SQL code
(my_backend/compiler.py):

.. code-block:: Python
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

omniscidb Interval statement allow just the follow date/time attribute: YEAR, DAY,
MONTH, HOUR, MINUTE, SECOND

To use the interval statement, it is necessary use a `integer literal/constant`
and use the `to_interval` method:

.. code-block:: Python

    >>> t['arr_timestamp'] + ibis.literal(1).to_interval('Y')

.. code-block:: Sql

    SELECT TIMESTAMPADD(YEAR, 1, "arr_timestamp") AS tmp
    FROM omniscidb.flights_2008_10k

Another way to use intervals is using `ibis.interval(years=1)`

**Extract date/time**

To extract a date part information from a timestamp, `extract` would be used:

.. code-block:: Python

    >>> t['arr_timestamp'].extract('YEAR')

The `extract` method is just available on `ibis.omniscidb` backend.

The operators allowed are: YEAR, QUARTER, MONTH, DAY, HOUR, MINUTE, SECOND,
DOW, ISODOW, DOY, EPOCH, QUARTERDAY, WEEK

**Direct functions to extract date/time**

There is some direct functions to extract date/time, the following shows how
to use that:

.. code-block:: Python

    >>> t['arr_timestamp'].year()
    >>> t['arr_timestamp'].month()
    >>> t['arr_timestamp'].day()
    >>> t['arr_timestamp'].hour()
    >>> t['arr_timestamp'].minute()
    >>> t['arr_timestamp'].second()

The result should be:

.. code-block:: Sql

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

.. code-block:: Python

    >>> t['arr_timestamp'].truncate(date_part)

The date part operators allowed are: YEAR or Y, QUARTER or Q, MONTH or M,
DAY or D, HOUR or h, MINUTE or m, SECOND or s, WEEK, MILLENNIUM, CENTURY,
DECADE, QUARTERDAY


String operations
-----------------

- `byte_length` is not part of `ibis` `string` operations, it will work just
for `omniscidb` backend.

`Not` operation can be done using `~` operator:

.. code-block:: Python

    >>> ~t['dest_name'].like('L%')

`regexp` and `regexp_like` operations can be done using `re_search` operation:

.. code-block:: Python

    >>> t['dest_name'].re_search('L%')


Aggregate operations
====================

The aggregation operations available are: max, min, mean, count, distinct and count, nunique, approx_nunique.

The follow examples show how to use count operations:

- get the row count of the table: `t['taxiin'].count()`
- get the distinct count of a field: `t['taxiin'].distinct().count()` or `t['taxiin'].nunique().name('v')`
- get the approximate distinct count of a field: `t['taxiin'].approx_nunique(10)`


Best practices
--------------

- Use `Numpy` starndard for docstring: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
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
