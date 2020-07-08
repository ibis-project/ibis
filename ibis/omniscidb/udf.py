"""OmniSciDB User Defined Function (UDF) Implementation."""
import types
from typing import Callable, List, Optional

from rbc.omniscidb import RemoteOmnisci

from ibis.expr import datatypes as dt
from ibis.omniscidb.dtypes import ibis_dtypes_to_str
from ibis.udf import vectorized


def _create_function(name, nargs):
    """
    Create a function dinamically for the given name and number of arguments.

    Parameters
    ----------
    name : str
        name of the function
    nargs : int
        number of the arguments the function accept.

    Returns
    -------
    Callable
    """
    # this function is used just as a template
    def y():
        pass

    args_name = tuple('arg{}'.format(i) for i in range(nargs))

    co_args = (
        nargs,  # argcount
        0,  # kwonlyargcount
        0,  # nlocals
        1,  # stacksize
        0,  # flags
        bytes(),  # codestring
        (),  # constants
        args_name,  # names
        args_name,  # varnames
        y.__code__.co_filename,  # filename
        name,  # name
        0,  # firstlineno
        bytes(),  # lnotab
    )
    new_code = types.CodeType(*co_args)
    return types.FunctionType(new_code, {}, name)


class OmniSciDBUDF:
    """
    OmniSciDB UDF class.

    OmniSciDB uses RBC (Remote Backend Client) to support UDF.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = 6274,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.remote_backend_compiler = RemoteOmnisci(
            host=host,
            port=port,
            dbname=database,
            user=user,
            password=password,
        )

    def _udf(
        self,
        udf_operation: Callable,
        input_type: List[dt.DataType],
        output_type: Optional[dt.DataType] = None,
        name: Optional[str] = None,
        infer_literal: bool = True,
    ) -> Callable:
        """
        Elementwise operation for UDF.

        Parameters
        ----------
        udf_operation : Callable
            Used for creating UDF operation
        input_type : List[dt.DataType]
        output_type: dt.DataType
        name : str
            Used for reusing an existent UDF
        infer_literal : bool, default True
            Define if literal scalar values should be infered or if this method
            should accept explicitly just ibis literal.

        Returns
        -------
        Callable

        """
        if name:
            f = _create_function(name, len(input_type))
            return udf_operation(
                input_type, output_type, infer_literal=infer_literal
            )(f)

        def omnisci_wrapper(f, input_type=input_type, output_type=output_type):
            signature = '{}({})'.format(
                ibis_dtypes_to_str[output_type],
                ', '.join([ibis_dtypes_to_str[v] for v in input_type]),
            )

            if isinstance(f, vectorized.UserDefinedFunction):
                f = f.func

            self.remote_backend_compiler(signature)(f)
            self.remote_backend_compiler.register()
            return udf_operation(
                input_type, output_type, infer_literal=infer_literal
            )(f)

        return omnisci_wrapper

    def elementwise(
        self,
        input_type: List[dt.DataType],
        output_type: Optional[dt.DataType] = None,
        name: Optional[str] = None,
        infer_literal: bool = True,
    ) -> Callable:
        """
        Create an elementwise UDF operation.

        Parameters
        ----------
        input_type : List[dt.DataType]
        output_type: dt.DataType
        name : str
            Used for reusing an existent UDF
        infer_literal : bool, default True
            Define if literal scalar values should be infered or if this method
            should accept explicitly just ibis literal.

        Returns
        -------
        Callable

        """
        return self._udf(
            udf_operation=vectorized.elementwise,
            input_type=input_type,
            output_type=output_type,
            name=name,
            infer_literal=infer_literal,
        )

    def reduction(
        self,
        input_type: List[dt.DataType],
        output_type: Optional[dt.DataType] = None,
        name: Optional[str] = None,
    ) -> Callable:
        """
        Create a reduction UDF operation.

        Parameters
        ----------
        input_type : List[dt.DataType]
        output_type: dt.DataType
        name : str
            Used for reusing an existent UDF
        Returns
        -------
        Callable
        """
        raise NotImplementedError('UDF Reduction is not implemented yet.')

    def analytic(
        self,
        input_type: List[dt.DataType],
        output_type: Optional[dt.DataType] = None,
        name: Optional[str] = None,
    ) -> Callable:
        """
        Create an analytic UDF operation.

        Parameters
        ----------
        input_type : List[dt.DataType]
        output_type: dt.DataType
        name : str
            Used for reusing an existent UDF
        Returns
        -------
        Callable
        """
        raise NotImplementedError('UDF Analytic is not implemented yet.')
