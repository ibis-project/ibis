from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ibis.expr.datatypes import DataType
from ibis.expr.schema import Schema

T = TypeVar('T', covariant=True)
S = TypeVar('S', covariant=True)
VS = TypeVar('VS', covariant=True)
VC = TypeVar('VC', covariant=True)
VT = TypeVar('VT', covariant=True)


class Format(ABC, Generic[T, S, VS, VC, VT]):
    @classmethod
    @abstractmethod
    def from_dtype(cls, dtype: DataType) -> T:
        """Convert an Ibis DataType to a format-specific type object.

        Parameters
        ----------
        dtype
            The Ibis DataType to convert.

        Returns
        -------
        Format-specific type object.
        """

    @classmethod
    @abstractmethod
    def to_dtype(cls, typ: T) -> DataType:
        """Convert a format-specific type object to an Ibis DataType.

        Parameters
        ----------
        typ
            The format-specific type object to convert.

        Returns
        -------
        Ibis DataType.
        """

    @classmethod
    @abstractmethod
    def from_schema(cls, schema: Schema) -> S:
        """Convert an Ibis Schema to a format-specific schema object.

        Parameters
        ----------
        schema
            The Ibis Schema to convert.

        Returns
        -------
        Format-specific schema object.
        """

    @classmethod
    @abstractmethod
    def to_schema(cls, obj: S) -> Schema:
        """Convert a format-specific schema object to an Ibis Schema.

        Parameters
        ----------
        obj
            The format-specific schema object to convert.

        Returns
        -------
        Ibis Schema.
        """

    @classmethod
    @abstractmethod
    def convert_scalar(cls, obj: VS, dtype: DataType) -> VS:
        """Convert a format-specific scalar to the given ibis datatype.

        Parameters
        ----------
        obj
            The format-specific scalar value to convert.
        dtype
            The Ibis datatype to convert to.

        Returns
        -------
        Format specific scalar corresponding to the given Ibis datatype.
        """

    @classmethod
    @abstractmethod
    def convert_column(cls, obj: VC, dtype: DataType) -> VC:
        """Convert a format-specific column to the given ibis datatype.

        Parameters
        ----------
        obj
            The format-specific column value to convert.
        dtype
            The Ibis datatype to convert to.

        Returns
        -------
        Format specific column corresponding to the given Ibis datatype.
        """

    @classmethod
    @abstractmethod
    def convert_table(cls, obj: VT, schema: Schema) -> VT:
        """Convert a format-specific table to the given ibis schema.

        Parameters
        ----------
        obj
            The format-specific table-like object to convert.
        schema
            The Ibis schema to convert to.

        Returns
        -------
        Format specific table-like object corresponding to the given Ibis schema.
        """
