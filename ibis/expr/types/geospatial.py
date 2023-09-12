from __future__ import annotations

from typing import TYPE_CHECKING

from public import public

import ibis.expr.operations as ops
from ibis.expr.types.numeric import NumericColumn, NumericScalar, NumericValue

if TYPE_CHECKING:
    import ibis.expr.types as ir


@public
class GeoSpatialValue(NumericValue):
    def area(self) -> ir.FloatingValue:
        """Compute the area of a geospatial value.

        Returns
        -------
        FloatingValue
            The area of `self`
        """
        return ops.GeoArea(self).to_expr()

    def as_binary(self) -> ir.BinaryValue:
        """Get the geometry as well-known bytes (WKB) without the SRID data.

        Returns
        -------
        BinaryValue
            Binary value
        """
        return ops.GeoAsBinary(self).to_expr()

    def as_ewkt(self) -> ir.StringValue:
        """Get the geometry as well-known text (WKT) with the SRID data.

        Returns
        -------
        StringValue
            String value
        """
        return ops.GeoAsEWKT(self).to_expr()

    def as_text(self) -> ir.StringValue:
        """Get the geometry as well-known text (WKT) without the SRID data.

        Returns
        -------
        StringValue
            String value
        """
        return ops.GeoAsText(self).to_expr()

    def as_ewkb(self) -> ir.BinaryValue:
        """Get the geometry as well-known bytes (WKB) with the SRID data.

        Returns
        -------
        BinaryValue
            WKB value
        """
        return ops.GeoAsEWKB(self).to_expr()

    def contains(self, right: GeoSpatialValue) -> ir.BooleanValue:
        """Check if the geometry contains the `right`.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        BooleanValue
            Whether `self` contains `right`
        """
        return ops.GeoContains(self, right).to_expr()

    def contains_properly(self, right: GeoSpatialValue) -> ir.BooleanValue:
        """Check if the first geometry contains the second one.

        Excludes common border points.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        BooleanValue
            Whether self contains right excluding border points.
        """
        return ops.GeoContainsProperly(self, right).to_expr()

    def covers(self, right: GeoSpatialValue) -> ir.BooleanValue:
        """Check if the first geometry covers the second one.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        BooleanValue
            Whether `self` covers `right`
        """
        return ops.GeoCovers(self, right).to_expr()

    def covered_by(self, right: GeoSpatialValue) -> ir.BooleanValue:
        """Check if the first geometry is covered by the second one.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        BooleanValue
            Whether `self` is covered by `right`
        """
        return ops.GeoCoveredBy(self, right).to_expr()

    def crosses(self, right: GeoSpatialValue) -> ir.BooleanValue:
        """Check if the geometries have at least one interior point in common.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        BooleanValue
            Whether `self` and `right` have at least one common interior point.
        """
        return ops.GeoCrosses(self, right).to_expr()

    def d_fully_within(
        self,
        right: GeoSpatialValue,
        distance: ir.FloatingValue,
    ) -> ir.BooleanValue:
        """Check if `self` is entirely within `distance` from `right`.

        Parameters
        ----------
        right
            Right geometry
        distance
            Distance to check

        Returns
        -------
        BooleanValue
            Whether `self` is within a specified distance from `right`.
        """
        return ops.GeoDFullyWithin(self, right, distance).to_expr()

    def disjoint(self, right: GeoSpatialValue) -> ir.BooleanValue:
        """Check if the geometries have no points in common.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        BooleanValue
            Whether `self` and `right` are disjoint
        """
        return ops.GeoDisjoint(self, right).to_expr()

    def d_within(
        self,
        right: GeoSpatialValue,
        distance: ir.FloatingValue,
    ) -> ir.BooleanValue:
        """Check if `self` is partially within `distance` from `right`.

        Parameters
        ----------
        right
            Right geometry
        distance
            Distance to check

        Returns
        -------
        BooleanValue
            Whether `self` is partially within `distance` from `right`.
        """
        return ops.GeoDWithin(self, right, distance).to_expr()

    def geo_equals(self, right: GeoSpatialValue) -> ir.BooleanValue:
        """Check if the geometries are equal.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        BooleanValue
            Whether `self` equals `right`
        """
        return ops.GeoEquals(self, right).to_expr()

    def geometry_n(self, n: int | ir.IntegerValue) -> GeoSpatialValue:
        """Get the 1-based Nth geometry of a multi geometry.

        Parameters
        ----------
        n
            Nth geometry index

        Returns
        -------
        GeoSpatialValue
            Geometry value
        """
        return ops.GeoGeometryN(self, n).to_expr()

    def geometry_type(self) -> ir.StringValue:
        """Get the type of a geometry.

        Returns
        -------
        StringValue
            String representing the type of `self`.
        """
        return ops.GeoGeometryType(self).to_expr()

    def intersects(self, right: GeoSpatialValue) -> ir.BooleanValue:
        """Check if the geometries share any points.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        BooleanValue
            Whether `self` intersects `right`
        """
        return ops.GeoIntersects(self, right).to_expr()

    def is_valid(self) -> ir.BooleanValue:
        """Check if the geometry is valid.

        Returns
        -------
        BooleanValue
            Whether `self` is valid
        """
        return ops.GeoIsValid(self).to_expr()

    def ordering_equals(self, right: GeoSpatialValue) -> ir.BooleanValue:
        """Check if two geometries are equal and have the same point ordering.

        Returns true if the two geometries are equal and the coordinates
        are in the same order.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        BooleanValue
            Whether points and orderings are equal.
        """
        return ops.GeoOrderingEquals(self, right).to_expr()

    def overlaps(self, right: GeoSpatialValue) -> ir.BooleanValue:
        """Check if the geometries share space, have the same dimension, and are not completely contained by each other.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        BooleanValue
            Overlaps indicator
        """
        return ops.GeoOverlaps(self, right).to_expr()

    def touches(self, right: GeoSpatialValue) -> ir.BooleanValue:
        """Check if the geometries have at least one point in common, but do not intersect.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        BooleanValue
            Whether self and right are touching
        """
        return ops.GeoTouches(self, right).to_expr()

    def distance(self, right: GeoSpatialValue) -> ir.FloatingValue:
        """Compute the distance between two geospatial expressions.

        Parameters
        ----------
        right
            Right geometry or geography

        Returns
        -------
        FloatingValue
            Distance between `self` and `right`
        """
        return ops.GeoDistance(self, right).to_expr()

    def length(self) -> ir.FloatingValue:
        """Compute the length of a geospatial expression.

        Returns
        -------
        FloatingValue
            Length of `self`
        """
        return ops.GeoLength(self).to_expr()

    def perimeter(self) -> ir.FloatingValue:
        """Compute the perimeter of a geospatial expression.

        Returns
        -------
        FloatingValue
            Perimeter of `self`
        """
        return ops.GeoPerimeter(self).to_expr()

    def max_distance(self, right: GeoSpatialValue) -> ir.FloatingValue:
        """Returns the 2-dimensional max distance between two geometries in projected units.

        If `self` and `right` are the same geometry the function will return
        the distance between the two vertices most far from each other in that
        geometry.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        FloatingValue
            Maximum distance
        """
        return ops.GeoMaxDistance(self, right).to_expr()

    def union(self, right: GeoSpatialValue) -> GeoSpatialValue:
        """Merge two geometries into a union geometry.

        Returns the pointwise union of the two geometries.
        This corresponds to the non-aggregate version the PostGIS ST_Union.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        GeoSpatialValue
            Union of geometries
        """
        return ops.GeoUnion(self, right).to_expr()

    def x(self) -> ir.FloatingValue:
        """Return the X coordinate of `self`, or NULL if not available.

        Input must be a point.

        Returns
        -------
        FloatingValue
            X coordinate of `self`
        """
        return ops.GeoX(self).to_expr()

    def y(self) -> ir.FloatingValue:
        """Return the Y coordinate of `self`, or NULL if not available.

        Input must be a point.

        Returns
        -------
        FloatingValue
            Y coordinate of `self`
        """
        return ops.GeoY(self).to_expr()

    def x_min(self) -> ir.FloatingValue:
        """Return the X minima of a geometry.

        Returns
        -------
        FloatingValue
            X minima
        """
        return ops.GeoXMin(self).to_expr()

    def x_max(self) -> ir.FloatingValue:
        """Return the X maxima of a geometry.

        Returns
        -------
        FloatingValue
            X maxima
        """
        return ops.GeoXMax(self).to_expr()

    def y_min(self) -> ir.FloatingValue:
        """Return the Y minima of a geometry.

        Returns
        -------
        FloatingValue
            Y minima
        """
        return ops.GeoYMin(self).to_expr()

    def y_max(self) -> ir.FloatingValue:
        """Return the Y maxima of a geometry.

        Returns
        -------
        FloatingValue
            Y maxima
        """
        return ops.GeoYMax(self).to_expr()

    def start_point(self) -> PointValue:
        """Return the first point of a `LINESTRING` geometry as a `POINT`.

        Return `NULL` if the input parameter is not a `LINESTRING`

        Returns
        -------
        PointValue
            Start point
        """
        return ops.GeoStartPoint(self).to_expr()

    def end_point(self) -> PointValue:
        """Return the last point of a `LINESTRING` geometry as a `POINT`.

        Return `NULL` if the input parameter is not a `LINESTRING`

        Returns
        -------
        PointValue
            End point
        """
        return ops.GeoEndPoint(self).to_expr()

    def point_n(self, n: ir.IntegerValue) -> PointValue:
        """Return the Nth point in a single linestring in the geometry.

        Negative values are counted backwards from the end of the LineString,
        so that -1 is the last point. Returns NULL if there is no linestring in
        the geometry.

        Parameters
        ----------
        n
            Nth point index

        Returns
        -------
        PointValue
            Nth point in `self`
        """
        return ops.GeoPointN(self, n).to_expr()

    def n_points(self) -> ir.IntegerValue:
        """Return the number of points in a geometry. Works for all geometries.

        Returns
        -------
        IntegerValue
            Number of points
        """
        return ops.GeoNPoints(self).to_expr()

    def n_rings(self) -> ir.IntegerValue:
        """Return the number of rings for polygons and multipolygons.

        Outer rings are counted as well.

        Returns
        -------
        IntegerValue
            Number of rings
        """
        return ops.GeoNRings(self).to_expr()

    def srid(self) -> ir.IntegerValue:
        """Return the spatial reference identifier for the ST_Geometry.

        Returns
        -------
        IntegerValue
            SRID
        """
        return ops.GeoSRID(self).to_expr()

    def set_srid(self, srid: ir.IntegerValue) -> GeoSpatialValue:
        """Set the spatial reference identifier for the `ST_Geometry`.

        Parameters
        ----------
        srid
            SRID integer value

        Returns
        -------
        GeoSpatialValue
            `self` with SRID set to `srid`
        """
        return ops.GeoSetSRID(self, srid=srid).to_expr()

    def buffer(self, radius: float | ir.FloatingValue) -> GeoSpatialValue:
        """Return all points whose distance from this geometry is less than or equal to `radius`.

        Calculations are in the Spatial Reference System of this Geometry.

        Parameters
        ----------
        radius
            Floating expression

        Returns
        -------
        GeoSpatialValue
            Geometry expression
        """
        return ops.GeoBuffer(self, radius=radius).to_expr()

    def centroid(self) -> PointValue:
        """Returns the centroid of the geometry.

        Returns
        -------
        PointValue
            The centroid
        """
        return ops.GeoCentroid(self).to_expr()

    def envelope(self) -> ir.PolygonValue:
        """Returns a geometry representing the bounding box of `self`.

        Returns
        -------
        PolygonValue
            A polygon
        """
        return ops.GeoEnvelope(self).to_expr()

    def within(self, right: GeoSpatialValue) -> ir.BooleanValue:
        """Check if the first geometry is completely inside of the second.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        BooleanValue
            Whether `self` is in `right`.
        """
        return ops.GeoWithin(self, right).to_expr()

    def azimuth(self, right: GeoSpatialValue) -> ir.FloatingValue:
        """Return the angle in radians from the horizontal of the vector defined by the inputs.

        Angle is computed clockwise from down-to-up on the clock: 12=0; 3=PI/2; 6=PI; 9=3PI/2.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        FloatingValue
            azimuth
        """
        return ops.GeoAzimuth(self, right).to_expr()

    def intersection(self, right: GeoSpatialValue) -> GeoSpatialValue:
        """Return the intersection of two geometries.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        GeoSpatialValue
            Intersection of `self` and `right`
        """
        return ops.GeoIntersection(self, right).to_expr()

    def difference(self, right: GeoSpatialValue) -> GeoSpatialValue:
        """Return the difference of two geometries.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        GeoSpatialValue
            Difference of `self` and `right`
        """
        return ops.GeoDifference(self, right).to_expr()

    def simplify(
        self,
        tolerance: ir.FloatingValue,
        preserve_collapsed: ir.BooleanValue,
    ) -> GeoSpatialValue:
        """Simplify a given geometry.

        Parameters
        ----------
        tolerance
            Tolerance
        preserve_collapsed
            Whether to preserve collapsed geometries

        Returns
        -------
        GeoSpatialValue
            Simplified geometry
        """
        return ops.GeoSimplify(self, tolerance, preserve_collapsed).to_expr()

    def transform(self, srid: ir.IntegerValue) -> GeoSpatialValue:
        """Transform a geometry into a new SRID.

        Parameters
        ----------
        srid
            Integer expression

        Returns
        -------
        GeoSpatialValue
            Transformed geometry
        """
        return ops.GeoTransform(self, srid).to_expr()

    def line_locate_point(self, right: PointValue) -> ir.FloatingValue:
        """Locate the distance a point falls along the length of a line.

        Returns a float between zero and one representing the location of the
        closest point on the linestring to the given point, as a fraction of
        the total 2d line length.

        Parameters
        ----------
        right
            Point geometry

        Returns
        -------
        FloatingValue
            Fraction of the total line length
        """
        return ops.GeoLineLocatePoint(self, right).to_expr()

    def line_substring(
        self, start: ir.FloatingValue, end: ir.FloatingValue
    ) -> ir.LineStringValue:
        """Clip a substring from a LineString.

        Returns a linestring that is a substring of the input one, starting
        and ending at the given fractions of the total 2d length. The second
        and third arguments are floating point values between zero and one.
        This only works with linestrings.

        Parameters
        ----------
        start
            Start value
        end
            End value

        Returns
        -------
        LineStringValue
            Clipped linestring
        """
        return ops.GeoLineSubstring(self, start, end).to_expr()

    def line_merge(self) -> ir.LineStringValue:
        """Merge a `MultiLineString` into a `LineString`.

        Returns a (set of) LineString(s) formed by sewing together the
        constituent line work of a MultiLineString. If a geometry other than
        a LineString or MultiLineString is given, this will return an empty
        geometry collection.

        Returns
        -------
        GeoSpatialValue
            Merged linestrings
        """
        return ops.GeoLineMerge(self).to_expr()


@public
class GeoSpatialScalar(NumericScalar, GeoSpatialValue):
    pass


@public
class GeoSpatialColumn(NumericColumn, GeoSpatialValue):
    def unary_union(self) -> ir.GeoSpatialScalar:
        """Aggregate a set of geometries into a union.

        This corresponds to the aggregate version of the PostGIS ST_Union.
        We give it a different name (following the corresponding method
        in GeoPandas) to avoid name conflicts with the non-aggregate version.

        Returns
        -------
        GeoSpatialScalar
            Union of geometries
        """
        return ops.GeoUnaryUnion(self).to_expr().name("union")


@public
class PointValue(GeoSpatialValue):
    pass


@public
class PointScalar(GeoSpatialScalar, PointValue):
    pass


@public
class PointColumn(GeoSpatialColumn, PointValue):
    pass


@public
class LineStringValue(GeoSpatialValue):
    pass


@public
class LineStringScalar(GeoSpatialScalar, LineStringValue):
    pass


@public
class LineStringColumn(GeoSpatialColumn, LineStringValue):
    pass


@public
class PolygonValue(GeoSpatialValue):
    pass


@public
class PolygonScalar(GeoSpatialScalar, PolygonValue):
    pass


@public
class PolygonColumn(GeoSpatialColumn, PolygonValue):
    pass


@public
class MultiLineStringValue(GeoSpatialValue):
    pass


@public
class MultiLineStringScalar(GeoSpatialScalar, MultiLineStringValue):
    pass


@public
class MultiLineStringColumn(GeoSpatialColumn, MultiLineStringValue):
    pass


@public
class MultiPointValue(GeoSpatialValue):
    pass


@public
class MultiPointScalar(GeoSpatialScalar, MultiPointValue):
    pass


@public
class MultiPointColumn(GeoSpatialColumn, MultiPointValue):
    pass


@public
class MultiPolygonValue(GeoSpatialValue):
    pass


@public
class MultiPolygonScalar(GeoSpatialScalar, MultiPolygonValue):
    pass


@public
class MultiPolygonColumn(GeoSpatialColumn, MultiPolygonValue):
    pass
