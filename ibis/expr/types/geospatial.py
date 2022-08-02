from __future__ import annotations

from typing import TYPE_CHECKING

from public import public

from ibis.expr.types.numeric import NumericColumn, NumericScalar, NumericValue

if TYPE_CHECKING:
    from ibis.expr import types as ir


@public
class GeoSpatialValue(NumericValue):
    def area(self) -> ir.FloatingValue:
        """Compute the area of a geospatial value.

        Returns
        -------
        FloatingValue
            The area of `self`
        """
        from ibis.expr import operations as ops

        op = ops.GeoArea(self)
        return op.to_expr()

    def as_binary(self) -> ir.BinaryValue:
        """Get the geometry as well-known bytes (WKB) without the SRID data.

        Returns
        -------
        BinaryValue
            Binary value
        """
        from ibis.expr import operations as ops

        op = ops.GeoAsBinary(self)
        return op.to_expr()

    def as_ewkt(self) -> ir.StringValue:
        """Get the geometry as well-known text (WKT) with the SRID data.

        Returns
        -------
        StringValue
            String value
        """
        from ibis.expr import operations as ops

        op = ops.GeoAsEWKT(self)
        return op.to_expr()

    def as_text(self) -> ir.StringValue:
        """Get the geometry as well-known text (WKT) without the SRID data.

        Returns
        -------
        StringValue
            String value
        """
        from ibis.expr import operations as ops

        op = ops.GeoAsText(self)
        return op.to_expr()

    def as_ewkb(self) -> ir.BinaryValue:
        """Get the geometry as well-known bytes (WKB) with the SRID data.

        Returns
        -------
        BinaryValue
            WKB value
        """
        from ibis.expr import operations as ops

        op = ops.GeoAsEWKB(self)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoContains(self, right)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoContainsProperly(self, right)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoCovers(self, right)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoCoveredBy(self, right)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoCrosses(self, right)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoDFullyWithin(self, right, distance)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoDisjoint(self, right)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoDWithin(self, right, distance)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoEquals(self, right)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoGeometryN(self, n)
        return op.to_expr()

    def geometry_type(self) -> ir.StringValue:
        """Get the type of a geometry.

        Returns
        -------
        StringValue
            String representing the type of `self`.
        """
        from ibis.expr import operations as ops

        op = ops.GeoGeometryType(self)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoIntersects(self, right)
        return op.to_expr()

    def is_valid(self) -> ir.BooleanValue:
        """Check if the geometry is valid.

        Returns
        -------
        BooleanValue
            Whether `self` is valid
        """
        from ibis.expr import operations as ops

        op = ops.GeoIsValid(self)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoOrderingEquals(self, right)
        return op.to_expr()

    def overlaps(self, right: GeoSpatialValue) -> ir.BooleanValue:
        """Check if the geometries share space, have the same dimension, and
        are not completely contained by each other.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        BooleanValue
            Overlaps indicator
        """
        from ibis.expr import operations as ops

        op = ops.GeoOverlaps(self, right)
        return op.to_expr()

    def touches(self, right: GeoSpatialValue) -> ir.BooleanValue:
        """Check if the geometries have at least one point in common, but do
        not intersect.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        BooleanValue
            Whether self and right are touching
        """
        from ibis.expr import operations as ops

        op = ops.GeoTouches(self, right)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoDistance(self, right)
        return op.to_expr()

    def length(self) -> ir.FloatingValue:
        """Compute the length of a geospatial expression.

        Returns
        -------
        FloatingValue
            Length of `self`
        """
        from ibis.expr import operations as ops

        op = ops.GeoLength(self)
        return op.to_expr()

    def perimeter(self) -> ir.FloatingValue:
        """Compute the perimeter of a geospatial expression.

        Returns
        -------
        FloatingValue
            Perimeter of `self`
        """
        from ibis.expr import operations as ops

        op = ops.GeoPerimeter(self)
        return op.to_expr()

    def max_distance(self, right: GeoSpatialValue) -> ir.FloatingValue:
        """Returns the 2-dimensional maximum distance between two geometries in
        projected units.

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
        from ibis.expr import operations as ops

        op = ops.GeoMaxDistance(self, right)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoUnion(self, right)
        return op.to_expr()

    def x(self) -> ir.FloatingValue:
        """Return the X coordinate of `self`, or NULL if not available.

        Input must be a point.

        Returns
        -------
        FloatingValue
            X coordinate of `self`
        """
        from ibis.expr import operations as ops

        op = ops.GeoX(self)
        return op.to_expr()

    def y(self) -> ir.FloatingValue:
        """Return the Y coordinate of `self`, or NULL if not available.

        Input must be a point.

        Returns
        -------
        FloatingValue
            Y coordinate of `self`
        """
        from ibis.expr import operations as ops

        op = ops.GeoY(self)
        return op.to_expr()

    def x_min(self) -> ir.FloatingValue:
        """Return the X minima of a geometry.

        Returns
        -------
        FloatingValue
            X minima
        """
        from ibis.expr import operations as ops

        op = ops.GeoXMin(self)
        return op.to_expr()

    def x_max(self) -> ir.FloatingValue:
        """Return the X maxima of a geometry.

        Returns
        -------
        FloatingValue
            X maxima
        """
        from ibis.expr import operations as ops

        op = ops.GeoXMax(self)
        return op.to_expr()

    def y_min(self) -> ir.FloatingValue:
        """Return the Y minima of a geometry.

        Returns
        -------
        FloatingValue
            Y minima
        """
        from ibis.expr import operations as ops

        op = ops.GeoYMin(self)
        return op.to_expr()

    def y_max(self) -> ir.FloatingValue:
        """Return the Y maxima of a geometry.

        Returns
        -------
        FloatingValue
            Y maxima
        """
        from ibis.expr import operations as ops

        op = ops.GeoYMax(self)
        return op.to_expr()

    def start_point(self) -> PointValue:
        """Return the first point of a `LINESTRING` geometry as a `POINT`.

        Return `NULL` if the input parameter is not a `LINESTRING`

        Returns
        -------
        PointValue
            Start point
        """
        from ibis.expr import operations as ops

        op = ops.GeoStartPoint(self)
        return op.to_expr()

    def end_point(self) -> PointValue:
        """Return the last point of a `LINESTRING` geometry as a `POINT`.

        Return `NULL` if the input parameter is not a `LINESTRING`

        Returns
        -------
        PointValue
            End point
        """
        from ibis.expr import operations as ops

        op = ops.GeoEndPoint(self)
        return op.to_expr()

    def point_n(self, n: ir.IntegerValue) -> PointValue:
        """Return the Nth point in a single linestring in the geometry.
        Negative values are counted backwards from the end of the LineString,
        so that -1 is the last point. Returns NULL if there is no linestring in
        the geometry

        Parameters
        ----------
        n
            Nth point index

        Returns
        -------
        PointValue
            Nth point in `self`
        """
        from ibis.expr import operations as ops

        op = ops.GeoPointN(self, n)
        return op.to_expr()

    def n_points(self) -> ir.IntegerValue:
        """Return the number of points in a geometry. Works for all geometries

        Returns
        -------
        IntegerValue
            Number of points
        """
        from ibis.expr import operations as ops

        op = ops.GeoNPoints(self)
        return op.to_expr()

    def n_rings(self) -> ir.IntegerValue:
        """Return the number of rings for polygons and multipolygons.

        Outer rings are counted as well.

        Returns
        -------
        IntegerValue
            Number of rings
        """
        from ibis.expr import operations as ops

        op = ops.GeoNRings(self)
        return op.to_expr()

    def srid(self) -> ir.IntegerValue:
        """Return the spatial reference identifier for the ST_Geometry.

        Returns
        -------
        IntegerValue
            SRID
        """
        from ibis.expr import operations as ops

        op = ops.GeoSRID(self)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoSetSRID(self, srid=srid)
        return op.to_expr()

    def buffer(self, radius: float | ir.FloatingValue) -> GeoSpatialValue:
        """Returns a geometry that represents all points whose distance from
        this Geometry is less than or equal to distance. Calculations are in
        the Spatial Reference System of this Geometry.

        Parameters
        ----------
        radius
            Floating expression

        Returns
        -------
        GeoSpatialValue
            Geometry expression
        """
        from ibis.expr import operations as ops

        op = ops.GeoBuffer(self, radius=radius)
        return op.to_expr()

    def centroid(self) -> PointValue:
        """Returns the centroid of the geometry.

        Returns
        -------
        PointValue
            The centroid
        """
        from ibis.expr import operations as ops

        op = ops.GeoCentroid(self)
        return op.to_expr()

    def envelope(self) -> ir.PolygonValue:
        """Returns a geometry representing the bounding box of `self`.

        Returns
        -------
        PolygonValue
            A polygon
        """
        from ibis.expr import operations as ops

        op = ops.GeoEnvelope(self)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoWithin(self, right)
        return op.to_expr()

    def azimuth(self, right: GeoSpatialValue) -> ir.FloatingValue:
        """Return the angle in radians from the horizontal of the vector
        defined by `self` and `right`.

        Angle is computed clockwise from down-to-up on the clock:
        12=0; 3=PI/2; 6=PI; 9=3PI/2.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        FloatingValue
            azimuth
        """
        from ibis.expr import operations as ops

        op = ops.GeoAzimuth(self, right)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoIntersection(self, right)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoDifference(self, right)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoSimplify(self, tolerance, preserve_collapsed)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoTransform(self, srid)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoLineLocatePoint(self, right)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoLineSubstring(self, start, end)
        return op.to_expr()

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
        from ibis.expr import operations as ops

        op = ops.GeoLineMerge(self)
        return op.to_expr()


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
        from ibis.expr import operations as ops

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
