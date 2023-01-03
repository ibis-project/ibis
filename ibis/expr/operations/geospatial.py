from __future__ import annotations

from public import public

from ibis.expr import datatypes as dt
from ibis.expr import rules as rlz
from ibis.expr.operations.core import Binary, Unary
from ibis.expr.operations.reductions import Reduction


@public
class GeoSpatialBinOp(Binary):
    """Geo Spatial base binary."""

    left = rlz.geospatial
    right = rlz.geospatial


@public
class GeoSpatialUnOp(Unary):
    """Geo Spatial base unary."""

    arg = rlz.geospatial


@public
class GeoDistance(GeoSpatialBinOp):
    """Returns minimum distance between two geospatial operands."""

    output_dtype = dt.float64


@public
class GeoContains(GeoSpatialBinOp):
    """Check if the first geo spatial data contains the second one."""

    output_dtype = dt.boolean


@public
class GeoContainsProperly(GeoSpatialBinOp):
    """Check if the left value contains the right one, with no shared no boundary points."""

    output_dtype = dt.boolean


@public
class GeoCovers(GeoSpatialBinOp):
    """Check if no point in the right operand is outside that of the left."""

    output_dtype = dt.boolean


@public
class GeoCoveredBy(GeoSpatialBinOp):
    """Check if no point in the left operand is outside that of the right."""

    output_dtype = dt.boolean


@public
class GeoCrosses(GeoSpatialBinOp):
    """Check if the inputs have some but not all interior points in common."""

    output_dtype = dt.boolean


@public
class GeoDisjoint(GeoSpatialBinOp):
    """Check if the Geometries do not spatially intersect."""

    output_dtype = dt.boolean


@public
class GeoEquals(GeoSpatialBinOp):
    """Returns True if the given geometries represent the same geometry."""

    output_dtype = dt.boolean


@public
class GeoGeometryN(GeoSpatialUnOp):
    """Returns the Nth Geometry of a Multi geometry."""

    n = rlz.integer
    output_dtype = dt.geometry


@public
class GeoGeometryType(GeoSpatialUnOp):
    """Returns the type of the geometry."""

    output_dtype = dt.string


@public
class GeoIntersects(GeoSpatialBinOp):
    """Returns True if the Geometries/Geography “spatially intersect in 2D”.

    - (share any portion of space) and False if they don`t (they are Disjoint).
    """

    output_dtype = dt.boolean


@public
class GeoIsValid(GeoSpatialUnOp):
    """Returns true if the geometry is well-formed."""

    output_dtype = dt.boolean


@public
class GeoLineLocatePoint(GeoSpatialBinOp):
    """Locate the distance a point falls along the length of a line.

    Returns a float between zero and one representing the location of
    the closest point on the linestring to the given point, as a
    fraction of the total 2d line length.
    """

    left = rlz.linestring
    right = rlz.point

    output_dtype = dt.halffloat


@public
class GeoLineMerge(GeoSpatialUnOp):
    """Merge a MultiLineString into a LineString.

    Returns a (set of) LineString(s) formed by sewing together the
    constituent line work of a multilinestring. If a geometry other than
    a linestring or multilinestring is given, this will return an empty
    geometry collection.
    """

    output_dtype = dt.geometry


@public
class GeoLineSubstring(GeoSpatialUnOp):
    """Clip a substring from a LineString.

    Returns a linestring that is a substring of the input one, starting
    and ending at the given fractions of the total 2d length. The second
    and third arguments are floating point values between zero and one.
    This only works with linestrings.
    """

    arg = rlz.linestring
    start = rlz.floating
    end = rlz.floating

    output_dtype = dt.linestring


@public
class GeoOrderingEquals(GeoSpatialBinOp):
    """Check if two geometries are equal and have the same point ordering.

    Returns true if the two geometries are equal and the coordinates are
    in the same order.
    """

    output_dtype = dt.boolean


@public
class GeoOverlaps(GeoSpatialBinOp):
    """Check if the inputs are of the same dimension but are not completely contained by each other."""

    output_dtype = dt.boolean


@public
class GeoTouches(GeoSpatialBinOp):
    """Check if the inputs have at least one point in common but their interiors do not intersect."""

    output_dtype = dt.boolean


@public
class GeoUnaryUnion(Reduction, GeoSpatialUnOp):
    """Returns the pointwise union of the geometries in the column."""

    output_dtype = dt.geometry


@public
class GeoUnion(GeoSpatialBinOp):
    """Returns the pointwise union of the two geometries."""

    output_dtype = dt.geometry


@public
class GeoArea(GeoSpatialUnOp):
    """Area of the geo spatial data."""

    output_dtype = dt.float64


@public
class GeoPerimeter(GeoSpatialUnOp):
    """Perimeter of the geo spatial data."""

    output_dtype = dt.float64


@public
class GeoLength(GeoSpatialUnOp):
    """Length of geo spatial data."""

    output_dtype = dt.float64


@public
class GeoMaxDistance(GeoSpatialBinOp):
    """Returns the 2-dimensional max distance between two geometries in projected units.

    If g1 and g2 is the same geometry the function will return the
    distance between the two vertices most far from each other in that
    geometry
    """

    output_dtype = dt.float64


@public
class GeoX(GeoSpatialUnOp):
    """Return the X coordinate of the point, or NULL if not available.

    Input must be a point
    """

    output_dtype = dt.float64


@public
class GeoY(GeoSpatialUnOp):
    """Return the Y coordinate of the point, or NULL if not available.

    Input must be a point
    """

    output_dtype = dt.float64


@public
class GeoXMin(GeoSpatialUnOp):
    """Returns Y minima of a bounding box 2d or 3d or a geometry."""

    output_dtype = dt.float64


@public
class GeoXMax(GeoSpatialUnOp):
    """Returns X maxima of a bounding box 2d or 3d or a geometry."""

    output_dtype = dt.float64


@public
class GeoYMin(GeoSpatialUnOp):
    """Returns Y minima of a bounding box 2d or 3d or a geometry."""

    output_dtype = dt.float64


@public
class GeoYMax(GeoSpatialUnOp):
    """Returns Y maxima of a bounding box 2d or 3d or a geometry."""

    output_dtype = dt.float64


@public
class GeoStartPoint(GeoSpatialUnOp):
    """Return the first point of a `LINESTRING` geometry as a POINT.

    Returns `NULL` if the input is not a LINESTRING.
    """

    output_dtype = dt.point


@public
class GeoEndPoint(GeoSpatialUnOp):
    """Return the last point of a `LINESTRING` geometry as a POINT.

    Returns `NULL` if the input is not a LINESTRING.
    """

    output_dtype = dt.point


@public
class GeoPoint(GeoSpatialBinOp):
    """Return a point constructed from the input coordinate values.

    Constant coordinates result in construction of a POINT literal.
    """

    left = rlz.numeric
    right = rlz.numeric

    output_dtype = dt.point


@public
class GeoPointN(GeoSpatialUnOp):
    """Return the Nth point in a single linestring in the geometry.

    Negative values are counted backwards from the end of the
    LineString, so that -1 is the last point. Returns NULL if there is
    no linestring in the geometry
    """

    n = rlz.integer
    output_dtype = dt.point


@public
class GeoNPoints(GeoSpatialUnOp):
    """Return the number of points in a geometry."""

    output_dtype = dt.int64


@public
class GeoNRings(GeoSpatialUnOp):
    """Return the number of rings for polygons or multipolygons.

    Outer rings are counted.
    """

    output_dtype = dt.int64


@public
class GeoSRID(GeoSpatialUnOp):
    """Returns the spatial reference identifier for the ST_Geometry."""

    output_dtype = dt.int64


@public
class GeoSetSRID(GeoSpatialUnOp):
    """Set the spatial reference identifier for the ST_Geometry."""

    srid = rlz.integer

    output_dtype = dt.geometry


@public
class GeoBuffer(GeoSpatialUnOp):
    """Return all points whose distance from this geometry is less than or equal to `radius`.

    Calculations are in the Spatial Reference System of this geometry.
    """

    radius = rlz.floating
    output_dtype = dt.geometry


@public
class GeoCentroid(GeoSpatialUnOp):
    """Returns the geometric center of a geometry."""

    output_dtype = dt.point


@public
class GeoDFullyWithin(GeoSpatialBinOp):
    """Check if the geometries are fully within `distance` of one another."""

    distance = rlz.floating

    output_dtype = dt.boolean


@public
class GeoDWithin(GeoSpatialBinOp):
    """Check if the geometries are within `distance` of one another."""

    distance = rlz.floating

    output_dtype = dt.boolean


@public
class GeoEnvelope(GeoSpatialUnOp):
    """The bounding box of the supplied geometry."""

    output_dtype = dt.polygon


@public
class GeoAzimuth(GeoSpatialBinOp):
    """Return the angle in radians from the horizontal of the vector defined by the two inputs.

    Angle is computed clockwise from down-to-up: on the clock: 12=0;
    3=PI/2; 6=PI; 9=3PI/2.
    """

    left = rlz.point
    right = rlz.point

    output_dtype = dt.float64


@public
class GeoWithin(GeoSpatialBinOp):
    """Returns True if the geometry A is completely inside geometry B."""

    output_dtype = dt.boolean


@public
class GeoIntersection(GeoSpatialBinOp):
    """Return a geometry that represents the point-set intersection of the inputs."""

    output_dtype = dt.geometry


@public
class GeoDifference(GeoSpatialBinOp):
    """Return a geometry that is the delta between the left and right inputs."""

    output_dtype = dt.geometry


@public
class GeoSimplify(GeoSpatialUnOp):
    """Returns a simplified version of the given geometry."""

    tolerance = rlz.floating
    preserve_collapsed = rlz.boolean

    output_dtype = dt.geometry


@public
class GeoTransform(GeoSpatialUnOp):
    """Returns a transformed version of the given geometry into a new SRID."""

    srid = rlz.integer

    output_dtype = dt.geometry


@public
class GeoAsBinary(GeoSpatialUnOp):
    """Return the Well-Known Binary (WKB) representation of the input, without SRID meta data."""

    output_dtype = dt.binary


@public
class GeoAsEWKB(GeoSpatialUnOp):
    """Return the Well-Known Binary representation of the input, with SRID meta data."""

    output_dtype = dt.binary


@public
class GeoAsEWKT(GeoSpatialUnOp):
    """Return the Well-Known Text representation of the input, with SRID meta data."""

    output_dtype = dt.string


@public
class GeoAsText(GeoSpatialUnOp):
    """Return the Well-Known Text (WKT) representation of the input, without SRID metadata."""

    output_dtype = dt.string
