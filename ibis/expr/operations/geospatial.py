from public import public

from .. import datatypes as dt
from .. import rules as rlz
from ..signature import Argument as Arg
from .core import BinaryOp, UnaryOp
from .reductions import Reduction


@public
class GeoSpatialBinOp(BinaryOp):
    """Geo Spatial base binary"""

    left = Arg(rlz.geospatial)
    right = Arg(rlz.geospatial)


@public
class GeoSpatialUnOp(UnaryOp):
    """Geo Spatial base unary"""

    arg = Arg(rlz.geospatial)


@public
class GeoDistance(GeoSpatialBinOp):
    """Returns minimum distance between two geo spatial data"""

    output_type = rlz.shape_like('args', dt.float64)


@public
class GeoContains(GeoSpatialBinOp):
    """Check if the first geo spatial data contains the second one"""

    output_type = rlz.shape_like('args', dt.boolean)


@public
class GeoContainsProperly(GeoSpatialBinOp):
    """Check if the first geo spatial data contains the second one,
    and no boundary points are shared."""

    output_type = rlz.shape_like('args', dt.boolean)


@public
class GeoCovers(GeoSpatialBinOp):
    """Returns True if no point in Geometry B is outside Geometry A"""

    output_type = rlz.shape_like('args', dt.boolean)


@public
class GeoCoveredBy(GeoSpatialBinOp):
    """Returns True if no point in Geometry/Geography A is
    outside Geometry/Geography B"""

    output_type = rlz.shape_like('args', dt.boolean)


@public
class GeoCrosses(GeoSpatialBinOp):
    """Returns True if the supplied geometries have some, but not all,
    interior points in common."""

    output_type = rlz.shape_like('args', dt.boolean)


@public
class GeoDisjoint(GeoSpatialBinOp):
    """Returns True if the Geometries do not “spatially intersect” -
    if they do not share any space together."""

    output_type = rlz.shape_like('args', dt.boolean)


@public
class GeoEquals(GeoSpatialBinOp):
    """Returns True if the given geometries represent the same geometry."""

    output_type = rlz.shape_like('args', dt.boolean)


@public
class GeoGeometryN(GeoSpatialUnOp):
    """Returns the Nth Geometry of a Multi geometry."""

    n = Arg(rlz.integer)

    output_type = rlz.shape_like('args', dt.geometry)


@public
class GeoGeometryType(GeoSpatialUnOp):
    """Returns the type of the geometry."""

    output_type = rlz.shape_like('args', dt.string)


@public
class GeoIntersects(GeoSpatialBinOp):
    """Returns True if the Geometries/Geography “spatially intersect in 2D”
    - (share any portion of space) and False if they don’t (they are Disjoint).
    """

    output_type = rlz.shape_like('args', dt.boolean)


@public
class GeoIsValid(GeoSpatialUnOp):
    """Returns true if the geometry is well-formed."""

    output_type = rlz.shape_like('args', dt.boolean)


@public
class GeoLineLocatePoint(GeoSpatialBinOp):
    """
    Locate the distance a point falls along the length of a line.

    Returns a float between zero and one representing the location of the
    closest point on the linestring to the given point, as a fraction of the
    total 2d line length.
    """

    left = Arg(rlz.linestring)
    right = Arg(rlz.point)

    output_type = rlz.shape_like('args', dt.halffloat)


@public
class GeoLineMerge(GeoSpatialUnOp):
    """
    Merge a MultiLineString into a LineString.

    Returns a (set of) LineString(s) formed by sewing together the
    constituent line work of a multilinestring. If a geometry other than
    a linestring or multilinestring is given, this will return an empty
    geometry collection.
    """

    output_type = rlz.shape_like('args', dt.geometry)


@public
class GeoLineSubstring(GeoSpatialUnOp):
    """
    Clip a substring from a LineString.

    Returns a linestring that is a substring of the input one, starting
    and ending at the given fractions of the total 2d length. The second
    and third arguments are floating point values between zero and one.
    This only works with linestrings.
    """

    arg = Arg(rlz.linestring)

    start = Arg(rlz.floating)
    end = Arg(rlz.floating)

    output_type = rlz.shape_like('args', dt.linestring)


@public
class GeoOrderingEquals(GeoSpatialBinOp):
    """
    Check if two geometries are equal and have the same point ordering.

    Returns true if the two geometries are equal and the coordinates
    are in the same order.
    """

    output_type = rlz.shape_like('args', dt.boolean)


@public
class GeoOverlaps(GeoSpatialBinOp):
    """Returns True if the Geometries share space, are of the same dimension,
    but are not completely contained by each other."""

    output_type = rlz.shape_like('args', dt.boolean)


@public
class GeoTouches(GeoSpatialBinOp):
    """Returns True if the geometries have at least one point in common,
    but their interiors do not intersect."""

    output_type = rlz.shape_like('args', dt.boolean)


@public
class GeoUnaryUnion(Reduction):
    """Returns the pointwise union of the geometries in the column."""

    arg = Arg(rlz.column(rlz.geospatial))

    def output_type(self):
        return dt.geometry.scalar_type()


@public
class GeoUnion(GeoSpatialBinOp):
    """Returns the pointwise union of the two geometries."""

    output_type = rlz.shape_like('args', dt.geometry)


@public
class GeoArea(GeoSpatialUnOp):
    """Area of the geo spatial data"""

    output_type = rlz.shape_like('args', dt.float64)


@public
class GeoPerimeter(GeoSpatialUnOp):
    """Perimeter of the geo spatial data"""

    output_type = rlz.shape_like('args', dt.float64)


@public
class GeoLength(GeoSpatialUnOp):
    """Length of geo spatial data"""

    output_type = rlz.shape_like('args', dt.float64)


@public
class GeoMaxDistance(GeoSpatialBinOp):
    """Returns the 2-dimensional maximum distance between two geometries in
    projected units. If g1 and g2 is the same geometry the function will
    return the distance between the two vertices most far from each other
    in that geometry
    """

    output_type = rlz.shape_like('args', dt.float64)


@public
class GeoX(GeoSpatialUnOp):
    """Return the X coordinate of the point, or NULL if not available.
    Input must be a point
    """

    output_type = rlz.shape_like('args', dt.float64)


@public
class GeoY(GeoSpatialUnOp):
    """Return the Y coordinate of the point, or NULL if not available.
    Input must be a point
    """

    output_type = rlz.shape_like('args', dt.float64)


@public
class GeoXMin(GeoSpatialUnOp):
    """Returns Y minima of a bounding box 2d or 3d or a geometry"""

    output_type = rlz.shape_like('args', dt.float64)


@public
class GeoXMax(GeoSpatialUnOp):
    """Returns X maxima of a bounding box 2d or 3d or a geometry"""

    output_type = rlz.shape_like('args', dt.float64)


@public
class GeoYMin(GeoSpatialUnOp):
    """Returns Y minima of a bounding box 2d or 3d or a geometry"""

    output_type = rlz.shape_like('args', dt.float64)


@public
class GeoYMax(GeoSpatialUnOp):
    """Returns Y maxima of a bounding box 2d or 3d or a geometry"""

    output_type = rlz.shape_like('args', dt.float64)


@public
class GeoStartPoint(GeoSpatialUnOp):
    """Returns the first point of a LINESTRING geometry as a POINT or
    NULL if the input parameter is not a LINESTRING
    """

    output_type = rlz.shape_like('arg', dt.point)


@public
class GeoEndPoint(GeoSpatialUnOp):
    """Returns the last point of a LINESTRING geometry as a POINT or
    NULL if the input parameter is not a LINESTRING
    """

    output_type = rlz.shape_like('arg', dt.point)


@public
class GeoPoint(GeoSpatialBinOp):
    """
    Return a point constructed on the fly from the provided coordinate values.
    Constant coordinates result in construction of a POINT literal.
    """

    left = Arg(rlz.numeric)
    right = Arg(rlz.numeric)
    output_type = rlz.shape_like('args', dt.point)


@public
class GeoPointN(GeoSpatialUnOp):
    """Return the Nth point in a single linestring in the geometry.
    Negative values are counted backwards from the end of the LineString,
    so that -1 is the last point. Returns NULL if there is no linestring in
    the geometry
    """

    n = Arg(rlz.integer)
    output_type = rlz.shape_like('args', dt.point)


@public
class GeoNPoints(GeoSpatialUnOp):
    """Return the number of points in a geometry. Works for all geometries"""

    output_type = rlz.shape_like('args', dt.int64)


@public
class GeoNRings(GeoSpatialUnOp):
    """If the geometry is a polygon or multi-polygon returns the number of
    rings. It counts the outer rings as well
    """

    output_type = rlz.shape_like('args', dt.int64)


@public
class GeoSRID(GeoSpatialUnOp):
    """Returns the spatial reference identifier for the ST_Geometry."""

    output_type = rlz.shape_like('args', dt.int64)


@public
class GeoSetSRID(GeoSpatialUnOp):
    """Set the spatial reference identifier for the ST_Geometry."""

    srid = Arg(rlz.integer)
    output_type = rlz.shape_like('args', dt.geometry)


@public
class GeoBuffer(GeoSpatialUnOp):
    """Returns a geometry that represents all points whose distance from this
    Geometry is less than or equal to distance. Calculations are in the
    Spatial Reference System of this Geometry.
    """

    radius = Arg(rlz.floating)

    output_type = rlz.shape_like('args', dt.geometry)


@public
class GeoCentroid(GeoSpatialUnOp):
    """Returns the geometric center of a geometry."""

    output_type = rlz.shape_like('arg', dt.point)


@public
class GeoDFullyWithin(GeoSpatialBinOp):
    """Returns True if the geometries are fully within the specified distance
    of one another.
    """

    distance = Arg(rlz.floating)

    output_type = rlz.shape_like('args', dt.boolean)


@public
class GeoDWithin(GeoSpatialBinOp):
    """Returns True if the geometries are within the specified distance
    of one another.
    """

    distance = Arg(rlz.floating)

    output_type = rlz.shape_like('args', dt.boolean)


@public
class GeoEnvelope(GeoSpatialUnOp):
    """Represents the bounding box of the supplied geometry."""

    output_type = rlz.shape_like('arg', dt.polygon)


@public
class GeoAzimuth(GeoSpatialBinOp):
    """Returns the angle in radians from the horizontal of the vector defined
    by pointA and pointB. Angle is computed clockwise from down-to-up:
    on the clock: 12=0; 3=PI/2; 6=PI; 9=3PI/2.
    """

    left = Arg(rlz.point)
    right = Arg(rlz.point)

    output_type = rlz.shape_like('args', dt.float64)


@public
class GeoWithin(GeoSpatialBinOp):
    """Returns True if the geometry A is completely inside geometry B"""

    output_type = rlz.shape_like('args', dt.boolean)


@public
class GeoIntersection(GeoSpatialBinOp):
    """Returns a geometry that represents the point set intersection
    of the Geometries.
    """

    output_type = rlz.shape_like('args', dt.geometry)


@public
class GeoDifference(GeoSpatialBinOp):
    """Returns a geometry that represents that part of geometry A
    that does not intersect with geometry B
    """

    output_type = rlz.shape_like('args', dt.geometry)


@public
class GeoSimplify(GeoSpatialUnOp):
    """Returns a simplified version of the given geometry."""

    tolerance = Arg(rlz.floating)
    preserve_collapsed = Arg(rlz.boolean)

    output_type = rlz.shape_like('arg', dt.geometry)


@public
class GeoTransform(GeoSpatialUnOp):
    """Returns a transformed version of the given geometry into a new SRID."""

    srid = Arg(rlz.integer)

    output_type = rlz.shape_like('arg', dt.geometry)


@public
class GeoAsBinary(GeoSpatialUnOp):
    """Return the Well-Known Binary (WKB) representation of the
    geometry/geography without SRID meta data.
    """

    output_type = rlz.shape_like('arg', dt.binary)


@public
class GeoAsEWKB(GeoSpatialUnOp):
    """Return the Well-Known Binary (WKB) representation of the
    geometry/geography with SRID meta data.
    """

    output_type = rlz.shape_like('arg', dt.binary)


@public
class GeoAsEWKT(GeoSpatialUnOp):
    """Return the Well-Known Text (WKT) representation of the
    geometry/geography with SRID meta data.
    """

    output_type = rlz.shape_like('arg', dt.string)


@public
class GeoAsText(GeoSpatialUnOp):
    """Return the Well-Known Text (WKT) representation of the
    geometry/geography without SRID metadata.
    """

    output_type = rlz.shape_like('arg', dt.string)
