from public import public

from .numeric import NumericColumn, NumericScalar, NumericValue


@public
class GeoSpatialValue(NumericValue):
    pass  # noqa: E701,E302


@public
class GeoSpatialScalar(NumericScalar, GeoSpatialValue):
    pass  # noqa: E701,E302,E501


@public
class GeoSpatialColumn(NumericColumn, GeoSpatialValue):
    pass  # noqa: E701,E302,E501


@public
class PointValue(GeoSpatialValue):
    pass  # noqa: E701,E302


@public
class PointScalar(GeoSpatialScalar, PointValue):
    pass  # noqa: E701,E302


@public
class PointColumn(GeoSpatialColumn, PointValue):
    pass  # noqa: E701,E302


@public
class LineStringValue(GeoSpatialValue):
    pass  # noqa: E701,E302


@public
class LineStringScalar(GeoSpatialScalar, LineStringValue):
    pass  # noqa: E701,E302,E501


@public
class LineStringColumn(GeoSpatialColumn, LineStringValue):
    pass  # noqa: E701,E302,E501


@public
class PolygonValue(GeoSpatialValue):
    pass  # noqa: E701,E302


@public
class PolygonScalar(GeoSpatialScalar, PolygonValue):
    pass  # noqa: E701,E302


@public
class PolygonColumn(GeoSpatialColumn, PolygonValue):
    pass  # noqa: E701,E302


@public
class MultiLineStringValue(GeoSpatialValue):
    pass  # noqa: E701,E302


@public
class MultiLineStringScalar(
    GeoSpatialScalar, MultiLineStringValue
):  # noqa: E302
    pass  # noqa: E701


@public
class MultiLineStringColumn(
    GeoSpatialColumn, MultiLineStringValue
):  # noqa: E302
    pass  # noqa: E701


@public
class MultiPointValue(GeoSpatialValue):
    pass  # noqa: E701,E302


@public
class MultiPointScalar(GeoSpatialScalar, MultiPointValue):  # noqa: E302
    pass  # noqa: E701


@public
class MultiPointColumn(GeoSpatialColumn, MultiPointValue):  # noqa: E302
    pass  # noqa: E701


@public
class MultiPolygonValue(GeoSpatialValue):
    pass  # noqa: E701,E302


@public
class MultiPolygonScalar(GeoSpatialScalar, MultiPolygonValue):  # noqa: E302
    pass  # noqa: E701


@public
class MultiPolygonColumn(GeoSpatialColumn, MultiPolygonValue):  # noqa: E302
    pass  # noqa: E701
