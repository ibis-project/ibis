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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.zones.fetch()
        >>> t.geom.area()
        ┏━━━━━━━━━━━━━━━┓
        ┃ GeoArea(geom) ┃
        ┡━━━━━━━━━━━━━━━┩
        │ float64       │
        ├───────────────┤
        │  7.903953e+07 │
        │  1.439095e+08 │
        │  3.168508e+07 │
        │  8.023733e+06 │
        │  5.041488e+07 │
        │  4.093479e+07 │
        │  3.934104e+07 │
        │  2.682802e+06 │
        │  3.416422e+07 │
        │  4.404143e+07 │
        │             … │
        └───────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.zones.fetch()
        >>> t.geom.as_text()
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ GeoAsText(geom)                                                              ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                                                                       │
        ├──────────────────────────────────────────────────────────────────────────────┤
        │ POLYGON ((933100.9183527103 192536.08569720192, 933091.0114800561 192572.17… │
        │ MULTIPOLYGON (((1033269.2435912937 172126.0078125, 1033439.6426391453 17088… │
        │ POLYGON ((1026308.7695066631 256767.6975403726, 1026495.5934945047 256638.6… │
        │ POLYGON ((992073.4667968601 203714.07598876953, 992068.6669922024 203711.50… │
        │ POLYGON ((935843.3104932606 144283.33585065603, 936046.5648079664 144173.41… │
        │ POLYGON ((966568.7466657609 158679.85468779504, 966615.255504474 158662.292… │
        │ POLYGON ((1010804.2179628164 218919.64069513977, 1011049.1648243815 218914.… │
        │ POLYGON ((1005482.2763733566 221686.46616631746, 1005304.8982993066 221499.… │
        │ POLYGON ((1043803.993348822 216615.9250395149, 1043849.7083857208 216473.16… │
        │ POLYGON ((1044355.0717166215 190734.32089698315, 1044612.1216432452 190156.… │
        │ …                                                                            │
        └──────────────────────────────────────────────────────────────────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> import shapely
        >>> t = ibis.examples.zones.fetch()
        >>> p = shapely.Point(935996.821, 191376.75)  # centroid for zone 1
        >>> plit = ibis.literal(p, "geometry")
        >>> t.geom.contains(plit).name("contains")
        ┏━━━━━━━━━━┓
        ┃ contains ┃
        ┡━━━━━━━━━━┩
        │ boolean  │
        ├──────────┤
        │ True     │
        │ False    │
        │ False    │
        │ False    │
        │ False    │
        │ False    │
        │ False    │
        │ False    │
        │ False    │
        │ False    │
        │ …        │
        └──────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> import shapely
        >>> t = ibis.examples.zones.fetch()

        Polygon area center in zone 1

        >>> z1_ctr_buff = shapely.Point(935996.821, 191376.75).buffer(10)
        >>> z1_ctr_buff_lit = ibis.literal(z1_ctr_buff, "geometry")
        >>> t.geom.covers(z1_ctr_buff_lit).name("covers")
        ┏━━━━━━━━━┓
        ┃ covers  ┃
        ┡━━━━━━━━━┩
        │ boolean │
        ├─────────┤
        │ True    │
        │ False   │
        │ False   │
        │ False   │
        │ False   │
        │ False   │
        │ False   │
        │ False   │
        │ False   │
        │ False   │
        │ …       │
        └─────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> import shapely
        >>> t = ibis.examples.zones.fetch()

        Polygon area center in zone 1

        >>> pol_big = shapely.Point(935996.821, 191376.75).buffer(10000)
        >>> pol_big_lit = ibis.literal(pol_big, "geometry")
        >>> t.geom.covered_by(pol_big_lit).name("covered_by")
        ┏━━━━━━━━━━━━┓
        ┃ covered_by ┃
        ┡━━━━━━━━━━━━┩
        │ boolean    │
        ├────────────┤
        │ True       │
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ …          │
        └────────────┘
        >>> pol_small = shapely.Point(935996.821, 191376.75).buffer(100)
        >>> pol_small_lit = ibis.literal(pol_small, "geometry")
        >>> t.geom.covered_by(pol_small_lit).name("covered_by")
        ┏━━━━━━━━━━━━┓
        ┃ covered_by ┃
        ┡━━━━━━━━━━━━┩
        │ boolean    │
        ├────────────┤
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ …          │
        └────────────┘
        """
        return ops.GeoCoveredBy(self, right).to_expr()

    def crosses(self, right: GeoSpatialValue) -> ir.BooleanValue:
        """Check if the geometries have at least one, but not all, interior points in common.

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        BooleanValue
            Whether `self` and `right` have at least one common interior point.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> import shapely
        >>> t = ibis.examples.zones.fetch()

        Line from center of zone 1 to center of zone 2

        >>> line = shapely.LineString([[935996.821, 191376.75], [1031085.719, 164018.754]])
        >>> line_lit = ibis.literal(line, "geometry")
        >>> t.geom.crosses(line_lit).name("crosses")
        ┏━━━━━━━━━┓
        ┃ crosses ┃
        ┡━━━━━━━━━┩
        │ boolean │
        ├─────────┤
        │ True    │
        │ True    │
        │ False   │
        │ False   │
        │ False   │
        │ False   │
        │ False   │
        │ False   │
        │ False   │
        │ False   │
        │ …       │
        └─────────┘
        >>> t.filter(t.geom.crosses(line_lit))[["zone", "LocationID"]]
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
        ┃ zone                   ┃ LocationID ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
        │ string                 │ int32      │
        ├────────────────────────┼────────────┤
        │ Newark Airport         │          1 │
        │ Jamaica Bay            │          2 │
        │ Canarsie               │         39 │
        │ East Flatbush/Farragut │         71 │
        │ Erasmus                │         85 │
        │ Flatbush/Ditmas Park   │         89 │
        │ Flatlands              │         91 │
        │ Green-Wood Cemetery    │        111 │
        │ Sunset Park West       │        228 │
        │ Windsor Terrace        │        257 │
        └────────────────────────┴────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> import shapely
        >>> t = ibis.examples.zones.fetch()
        >>> p = shapely.Point(935996.821, 191376.75)  # zone 1 centroid
        >>> plit = ibis.literal(p, "geometry")
        >>> t.geom.disjoint(plit).name("disjoint")
        ┏━━━━━━━━━━┓
        ┃ disjoint ┃
        ┡━━━━━━━━━━┩
        │ boolean  │
        ├──────────┤
        │ False    │
        │ True     │
        │ True     │
        │ True     │
        │ True     │
        │ True     │
        │ True     │
        │ True     │
        │ True     │
        │ True     │
        │ …        │
        └──────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> import shapely
        >>> t = ibis.examples.zones.fetch()
        >>> penn_station = shapely.Point(986345.399, 211974.446)
        >>> penn_lit = ibis.literal(penn_station, "geometry")

        Check zones within 1000ft of Penn Station centroid

        >>> t.geom.d_within(penn_lit, 1000).name("d_within_1000")
        ┏━━━━━━━━━━━━━━━┓
        ┃ d_within_1000 ┃
        ┡━━━━━━━━━━━━━━━┩
        │ boolean       │
        ├───────────────┤
        │ False         │
        │ False         │
        │ False         │
        │ False         │
        │ False         │
        │ False         │
        │ False         │
        │ False         │
        │ False         │
        │ False         │
        │ …             │
        └───────────────┘
        >>> t.filter(t.geom.d_within(penn_lit, 1000))[["zone"]]
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ zone                         ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                       │
        ├──────────────────────────────┤
        │ East Chelsea                 │
        │ Midtown South                │
        │ Penn Station/Madison Sq West │
        └──────────────────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.zones.fetch()
        >>> t.geom.geo_equals(t.geom)
        ┏━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ GeoEquals(geom, geom) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean               │
        ├───────────────────────┤
        │ True                  │
        │ True                  │
        │ True                  │
        │ True                  │
        │ True                  │
        │ True                  │
        │ True                  │
        │ True                  │
        │ True                  │
        │ True                  │
        │ …                     │
        └───────────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.zones.fetch()
        >>> t.geom.geometry_type()
        ┏━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ GeoGeometryType(geom) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                │
        ├───────────────────────┤
        │ POLYGON               │
        │ MULTIPOLYGON          │
        │ POLYGON               │
        │ POLYGON               │
        │ POLYGON               │
        │ POLYGON               │
        │ POLYGON               │
        │ POLYGON               │
        │ POLYGON               │
        │ POLYGON               │
        │ …                     │
        └───────────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> import shapely
        >>> t = ibis.examples.zones.fetch()
        >>> p = shapely.Point(935996.821, 191376.75)  # zone 1 centroid
        >>> plit = ibis.literal(p, "geometry")
        >>> t.geom.intersects(plit).name("intersects")
        ┏━━━━━━━━━━━━┓
        ┃ intersects ┃
        ┡━━━━━━━━━━━━┩
        │ boolean    │
        ├────────────┤
        │ True       │
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ False      │
        │ …          │
        └────────────┘
        """
        return ops.GeoIntersects(self, right).to_expr()

    def is_valid(self) -> ir.BooleanValue:
        """Check if the geometry is valid.

        Returns
        -------
        BooleanValue
            Whether `self` is valid

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.zones.fetch()
        >>> t.geom.is_valid()
        ┏━━━━━━━━━━━━━━━━━━┓
        ┃ GeoIsValid(geom) ┃
        ┡━━━━━━━━━━━━━━━━━━┩
        │ boolean          │
        ├──────────────────┤
        │ True             │
        │ True             │
        │ True             │
        │ True             │
        │ True             │
        │ True             │
        │ True             │
        │ True             │
        │ True             │
        │ True             │
        │ …                │
        └──────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> import shapely
        >>> t = ibis.examples.zones.fetch()

        Polygon center in an edge point of zone 1

        >>> p_edge_buffer = shapely.Point(933100.918, 192536.086).buffer(100)
        >>> buff_lit = ibis.literal(p_edge_buffer, "geometry")
        >>> t.geom.overlaps(buff_lit).name("overlaps")
        ┏━━━━━━━━━━┓
        ┃ overlaps ┃
        ┡━━━━━━━━━━┩
        │ boolean  │
        ├──────────┤
        │ True     │
        │ False    │
        │ False    │
        │ False    │
        │ False    │
        │ False    │
        │ False    │
        │ False    │
        │ False    │
        │ False    │
        │ …        │
        └──────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> import shapely
        >>> t = ibis.examples.zones.fetch()

        Edge point of zone 1

        >>> p_edge = shapely.Point(933100.9183527103, 192536.08569720192)
        >>> p_edge_lit = ibis.literal(p_edge, "geometry")
        >>> t.geom.touches(p_edge_lit).name("touches")
        ┏━━━━━━━━━┓
        ┃ touches ┃
        ┡━━━━━━━━━┩
        │ boolean │
        ├─────────┤
        │ True    │
        │ False   │
        │ False   │
        │ False   │
        │ False   │
        │ False   │
        │ False   │
        │ False   │
        │ False   │
        │ False   │
        │ …       │
        └─────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> import shapely
        >>> t = ibis.examples.zones.fetch()

        Penn station zone centroid

        >>> penn_station = shapely.Point(986345.399, 211974.446)
        >>> penn_lit = ibis.literal(penn_station, "geometry")
        >>> t.geom.distance(penn_lit).name("distance_penn")
        ┏━━━━━━━━━━━━━━━┓
        ┃ distance_penn ┃
        ┡━━━━━━━━━━━━━━━┩
        │ float64       │
        ├───────────────┤
        │  47224.139856 │
        │  55992.665470 │
        │  54850.880098 │
        │   8011.846870 │
        │  84371.995209 │
        │  54196.904809 │
        │  15965.509896 │
        │  20566.442476 │
        │  54070.543584 │
        │  56994.826531 │
        │             … │
        └───────────────┘
        """
        return ops.GeoDistance(self, right).to_expr()

    def length(self) -> ir.FloatingValue:
        """Compute the length of a geospatial expression.

        Returns zero for polygons.

        Returns
        -------
        FloatingValue
            Length of `self`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> con = ibis.get_backend()
        >>> con.load_extension("spatial")
        >>> import shapely
        >>> line = shapely.LineString([[0, 0], [1, 0], [1, 1]])
        >>> line_lit = ibis.literal(line, type="geometry")
        >>> line_lit.length()
        2.0
        >>> t = ibis.examples.zones.fetch()
        >>> t.geom.length()
        ┏━━━━━━━━━━━━━━━━━┓
        ┃ GeoLength(geom) ┃
        ┡━━━━━━━━━━━━━━━━━┩
        │ float64         │
        ├─────────────────┤
        │             0.0 │
        │             0.0 │
        │             0.0 │
        │             0.0 │
        │             0.0 │
        │             0.0 │
        │             0.0 │
        │             0.0 │
        │             0.0 │
        │             0.0 │
        │               … │
        └─────────────────┘
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

        Parameters
        ----------
        right
            Right geometry

        Returns
        -------
        GeoSpatialValue
            Union of geometries

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> import shapely
        >>> t = ibis.examples.zones.fetch()

        Penn station zone centroid

        >>> penn_station = shapely.Point(986345.399, 211974.446)
        >>> penn_lit = ibis.literal(penn_station, "geometry")
        >>> t.geom.centroid().union(penn_lit).name("union_centroid_penn")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ union_centroid_penn                                          ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ geospatial:geometry                                          │
        ├──────────────────────────────────────────────────────────────┤
        │ <MULTIPOINT (935996.821 191376.75, 986345.399 211974.446)>   │
        │ <MULTIPOINT (1031085.719 164018.754, 986345.399 211974.446)> │
        │ <MULTIPOINT (1026452.617 254265.479, 986345.399 211974.446)> │
        │ <MULTIPOINT (990633.981 202959.782, 986345.399 211974.446)>  │
        │ <MULTIPOINT (931871.37 140681.351, 986345.399 211974.446)>   │
        │ <MULTIPOINT (964319.735 157998.936, 986345.399 211974.446)>  │
        │ <MULTIPOINT (1006496.679 216719.218, 986345.399 211974.446)> │
        │ <MULTIPOINT (1005551.571 222936.088, 986345.399 211974.446)> │
        │ <MULTIPOINT (1043002.677 212969.849, 986345.399 211974.446)> │
        │ <MULTIPOINT (1042223.605 186706.496, 986345.399 211974.446)> │
        │ …                                                            │
        └──────────────────────────────────────────────────────────────┘
        """
        return ops.GeoUnion(self, right).to_expr()

    def x(self) -> ir.FloatingValue:
        """Return the X coordinate of `self`, or NULL if not available.

        Input must be a point.

        Returns
        -------
        FloatingValue
            X coordinate of `self`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.zones.fetch()
        >>> t.geom.centroid().x()
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ GeoX(GeoCentroid(geom)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ float64                 │
        ├─────────────────────────┤
        │            9.359968e+05 │
        │            1.031086e+06 │
        │            1.026453e+06 │
        │            9.906340e+05 │
        │            9.318714e+05 │
        │            9.643197e+05 │
        │            1.006497e+06 │
        │            1.005552e+06 │
        │            1.043003e+06 │
        │            1.042224e+06 │
        │                       … │
        └─────────────────────────┘
        """
        return ops.GeoX(self).to_expr()

    def y(self) -> ir.FloatingValue:
        """Return the Y coordinate of `self`, or NULL if not available.

        Input must be a point.

        Returns
        -------
        FloatingValue
            Y coordinate of `self`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.zones.fetch()
        >>> t.geom.centroid().y()
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ GeoY(GeoCentroid(geom)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ float64                 │
        ├─────────────────────────┤
        │           191376.749531 │
        │           164018.754403 │
        │           254265.478659 │
        │           202959.782391 │
        │           140681.351376 │
        │           157998.935612 │
        │           216719.218169 │
        │           222936.087552 │
        │           212969.849014 │
        │           186706.496469 │
        │                       … │
        └─────────────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> con = ibis.get_backend()
        >>> con.load_extension("spatial")
        >>> import shapely
        >>> line = shapely.LineString([[0, 0], [1, 0], [1, 1]])
        >>> line_lit = ibis.literal(line, type="geometry")
        >>> line_lit.start_point()
        <POINT (0 0)>
        """
        return ops.GeoStartPoint(self).to_expr()

    def end_point(self) -> PointValue:
        """Return the last point of a `LINESTRING` geometry as a `POINT`.

        Return `NULL` if the input parameter is not a `LINESTRING`

        Returns
        -------
        PointValue
            End point

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> con = ibis.get_backend()
        >>> con.load_extension("spatial")
        >>> import shapely
        >>> line = shapely.LineString([[0, 0], [1, 0], [1, 1]])
        >>> line_lit = ibis.literal(line, type="geometry")
        >>> line_lit.end_point()
        <POINT (1 1)>
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.zones.fetch()
        >>> t.geom.n_points()
        ┏━━━━━━━━━━━━━━━━━━┓
        ┃ GeoNPoints(geom) ┃
        ┡━━━━━━━━━━━━━━━━━━┩
        │ int64            │
        ├──────────────────┤
        │              232 │
        │             2954 │
        │              121 │
        │               88 │
        │              170 │
        │              277 │
        │              182 │
        │               40 │
        │              189 │
        │              157 │
        │                … │
        └──────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.zones.fetch()
        >>> p = t.x_cent.point(t.y_cent)
        >>> p.buffer(10)  # note buff.area.mean() ~ pi * r^2
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ GeoBuffer(GeoPoint(x_cent, y_cent), 10.0)                                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ geospatial:geometry                                                          │
        ├──────────────────────────────────────────────────────────────────────────────┤
        │ <POLYGON ((936006.821 191376.75, 936006.629 191374.799, 936006.06            │
        │ 191372.923...>                                                               │
        │ <POLYGON ((1031095.719 164018.754, 1031095.526 164016.803, 1031094.957       │
        │ 16401...>                                                                    │
        │ <POLYGON ((1026462.617 254265.479, 1026462.425 254263.528, 1026461.856       │
        │ 25426...>                                                                    │
        │ <POLYGON ((990643.981 202959.782, 990643.788 202957.831, 990643.219          │
        │ 202955.9...>                                                                 │
        │ <POLYGON ((931881.37 140681.351, 931881.178 140679.4, 931880.609             │
        │ 140677.525,...>                                                              │
        │ <POLYGON ((964329.735 157998.936, 964329.543 157996.985, 964328.974          │
        │ 157995.1...>                                                                 │
        │ <POLYGON ((1006506.679 216719.218, 1006506.487 216717.267, 1006505.918       │
        │ 21671...>                                                                    │
        │ <POLYGON ((1005561.571 222936.088, 1005561.379 222934.137, 1005560.81        │
        │ 222932...>                                                                   │
        │ <POLYGON ((1043012.677 212969.849, 1043012.485 212967.898, 1043011.916       │
        │ 21296...>                                                                    │
        │ <POLYGON ((1042233.605 186706.496, 1042233.413 186704.546, 1042232.844       │
        │ 18670...>                                                                    │
        │ …                                                                            │
        └──────────────────────────────────────────────────────────────────────────────┘
        """
        return ops.GeoBuffer(self, radius=radius).to_expr()

    def centroid(self) -> PointValue:
        """Returns the centroid of the geometry.

        Returns
        -------
        PointValue
            The centroid

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.zones.fetch()
        >>> t.geom.centroid()
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ GeoCentroid(geom)                ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ point                            │
        ├──────────────────────────────────┤
        │ <POINT (935996.821 191376.75)>   │
        │ <POINT (1031085.719 164018.754)> │
        │ <POINT (1026452.617 254265.479)> │
        │ <POINT (990633.981 202959.782)>  │
        │ <POINT (931871.37 140681.351)>   │
        │ <POINT (964319.735 157998.936)>  │
        │ <POINT (1006496.679 216719.218)> │
        │ <POINT (1005551.571 222936.088)> │
        │ <POINT (1043002.677 212969.849)> │
        │ <POINT (1042223.605 186706.496)> │
        │ …                                │
        └──────────────────────────────────┘
        """
        return ops.GeoCentroid(self).to_expr()

    def envelope(self) -> ir.PolygonValue:
        """Returns a geometry representing the bounding box of `self`.

        Returns
        -------
        PolygonValue
            A polygon

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.zones.fetch()
        >>> t.geom.envelope()
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ GeoEnvelope(geom)                                                            ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ polygon                                                                      │
        ├──────────────────────────────────────────────────────────────────────────────┤
        │ <POLYGON ((931553.491 183788.05, 941810.009 183788.05, 941810.009            │
        │ 197256.211...>                                                               │
        │ <POLYGON ((1018052.359 151230.391, 1049019.806 151230.391, 1049019.806       │
        │ 17247...>                                                                    │
        │ <POLYGON ((1022540.66 251648.652, 1031732.164 251648.652, 1031732.164        │
        │ 257811...>                                                                   │
        │ <POLYGON ((988733.885 201170.618, 992114.154 201170.618, 992114.154          │
        │ 205029.4...>                                                                 │
        │ <POLYGON ((927766.539 134554.848, 936499.881 134554.848, 936499.881          │
        │ 145354.2...>                                                                 │
        │ <POLYGON ((958451.703 153868.426, 969746.332 153868.426, 969746.332          │
        │ 161198.9...>                                                                 │
        │ <POLYGON ((1001260.812 213309.301, 1011389.066 213309.301, 1011389.066       │
        │ 21982...>                                                                    │
        │ <POLYGON ((1004114.386 221499.117, 1006968.78 221499.117, 1006968.78         │
        │ 224422....>                                                                  │
        │ <POLYGON ((1040414.414 208318.344, 1046392.971 208318.344, 1046392.971       │
        │ 22002...>                                                                    │
        │ <POLYGON ((1038120.442 182170.756, 1047431.161 182170.756, 1047431.161       │
        │ 19073...>                                                                    │
        │ …                                                                            │
        └──────────────────────────────────────────────────────────────────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> import shapely
        >>> t = ibis.examples.zones.fetch()
        >>> penn_station_buff = shapely.Point(986345.399, 211974.446).buffer(5000)
        >>> penn_lit = ibis.literal(penn_station_buff, "geometry")
        >>> t.filter(t.geom.within(penn_lit))["zone"]
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ zone                         ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                       │
        ├──────────────────────────────┤
        │ East Chelsea                 │
        │ Flatiron                     │
        │ Garment District             │
        │ Midtown South                │
        │ Penn Station/Madison Sq West │
        └──────────────────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.zones.fetch()
        >>> t.geom.intersection(t.geom.centroid())
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ GeoIntersection(geom, GeoCentroid(geom)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ geospatial:geometry                      │
        ├──────────────────────────────────────────┤
        │ <POINT (935996.821 191376.75)>           │
        │ <POINT (1031085.719 164018.754)>         │
        │ <POINT (1026452.617 254265.479)>         │
        │ <POINT (990633.981 202959.782)>          │
        │ <POINT (931871.37 140681.351)>           │
        │ <POINT (964319.735 157998.936)>          │
        │ <POINT (1006496.679 216719.218)>         │
        │ <POINT (1005551.571 222936.088)>         │
        │ <POINT (1043002.677 212969.849)>         │
        │ <POINT (1042223.605 186706.496)>         │
        │ …                                        │
        └──────────────────────────────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.zones.fetch()
        >>> t.geom.difference(t.geom.centroid())
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ GeoDifference(geom, GeoCentroid(geom))                                       ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ geospatial:geometry                                                          │
        ├──────────────────────────────────────────────────────────────────────────────┤
        │ <POLYGON ((933091.011 192572.175, 933088.585 192604.97, 933121.56            │
        │ 192857.382...>                                                               │
        │ <MULTIPOLYGON (((1033439.643 170883.946, 1033473.265 170808.208, 1033504.66  │
        │ ...>                                                                         │
        │ <POLYGON ((1026495.593 256638.616, 1026567.23 256589.859, 1026729.235        │
        │ 256481...>                                                                   │
        │ <POLYGON ((992068.667 203711.502, 992061.716 203711.772, 992049.866          │
        │ 203627.2...>                                                                 │
        │ <POLYGON ((936046.565 144173.418, 936387.922 143967.756, 936481.134          │
        │ 143911.7...>                                                                 │
        │ <POLYGON ((966615.256 158662.292, 966524.882 158822.266, 966153.394          │
        │ 159414.4...>                                                                 │
        │ <POLYGON ((1011049.165 218914.083, 1011117.534 218916.104, 1011186.09        │
        │ 218913...>                                                                   │
        │ <POLYGON ((1005304.898 221499.117, 1004958.187 221747.929, 1004935.368       │
        │ 22176...>                                                                    │
        │ <POLYGON ((1043849.708 216473.163, 1043900.798 216332.234, 1043957.136       │
        │ 21619...>                                                                    │
        │ <POLYGON ((1044612.122 190156.818, 1044849.742 190262.973, 1045120.559       │
        │ 18965...>                                                                    │
        │ …                                                                            │
        └──────────────────────────────────────────────────────────────────────────────┘
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

    def convert(
        self, source: ir.StringValue, target: ir.StringValue | ir.IntegerValue
    ) -> GeoSpatialValue:
        """Transform a geometry into a new SRID (CRS).

        Coordinates are assumed to always be XY (Longitude-Latitude).

        Parameters
        ----------
        source
            CRS/SRID of input geometry
        target
            Target CRS/SRID

        Returns
        -------
        GeoSpatialValue
            Transformed geometry

        See Also
        --------
        [`flip_coordinates`](#ibis.expr.types.geospatial.GeoSpatialValue.flip_coordinates)

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.zones.fetch()

        Data is originally in epsg:2263

        >>> t.geom.convert("EPSG:2263", "EPSG:4326")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ GeoConvert(geom)                                                             ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ geospatial:geometry                                                          │
        ├──────────────────────────────────────────────────────────────────────────────┤
        │ <POLYGON ((-74.184 40.695, -74.184 40.695, -74.184 40.695, -74.184 40.696,   │
        │ -...>                                                                        │
        │ <MULTIPOLYGON (((-73.823 40.639, -73.823 40.636, -73.823 40.635, -73.823     │
        │ 40....>                                                                      │
        │ <POLYGON ((-73.848 40.871, -73.847 40.871, -73.847 40.871, -73.846 40.871,   │
        │ -...>                                                                        │
        │ <POLYGON ((-73.972 40.726, -73.972 40.726, -73.972 40.726, -73.972 40.726,   │
        │ -...>                                                                        │
        │ <POLYGON ((-74.174 40.563, -74.173 40.562, -74.172 40.562, -74.172 40.562,   │
        │ -...>                                                                        │
        │ <POLYGON ((-74.064 40.602, -74.064 40.602, -74.064 40.603, -74.065 40.604,   │
        │ -...>                                                                        │
        │ <POLYGON ((-73.904 40.768, -73.903 40.768, -73.903 40.768, -73.903 40.768,   │
        │ -...>                                                                        │
        │ <POLYGON ((-73.923 40.775, -73.924 40.775, -73.925 40.775, -73.925 40.775,   │
        │ -...>                                                                        │
        │ <POLYGON ((-73.785 40.761, -73.785 40.761, -73.785 40.76, -73.784 40.76,     │
        │ -73...>                                                                      │
        │ <POLYGON ((-73.783 40.69, -73.782 40.688, -73.781 40.689, -73.781 40.687,    │
        │ -7...>                                                                       │
        │ …                                                                            │
        └──────────────────────────────────────────────────────────────────────────────┘
        """
        return ops.GeoConvert(self, source, target).to_expr()

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

    def flip_coordinates(self) -> GeoSpatialValue:
        """Flip coordinates of a geometry so that x = y  and y = x.

        Returns
        -------
        GeoSpatialValue
            New geometry with flipped coordinates

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.zones.fetch()
        >>> t.geom.centroid().flip_coordinates()
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ GeoFlipCoordinates(GeoCentroid(geom)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ geospatial:geometry                   │
        ├───────────────────────────────────────┤
        │ <POINT (191376.75 935996.821)>        │
        │ <POINT (164018.754 1031085.719)>      │
        │ <POINT (254265.479 1026452.617)>      │
        │ <POINT (202959.782 990633.981)>       │
        │ <POINT (140681.351 931871.37)>        │
        │ <POINT (157998.936 964319.735)>       │
        │ <POINT (216719.218 1006496.679)>      │
        │ <POINT (222936.088 1005551.571)>      │
        │ <POINT (212969.849 1043002.677)>      │
        │ <POINT (186706.496 1042223.605)>      │
        │ …                                     │
        └───────────────────────────────────────┘

        """
        return ops.GeoFlipCoordinates(self).to_expr()


@public
class GeoSpatialScalar(NumericScalar, GeoSpatialValue):
    pass


@public
class GeoSpatialColumn(NumericColumn, GeoSpatialValue):
    def unary_union(
        self, where: bool | ir.BooleanValue | None = None
    ) -> ir.GeoSpatialScalar:
        """Aggregate a set of geometries into a union.

        This corresponds to the aggregate version of the union.
        We give it a different name (following the corresponding method
        in GeoPandas) to avoid name conflicts with the non-aggregate version.

        Parameters
        ----------
        where
            Filter expression

        Returns
        -------
        GeoSpatialScalar
            Union of geometries

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.zones.fetch()
        >>> t.geom.unary_union()
        <MULTIPOLYGON (((934491.267 196304.019, 934656.105 196375.819, 934810.948 19...>
        """
        return ops.GeoUnaryUnion(self, where=where).to_expr()


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
