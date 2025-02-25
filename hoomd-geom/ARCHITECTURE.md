# Notes

## Traits

The `Shape` trait provides basic, general methods that are useful for most geometries and
applications. Bounding spheres (not necessarily minimal) are a good example, as they are
well defined for arbitrary geometries and a reasonably tight bounding radius is simple
to determine for most shapes.

The `Volume` trait provides a notion of n-hypervolume for `Shape`s or other structs: however,
it is kept as a separate trait to allow for the creation of abstract & less well-defined
shapes like self-intersecting polyhedra or infinite geometries.

## Structs

The `Sphere` is an excellent prototype of HOOMD-geom's function: it implements both `Shape`
and `Volume`, and its dimension can be specified with the const generic param `N`. It also
has utility as the return type of the `bounding_sphere` method of the `Shape` trait.

It is important to note that structs in this crate do not have fields for the center of mass:
rather, the additional `Centered` struct provides this functionality via encapsulation TODO.
This simplification allows the same structs and methods to be used in both simulation and computational geometry codes.

The `Cuboid` struct also provides a useful example of the idea of an orientable geometry. While an axis-aligned cuboid has an orientation by definition, there is no need to explicitly store that information in most cases. The `Oriented` wrapper struct can encapsulate cuboids if this functionality is needed

## Intersections

TODO
