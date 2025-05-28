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

The `Sphere` is an excellent prototype of hoomd-geometry's function: it implements both `Shape`
and `Volume`, and its dimension can be specified with the const generic param `N`. It also
has utility as the return type of the `bounding_sphere` method of the `Shape` trait.

It is important to note that structs in this crate do not have fields for the center of mass:
rather, the additional `Centered` struct provides this functionality via encapsulation.
This simplification allows the same structs and methods to be used in both simulation and computational geometry codes.

The `Cuboid` struct also provides a useful example of the idea of an orientable geometry. While an axis-aligned cuboid has an orientation by definition, there is no need to explicitly store that information in most cases. The `Oriented` wrapper struct can encapsulate cuboids if this functionality is needed.

## Intersections

The `IntersectsAt` trait makes up a significant portion of the code in this crate, and provides a wide variety of methods that allow for the calculation of overlaps between geometric primitives. Most provided methods are an overlap _test_, taking in two geometries, a displacement `Vector`, and a `Rotation` and returning a boolean indicating whether the geometries intersect. Many shapes -- including spheres, oriented cuboids, and tetrahedra -- implement shape-specific overlap checks that are faster than general methods for determining overlaps. An implementation of the Xenocollide collision detection test is included in the `collide[2|3]d` functions, and will function properly for any convex geometry that implements `SupportFn<V: Cartesian<[2|3]>>`. While all methods should provide the same results, more specialized subroutines often provide greater performance than Xenocollide.

The `IntersectsAt` trait includes a helper type that allows implementations to accept either an `&R: Rotation` or `&Option<R: Rotation>`. This allows for much clearer distinction of axis-aligned intersection modes for AABBs, and ensures users do not have to initialize an `Angle` or `Versor` for sphere overlaps.

Note that, although the following code is valid, such an implementation precludes specific, optimized overlap methods. Instead, this method should be implemented for
each `T` to ensure special cases can be handled performantly.
```rust
impl<S: SupportFn<Cartesian<3>>, R: Rotate<Cartesian<3>> + Rotation + Copy, T>
    IntersectsAt<S, Cartesian<3>, R> for T
where
    RotationMatrix<3>: From<R>,
    T: SupportFn<Cartesian<3>>,
{
    /// Determine whether a convex object intersects another shape at some position and orientation.
    #[inline]
    fn intersects_at(&self, other: &S, v_ij: &Cartesian<3>, o_ij: &R) -> bool {
        collide3d(self, other, v_ij, o_ij)
    }
}
```

To implement `IntersectsAt` for concave geometries, subdivide the primitive into convex subsets and apply the appropriate collision detection algorithms. While no examples are currently included with `hoomd-geometry` a `ShapeUnion` struct would be the most natural extension.
