# Release notes

## Next release

*Added:*
Adds new shapes, `Hyperparallelepiped`, `Triclinic`, and `Rhomboid` for use in simulations with sheared boxes. All shapes can be used as boundary conditions for simulations, although `Triclinic`, and `Rhomboid` should be preferred for simulations in 2D/3D.

As a helper method for inverting matrices in a numerically stable way, `qr` has also been added as a method to the `matrix` crate to calculate the QR factorization of a tall matrix.
*Changed:*

*Deprecated:*

*Removed:*

*Fixed:*

## 1.1.0 (2026-04-17)

*Highlights:*

**hoomd-rs** 1.1 adds a new shape type, `ConvexSurfaceMesh2d`. Provide a set
of points and `ConvexSurfaceMesh2d` will construct the vertices and edges
of the convex hull. Intersection tests between two `ConvexSurfaceMesh2d`
shapes take approximately half the time of intersection tests between
two `Convex(ConvexPolygon)` shapes. Therefore, you should prefer
`ConvexSurfaceMesh2d` for simulations of hard convex polygons. Use
`ConvexPolygon` when you have mixed shape types or are modeling spheropolygons.
`ConvexSurfaceMesh2d` implements `Volume` and `IsPointInside`, making it
viable for use as a closed boundary condition.

**hoomd-rs** 1.1 also adds `apply_with_filter` and related methods to `Sweep`.
Use `apply_with_filter` to model systems where some bodies remain fixed in
space. The new *Seeded Self-Assembly* tutorial demonstrates `apply_with_filter`.

*Added:*

* `[hoomd-geometry]`: Add `ConvexSurfaceMesh2d` shape that stores the vertices and
  edges of a convex polygon. Initialize a `ConvexSurfaceMesh2d` as the convex hull
  of a point set (#259).
* `[hoomd-geometry]`: Implement `IntersectsAt` for `ConvexSurfaceMesh2d` intersection tests.
  The separating planes method is faster for small *n* than the Xenocollide
  algorithm implemented for `Convex<ConvexPolygon>` (#260).
* `[hoomd-mc]`: Add `TuneOptions` type that describes move size tuning options (#268).
* `[hoomd-mc]`: Added `Tune::tune_with_options` associated method that tunes trial move
  sizes with options passed via `TuneOptions` (#268).
* `[hoomd-mc]`: Added `Sweep::apply_with_filter` associated method that applies trial
  moves only to bodies that match a filter. For example, use `apply_with_filter`
  to keep a crystal seed fixed during the simulation (#268).
* `[hoomd-mc]`: Added `Sweep::tune_with_options_and_filter` associated method that
  tunes trial move sizes while only applying trial moves to bodies that match
  a filter. Use `tune_with_options_and_filter` with the same filter given to
  `apply_with_filter` to accurately tune move sizes (#268).
* `[tutorial]`: Added *Seeded Self-Assembly* tutorial (#268).

*Changed:*

* `[benchmarks]`: Adjust benchmark parameters to make accurate comparisons with HOOMD-blue (#260).
* `[hoomd-geometry]`: Store `ConvexPolytope` vertices using an `ArrayVec` so that
  `ConvexPolytope` can now be stored on the stack (#259).
* `[hoomd-geometry]`: Remove unnecessary trait bounds on `IntersectsAt` implementation (#260).

*Deprecated:*

* `[hoomd-mc]`: Deprecated `Tune::tune`. Use `tune_with_options` (#268).
* `[hoomd-mc]`: Deprecated `Tune::tune_default`. Use `tune_with_options(..., &TuneOptions::default())` (#268).

## 1.0.2 (2026-03-20)

*Fixed:*

* Trusted publishing workflow.

## 1.0.1 (2026-03-20)

*Fixed:*

* `hoomd-microstate` documentation builds on docs.rs (#253).
* Unit tests pass on the Windows native platform (#253).

## 1.0.0 (2026-03-17)

*Initial release*.
