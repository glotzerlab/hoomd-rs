# Release notes

## Next release

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
* `[tutorial]`: Added *"Seeded Self-Assembly"* tutorial (#268).

*Changed:*

* `[benchmarks]`: Adjust benchmark parameters to make accurate comparisons with HOOMD-blue (#260).
* `[hoomd-geometry]`: Store `ConvexPolytope` vertices using an `ArrayVec` so that
  `ConvexPolytope` can now be stored on the stack (#259).
* `[hoomd-geometry]`: Remove unnecessary trait bounds on `IntersectsAt` implementation (#260).

*Changed:*

*Deprecated:*

* `[hoomd-mc]`: Deprecated `Tune::tune`. Use `tune_with_options` (#268).
* `[hoomd-mc]`: Deprecated `Tune::tune_default`. Use `tune_with_options(..., &TuneOptions::default())` (#268).

*Removed:*

*Fixed:*

## 1.0.2 (2026-03-20)

*Fixed:*

* Trusted publishing workflow.

## 1.0.1 (2026-03-20)

*Fixed:*

* `hoomd-microstate` documentation builds on docs.rs (#253).
* Unit tests pass on the Windows native platform (#253).

## 1.0.0 (2026-03-17)

*Initial release*.
