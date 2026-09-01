# Release Notes

## Next release

*Added:*

*Changed:*

`[tutorials]`:  The workflow tutorial now uses `hoomd-workspace` (#403).

*Deprecated:*

*Removed:*

*Fixed:*

## 1.3.0 (2026-08-26)

*Highlights:*

**hoomd-rs 1.3** can now perform simulations in sheared boxes. Use `Rhomboid` for 2D, `Triclinic` for 3D or `Hyperparallelepiped` for N-D simulations. It can also perform Monte Carlo simulations in the grand canonical distribution, replicate periodic microstates, perform simulations with N-D bodies constrained to (N-1)-D manifolds, and interoperate directly with [signac] workspaces.

Check out the many new examples and learn how to use **hoomd-rs** for your simulation models.

[signac]: https://signac.readthedocs.io

*Added:*

* `[examples]`: Add "Melt an ideal hexagonal structure of hexagons" example (#393).
* `[examples]`: Add "Multi-site patchy shapes" example (#395).
* `[examples]`: Add "3D shapes confined to a 2D plane" example (#397).
* `[examples]`: Add "Binary hard shape systems" example (#401).
* `[examples]`: Use the mouse wheel to zoom examples with 3D cameras (#397).
* `[examples]`: Click and drag to orbit the camera in examples with 3D cameras (#397).
* `[hoomd-geometry]`: Add `Hyperparallelepiped`, `Triclinic`, and `Rhomboid` shapes (#88).
* `[hoomd-geometry]`: Implement `Scale` for `ConvexSurfaceMesh2d` (#401).
* `[hoomd-linear-algebra]` Add `matrix::qr` module with methods to compute the QR factorization of matrices (#88).
* `[hoomd-linear-algebra]` Add methods `iter_column_slice`, `iter_column_slice_mut`, `iter_submatrix`, to `Matrix<N, M>` (#88).
* `[hoomd-macrostate]`: Add `Fugacity` trait that access the fugacity of a macrostate (#354).
* `[hoomd-macrostate]`: Add `IsothermoalIsofugacity` type that stores the system's temperature and fugacity (#354).
* `[hoomd-mc]`: Add `GrandCanonical` trial move type. It inserts and removes bodies and samples from the constant chemical potential, constant volume, constant temperature ensemble (#354).
* `[hoomd-mc`]: Add `InsertRemoveCount` type that counts the accepted and rejected insertion and removal moves (#354).
* `[hoomd-microstate`]: Implement all traits necessary to use `Hyperparallelepiped`, `Triclinic`, and `Rhomboid` as simulation boundaries (#88).
* `[hoomd-microstate`]: Implement `AppendMicrostate` for microstates with `Triclinic` and `Rhomboid` boundaries (#88).
* `[hoomd-microstate`]: Add `Microstate::iter_bodies_tag_order` method that iterates over all bodies in tag order (#393).
* `[hoomd-microstate`]: Add `Replicate` trait and implement it for microstates with periodic `Rhomboid`, `Triclinic`, and `Hypercuboid` boundaries.
   Call `microstate.replicate` to repeat a periodic structure N x N (...) x N times (#393).
* `[hoomd-vector`]: Implement `Mul<Cartesian<N>> for f64`, `Mul<PositiveReal> for Cartesian<N>`, `Mul<Cartesian<N>> for PositiveReal`, `MulAssign<f64> for Cartesian<N>`, and `MulAssign<PositiveReal> for Cartesian<N>` (#88).
* `[hoomd-vector`]: Add `Cartesian::basis` method that constructs Cartesian basis vectors (#393).
* `[hoomd-workspace`]: Add the `hoomd-workspace` crate. Use it to initialize and read state points in a signac workspace (#359).

*Changed:*

* Build the documentation with mdBook 0.5.4 and KaTeX 0.18.1 (#376).
* `[hoomd-microstate]`: Relax trait bounds to allow bodie and site positions to
  have different types (#397).
* `[hoomd-mc]`: Relax trait bounds to allow bodie and site positions to
  have different types (#397).

*Fixed:*

* `[examples]`: Properly auto size egui parameter windows (#354).
* `[examples]`: Reduce the zoom speed (#393).
* `[hoomd-interaction]`: Fix typos in documentation (#358).
* `[hoomd-microstate]`: Fix typos in documentation (#358).

## 1.2.0 (2026-07-07)

*Highlights:*

**hoomd-rs** 1.2 can perform molecular dynamics simulations with the new `hoomd-md` crate! The release also expands the Monte Carlo capabilities of high dimensional and curved space simulations.

*Added:*

* `[hoomd-derive`]: Derive macros for `NetSiteForceAndVirial` and `NetSiteForceVirialAndTorque` (#222).
* `[hoomd-geometry]`: Add `TwelveTwelve` hyperbolic space boundary condition (#296).
* `[hoomd-interaction]`: Add `SiteForceAndVirial`, `SiteForceVirialAndTorque`, `NetSiteForceAndVirial`, and `NetSiteForceVirialAndTorque` traits. Implement them for all applicable types (#222).
* `[hoomd-interaction]`: Add `NetBodyForceAndVirial` and `NetBodyForceVirialAndTorque` traits. Implement them for the new `Rigid` force interaction model type (#222).
* `[hoomd-interaction]`: Add `ConstantForce` and `ConstantTorque` types that apply a constant force or torque to all sites (#222).
* `[hoomd-manifold]`: Add `Spherical<4>::from_versor` and the corresponding `::to_versor` (#285).
* `[hoomd-mc]`: Implement translation moves for `Point<Spherical<4>>` (#287).
* `[hoomd-mc]`: Implement `BodyDistribution` for `DynamicPoint` and `DynamicOrientedPoint` (#222).
* `[hoomd-md]`: Add the `hoomd-md` crate that implements constant volume integration for translational and rotational degrees of freedom, momentum scaling thermostats, and methods to thermalize and zero system momentum (#222).
* `[hoomd-microstate]`: Implement `AppendMicrostate` for the site types `Point<Spherical<3>>`, `Point<Spherical<4>>`, `OrientedHyperbolicPoint<3, Angle>`, and `Point<Hyperbolic<3>>` (#286).
* `[hoomd-microstate]`: Add `Mass`, `Momentum`, `MomentOfInertia`, `AngularMomentum`, `NetForce`, and `NetTorque` traits (#222).
* `[hoomd-microstate`]: Add `DynamicPoint` and `DynamicOrientedPoint` body property types. Implement boundary and transform traits for them (#222).
* `[hoomd-utility]`: Implement `Eq`, `PartialOrd`, and `Ord` for `PositiveReal` (#287).
* `[hoomd-vector]`: Implement `Cartesian<4>::counary_cross` (#305).
* `[hoomd-vector]`: Implement wedge and outer product traits (#222).
* `[hoomd-vector]`: Add `Quaternion::pure` constructor that forms a pure quaternion from a vector (#222).

*Changed:*

* `[hoomd-mc]`: Improve the numerical stability of translation moves for `Point<Spherical<3>>` (#287).
* `[hoomd-geometry`]: Improve performance `ConvexPolytope` intersection tests (#332).
* `[hoomd-mc]`: `HypercuboidCheckerboard` now builds in an arbitrary number of dimensions (#318).
* `[hoomd-vector]`: `n_dimensions` is now an associated method (#222).
* Build the documentation with mdBook 0.5.3 (#295).

*Fixed:*

* `[hoomd-manifold]`: Fixed numerical stability issue in `Spherical<3>::distance` where the dot product could result in an out of bounds value (#285).
* `[hoomd-manifold]`: Improved numerical stability of Hyperbolic space (#303).

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
