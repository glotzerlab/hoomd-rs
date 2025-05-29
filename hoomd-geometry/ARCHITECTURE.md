# hoomd_geometry

The `hoomd_geometry` crate implements types that describe geometric shapes and
traits that operate on them. The initial design prioritizes intersection tests
between pairs of particles for use in hard particle Monte Carlo simulations.
Over time, `hoomd_geometry` will grow with new methods that compute shape
properties and manipulate shapes in ways that researchers need for simulation
an analysis.

As `hoomd_geometry` expands, it cannot lose sight of the primary goal. For
example, the `ConvexPolytope` class should always remain as a set of vertices
(and possibly a precomputed bounding radius) where the shape is implicitly the
convex hull of the vertices. That is the representation needed for fast overlap
checks with Xenocollide. When `hoomd_geometry` gains methods that operate on
faces or edges, those should be implemented on a more general `Mesh` type. Users
can convert a `ConvexPolytope` to a `Mesh` when needed.

## Shapes

`hoomd_geometry` provides a number of types, each of which can describe a shape
in a given class: `Hyperellipsoid` and `ConvexPolytope`, for example. When
possible, shape types are generic on the number of dimensions. Each shape (for
example a `Sphere` with radius `r`) describes both the *surface* of the shape,
and the *set of interior points*.

Each shape is expressed as if it exists in a local space centered on some
natural origin. `hoomd_geometry` makes no specific requirements on that origin
in general, though the `xenocollide` implementation currently requires that the
origin lie inside the shape.

## Traits

It is not straightforward (or desirable) to implement all shape properties for
every single shape type. Therefore, `hoomd_geometry` provides each property
(and operation) as a separate trait that types can implement when possible
and necessary.

* `IntersectsAt`: Determine whether two shapes (possibly of different
  types) intersect given their relative position and orientation. Formally,
  `intersects_at` tests if the set intersection of the two shape's interior
  points is not empty.
* `IsInside`: Test if a point is inside the shape.
* `MinDistance`: Compute the minimum distance (TODO: between two shapes?
  Or between a point and a shape?).
* `SupportMapping`: Find the point in the shape that is furthest in a particular
  direction.
* `Volume`: Compute the hypervolume of the space contained within the shape.

`hoomd_geometry` will add traits over time.

## Intersection algorithms

Shapes may implement `IntersectsAt` via any appropriate algorithm. For
example, `Sphere` intersection tests are trivial. There are also fast `Capsule`
intersection tests based in the literature that we could use.

`IntersectsAt` requires that the shapes be orientable. In some cases (such as
axis aligned boxes), there are extremely fast intersection tests that assume the
shapes are not rotated. These types will implement ad-hoc intersection methods
in their inherent implementations.

## XenoCollide for convex geometry

`hoomd_geometry` implements the XenoCollide algorithm, which can determine
whether any two convex shapes overlap in 2 and 3 dimensions. Most shape types
perform `IntersectsAt` checks using XenoCollide, see code in `shapes` for
examples.

## Concave geometries

Concave geometries are particularly challenging to test for overlaps. There are
several strategies available:
* If possible, users can express a concave geometry as a union of convex
  geometries. Ideally, users would add bodies to the microstate where each site
  is one of the convex shapes in the decomposition. This approach allows the use
  of spatial data structures to minimize the number of shape overlaps checks.
* Alternately, a single shape type could implement the N^2 checks needed in its
  `intersects_at` implementation.
* Not all concave shapes nicely decompose into a set of convex shapes (e.g.
  an arbitrary simple polygon). In such cases, developers must implement an
  appropriately generic algorithm.
