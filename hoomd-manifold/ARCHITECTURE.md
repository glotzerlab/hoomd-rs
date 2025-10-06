# hoomd_manifold

## Curved Manifold

The `hoomd_manifold` crate defines and implements methods for non-Euclidean manifolds embedded in metric vector spaces. The manifolds themselves must have trait `Metric`.

## Sphere

`hoomd_manifold` includes the struct `Sphere` for implementing an embedding of a sphere in cartesian space.

## Hyperbolic

`hoomd_manifold` includes the structs `Minkowski` and `Hyperbolic` to implement the Hyperbolic model of hyperbolic space.

## Rotations in Hyperbolic Space

Specific representations of SO(2,1) and SO(3,1) are implemented in `HyperbolicAngle` and `Biquaternion`, respectively. Analogous to the Eucclidean group E(n) for cartesian space, SO(n,1) is the group of isometries for n-dimensional hyperbolic space.
