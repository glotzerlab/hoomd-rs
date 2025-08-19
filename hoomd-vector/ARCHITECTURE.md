# hoomd_vector

## Vector

The `hoomd_vector` crate defines a generic `Vector` trait that is independent of
representation. The trait consists of methods that can _only_ be applied to all vectors
in a normed vector space with *n* dimensions:

- Vector addition & subtraction
- Multiplication by a scalar
- Dot product
- Norm and norm squared

This design allows the majority of hoomd-rs code to be written _independent_ of the
vector's representation and dimension. Some specific calculations may require
cross products, defined in specific trait: `Cross`.

### Cartesian vector

`hoomd_vector` implements an n-dimension `Cartesian` vector type for general use,
which includes methods for element access, and other operations specific to Cartesian
vectors.

## Rotations

The `Rotation` trait describes types that represent a given rotation operation.
`Rotations` can be _combined_ to chain their operation and there always exists an
**identity** rotation. The `Rotate` trait applies to types that can apply a rotation
operation to a vector.

### Angle

The `Angle` type rotates 2D Cartesian vectors.

### Versor

The `Versor` type, a unit `Quaternion`, rotates 3D Cartesian vectors.

## Random sampling

`hoomd_vector` implements [`rand`] distributions to sample random vectors and
rotations.

[`rand`]: https://docs.rs/rand/latest/rand/
