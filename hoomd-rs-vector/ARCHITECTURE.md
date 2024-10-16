# hoomd_rs_vector

## Vector

The `hoomd_rs_vector` crate defines a generic `Vector` trait that is independent of
representation. The trait consists of methods that can _only_ be applied to all vectors
in a vector space with *n* dimensions:

- Vector addition & subtraction
- Multiplication by a scalar
- Dot product
- Length & length squared

This design allows the majority of HOOMD-rs code to be written _independent_ of the
vector's representation and dimension. Some specific calculations may require
cross products, defined in specific trait: `Cross`.

## CartesianVector

`hoomd_rs_vector` implements an n-dimension `CartesianVector` type for general use,
which includes methods for element access, element-wise multiplication, and other
operations specific to Cartesian vectors.

## User-defined vectors

Users can implement custom types (e.g. spherical coordinates) that implement `Vector`
as needed. Many internal computations inside HOOMD-rs rely on Cartesian vectors, so
all user-defined vectors must implement the conversion traits:
```
impl From<CustomVector> for CartesianVector<3> {
...
}
```
and
```
impl From<CartesianVector<3>> for CustomVector {
...
}
```

## Rotate

TODO

## Random sampling

`hoomd_rs_vector` implements [`rand`] distributions to sample random vectors and
rotations.
