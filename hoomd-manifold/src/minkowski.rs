// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement vector types in Minkowski space.
 */

use std::array;
use std::fmt;
use std::iter::zip;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};
use libm::acosh;
use hoomd_vector::Vector;

use crate::{Error,Hyperboloid};

/** Description of Minkowski space, examples of usage
*/

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Minkowski<const N: usize> {
    // The vector's coordinates
    pub coordinates: [f64; N],
}

impl<const N: usize> Default for Minkowski<N> {
    /** Create a 0 vector in Minkowski space.

    # Example
    ```
    use hoomd_manifold::Minkowski;

    let v = Minkowski::<3>::default();
    assert_eq!(v, [0.0; 3].into())
    ```
    */
    #[inline]
    fn default() -> Self {
        Minkowski::from([0.0; N])
    }
}

impl<const N: usize> From<[f64; N]> for Minkowski<N> {
    /** Create a vector in Minkowski space with the given coordinates. Note that
    the last component has a (-) signature, while the preceeding coordinates have
     (+) signatures in the metric. 

    # Example
    ```
    use hoomd_manifold::Minkowski;

    let v = Minkowski::from([1.0, 2.0]);
    ```
    */
    #[inline]
    fn from(coordinates: [f64; N]) -> Self {
        Self { coordinates }
    }
}

impl From<(f64, f64)> for Minkowski<2> {
    #[inline]
    fn from(coordinates: (f64, f64)) -> Self {
        Self {
            coordinates: coordinates.into(),
        }
    }
}

impl From<(f64, f64, f64)> for Minkowski<3> {
    #[inline]
    fn from(coordinates: (f64, f64, f64)) -> Self {
        Self {
            coordinates: coordinates.into(),
        }
    }
}

impl<const N: usize> TryFrom<Vec<f64>> for Minkowski<N> {
    type Error = Error;

    /** Create a vector in Minkowski with coordinates given by a [`Vec<f64>`]

    # Example
    ```
    use hoomd_manifold::Minkowski;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v = Minkowski::<3>::try_from(vec![5.0, 4.0, 3.0])?;
    assert_eq!(v, [5.0, 4.0, 3.0].into());
    # Ok(())
    # }
    ```
    <div class="warning">

    Use `Minkowski::From<[f64; N]>` in performance critical code.

    </div>
    */
    #[inline]
    fn try_from(value: Vec<f64>) -> Result<Self, Self::Error> {
        let coordinates = value.try_into().map_err(|_| Error::InvalidVectorLength)?;
        Ok(Self { coordinates })
    }
}

impl<const N: usize> TryFrom<std::ops::Range<usize>> for Minkowski<N> {
    type Error = Error;

    /** Create a vector in Minkowski space with coordinates given by a range.

    # Example
    ```
    use hoomd_manifold::Minkowski;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v = Minkowski::<3>::try_from(1..4)?;
    assert_eq!(v, [1.0, 2.0, 3.0].into());
    # Ok(())
    # }
    ```

    <div class="warning">

    Use `Minkowski::From<[f64; N]>` in performance critical code.

    </div>
    */
    #[inline]
    fn try_from(value: std::ops::Range<usize>) -> Result<Self, Self::Error> {
        if value.len() != N {
            return Err(Error::InvalidVectorLength);
        }

        // The default value of 0 will never be used due to the above error check.
        let mut iter = value;
        let coordinates = array::from_fn(|_| iter.next().unwrap_or(0) as f64);
        Ok(Self { coordinates })
    }
}

impl<const N: usize> Vector for Minkowski<N> {
    /** Computes the squared distance between two points in Minkowski space with 
    "mostly pluses" metric signature. 
    ```math
    d^2_M(\vec{x},\vec{y}) = -(x_N-y_N)^2 + \sum_{i=1}^{N-1} (x_i - y_i)^2
    ```

    # Example
    ```
    use hoomd_manifold::Minkowski;
    use hoomd_vector::Vector;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Minkowski::from([0.0, 2.0, 3.0]);
    let y = Minkowski::from([1.0, 0.0, 0.0]);
    assert_eq!(-4.0, x.distance_squared(&y));
    # Ok(())
    # }
    */
    #[inline]
    fn distance_squared(&self, other: &Self) -> f64 {
        let last_component = -1.0 * (self.coordinates[N-1] - other.coordinates[N-1]).powi(2);
        zip(self.coordinates[0..N-1].iter(), other.coordinates[0..N-1].iter())
            .fold(last_component, |product, x| product + (x.0 - x.1).powi(2))
    }
}

impl<const N: usize> Add for Minkowski<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut coordinates = [0.0; N];

        for (result, (a, b)) in coordinates
            .iter_mut()
            .zip(self.coordinates.iter().zip(rhs.coordinates.iter()))
        {
            *result = a + b;
        }
        Self { coordinates }
    }
}

impl<const N: usize> AddAssign for Minkowski<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for (result, a) in self.coordinates.iter_mut().zip(rhs.coordinates.iter()) {
            *result += a;
        }
    }
}

impl<const N: usize> Div<f64> for Minkowski<N> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self {
        let mut coordinates = [0.0; N];

        for (result, a) in coordinates.iter_mut().zip(self.coordinates) {
            *result = a / rhs;
        }
        Self { coordinates }
    }
}

impl<const N: usize> DivAssign<f64> for Minkowski<N> {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        for result in &mut self.coordinates {
            *result /= rhs;
        }
    }
}

impl<const N: usize> Mul<f64> for Minkowski<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        let mut coordinates = [0.0; N];
        for (result, a) in coordinates.iter_mut().zip(self.coordinates) {
            *result = a * rhs;
        }
        Self { coordinates }
    }
}

impl<const N: usize> MulAssign<f64> for Minkowski<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        for result in &mut self.coordinates {
            *result *= rhs;
        }
    }
}

impl<const N: usize> Sub for Minkowski<N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut coordinates = [0.0; N];
        for (result, (a, b)) in coordinates
            .iter_mut()
            .zip(self.coordinates.iter().zip(rhs.coordinates.iter()))
        {
            *result = a - b;
        }
        Self { coordinates }
    }
}

impl<const N: usize> SubAssign for Minkowski<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for (result, a) in self.coordinates.iter_mut().zip(rhs.coordinates) {
            *result -= a;
        }
    }
}

impl<const N: usize> fmt::Display for Minkowski<N> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.coordinates
                .iter()
                .map(f64::to_string)
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

impl<const N: usize> Neg for Minkowski<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        let mut result = self;
        result.coordinates.iter_mut().for_each(|x| *x = -*x);
        result
    }
}

impl<const N: usize, T> Index<T> for Minkowski<N>
where
    T: Into<usize> + std::slice::SliceIndex<[f64], Output = f64>,
{
    type Output = f64;
    /** Get the value of the vector at coordinate i.

    # Example
    ```
    use hoomd_manifold::Minkowski;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v = Minkowski::<3>::try_from(4..7)?;
    assert_eq!((v[0], v[1], v[2]), (4.0, 5.0, 6.0));
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn index(&self, index: T) -> &Self::Output {
        &self.coordinates[index]
    }
}

impl<const N: usize, T> IndexMut<T> for Minkowski<N>
where
    T: Into<usize> + std::slice::SliceIndex<[f64], Output = f64>,
{
    /** Get a mutable reference to the value of the vector at coordinate i.

    # Example
    ```
    use hoomd_manifold::Minkowski;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut v = Minkowski::<3>::try_from(4..7)?;
    assert_eq!((v[0], v[1], v[2]), (4.0, 5.0, 6.0));
    v[0] += 1.0;
    assert_eq!(v[0], 5.0);
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn index_mut(&mut self, index: T) -> &mut Self::Output {
        &mut self.coordinates[index]
    }
}

/** Embedding of the top sheet of a two-sheeted hyperboloid in Minkowski space. This surface has constant
 negative curvature and therefore serves as a model of hyperbolic space.

Explicitly, for three-dimensional Minkowski space with signature (+,+,-), the hyperboloid is defined by 
```math
x^2 + y^2 -z^2 = -1
```
where the z>0 solutions are taken to be the hyperbolic surface. Note that points on the hyperboloid 
are parameterized by two coordinates (v,\theta), where v\in[0,\infty) and \theta\in[0,2\pi) via the 
embedding 
```math
\vec{p} = (\sinh(v)\cos(\theta), \sinh(v)\sin(\theta), \cosh(v))
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hyperbolic<const N: usize> {
    pub coordinates: [f64; N],
}

impl<const N: usize> Hyperboloid for Minkowski<N> {
    /** Computes the length of the geodesic passing between two points. From N-dimensional Minkowski space 
    with signature (+\cdots +-), one can obtain the corresponding Minkowski bilinear form 
    ```math
    B(\vec{u},\vec{v}) = u_1v_1 + \cdots + u_{N-1}v_{N-1} - u_Nv_N
    ``` 
    Now the distance (according to the Minkowski metric) between two points \vec{u} and \vec{v} on the hyperboloid
    is given by 
    ```math
    d_{H^2}(\vec{u},\vec{v}) = \rho\cdot\cosh^{-1}(-\frac{1}{\rho^2}B(\vec{u},\vec{v}))
    ```

    # Example
    ```
    use libm::acosh;
    use hoomd_vector::Vector;
    use hoomd_manifold::{Minkowski, Hyperboloid};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Minkowski::from([0.0, 0.0, 1.0]);
    let y = Minkowski::from([1.0, 0.0, (2.0_f64).sqrt()]);
    let c = acosh((2.0_f64).sqrt());
    assert_eq!(c,x.hyperbolic_distance(&y,1.0));
    # Ok(())
    # }
    ```
     */
    #[inline]
    fn hyperbolic_distance(&self, other: &Self, skirt: f64) -> f64 {
        let last_component = self.coordinates[N-1] * other.coordinates[N-1];
        let arg = zip(self.coordinates[0..N-1].iter(), other.coordinates[0..N-1].iter())
            .fold(last_component, |product, x| product - (x.0 * x.1));
        skirt*acosh(arg/(skirt.powi(2)))
    }
}


/** Rotate Minkowski vectors.

Construct a [`MinkowskiRotationMatrix`] to efficiently rotate many vectors by the same rotation.

See:
* [`RotationMatrix::from<Angle>`]

[`RotationMatrix`] _intentionally_ does not implement [`Rotation`](crate::Rotation).
[`Angle`](crate::Angle) and [`Versor`](crate::Versor) are representations of
rotations that are often the most effective and numerically stable to
manipulate.
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinkowskiRotationMatrix<const N: usize> {
    /// Rows of the rotation matrix.
    pub(crate) rows: [Cartesian<N>; N],
}

impl<const N: usize> Rotate<Minkowski<N>> for MinkowskiRotationMatrix<N> {
    type Matrix = MinkowskiRotationMatrix<N>;

    #[inline]
    /** Rotate a [`Minkowski<N>`] by a [`MinkowskiRotationMatrix`]

    # Examples
    ```
    use hoomd_vector::{Angle, Rotate};
    use hoomd_manifold::{MinkowskiRotationMatrix, Minkowski};
    use std::f64::consts::PI;

    let v = Minkowski::from([1.0, 0.0, (2.0_f64).sqrt()]);
    let a = Angle::from(PI/2.0);

    let matrix = MinkowskiRotationMatrix::from(a);
    let rotated = matrix.rotate(&v);
    // rotated is approximately [0.0, 1.0, (2.0_f64).sqrt()]
    ```
    */
    fn rotate(&self, vector: &Minkowski<N>) -> Minkowski<N> {
        let mut coordinates = [0.0; N];

        for (result, row) in coordinates.iter_mut().zip(self.rows.iter()) {
            *result = row.dot(vector);
        }

        Minkowski { coordinates }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paste::paste;

    // Parameterize a test function over an array of vector lengths
    macro_rules! parameterize_vector_length {
        // macro with name as above that takes an identifier (fn) and an expression
        // $(...),* matches 0 or more expressions (values) separated by commas
        ($test_body:ident, [$($dim:expr),*]) => {

            // Now, we repeat the test block 0 or more times, one for each $dim
            $(
                // paste package combines values in [< >] to form a new ident
                paste! {
                    #[test]
                    fn [< $test_body "_" $dim>]() {
                        const DIM: usize = $dim;
                        $test_body::<DIM>();
                    }
                }
            )*
        };
    }

    /// Generate a pair of length N vectors.
    /// The first vector ranges from [0, N-1] and the second ranges from [N, 2*N-1]
    fn generate_vector_pair<const N: usize>() -> (Minkowski<N>, Minkowski<N>) {
        (
            Minkowski::try_from(0..N).unwrap(),
            Minkowski::try_from(N..N * 2).unwrap(),
        )
    }

    fn index<const N: usize>() {
        let (_, b) = generate_vector_pair::<N>();
        assert!(zip(0..N, b.coordinates.iter()).all(|(i, &x)| b[i] == x));
    }
    parameterize_vector_length!(index, [2, 3, 4, 8, 16, 32]);

    fn index_mut<const N: usize>() {
        let (a, mut b) = generate_vector_pair::<N>();
        zip(0..N, b.coordinates.iter_mut()).for_each(|(i, x)| *x = a[i]);
        assert_eq!(a, b);
    }
    parameterize_vector_length!(index_mut, [2, 3, 4, 8, 16, 32]);

    fn add_explicit<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();
        let c = a.add(b);

        let addition_answer: Vec<f64> = (0..(2 * N))
            .step_by(2)
            .map(|x| (x + N) as f64)
            .collect::<Vec<_>>();

        assert_eq!(c, Minkowski::try_from(addition_answer).unwrap());
    }
    parameterize_vector_length!(add_explicit, [2, 3, 4, 8, 16, 32]);

    fn add_operator<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();
        let c = a + b;

        let addition_answer: Vec<f64> = (0..(2 * N))
            .step_by(2)
            .map(|x| (x + N) as f64)
            .collect::<Vec<_>>();

        assert_eq!(c, Minkowski::try_from(addition_answer).unwrap());
    }
    parameterize_vector_length!(add_operator, [2, 3, 4, 8, 16, 32]);

    fn add_assign<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();
        let mut c = a;
        c += b;

        let addition_answer: Vec<f64> = (0..(2 * N))
            .step_by(2)
            .map(|x| (x + N) as f64)
            .collect::<Vec<_>>();

        assert_eq!(c, Minkowski::try_from(addition_answer).unwrap());
    }

    parameterize_vector_length!(add_assign, [2, 3, 4, 8, 16, 32]);

    fn sub_operator<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();
        let c = a - b;

        let subtraction_answer = [-(N as f64); N];

        assert_eq!(c, subtraction_answer.into());
    }
    parameterize_vector_length!(sub_operator, [2, 3, 4, 8, 16, 32]);

    fn sub_assign<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();
        let mut c = a;
        c -= b;

        let subtraction_answer = [-(N as f64); N];

        assert_eq!(c, subtraction_answer.into());
    }

    parameterize_vector_length!(sub_assign, [2, 3, 4, 8, 16, 32]);

    fn mul_operator<const N: usize>() {
        let (a, _) = generate_vector_pair::<N>();
        let b = 12.0;
        let c = a * b;

        let multiplication_answer: Vec<f64> = (0..N).map(|x| (x as f64) * b).collect::<Vec<_>>();

        assert_eq!(c, Minkowski::try_from(multiplication_answer).unwrap());
    }
    parameterize_vector_length!(mul_operator, [2, 3, 4, 8, 16, 32]);

    fn mul_assign<const N: usize>() {
        let (mut a, _) = generate_vector_pair::<N>();
        let b = 12.0;
        a *= b;

        let multiplication_answer: Vec<f64> = (0..N).map(|x| (x as f64) * b).collect::<Vec<_>>();

        assert_eq!(a, Minkowski::try_from(multiplication_answer).unwrap());
    }

    parameterize_vector_length!(mul_assign, [2, 3, 4, 8, 16, 32]);

    fn div_operator<const N: usize>() {
        let (a, _) = generate_vector_pair::<N>();
        let b = 12.0;
        let c = a / b;

        let division_answer: Vec<f64> = (0..N).map(|x| (x as f64) / b).collect::<Vec<_>>();

        assert_eq!(c, Minkowski::try_from(division_answer).unwrap());
    }
    parameterize_vector_length!(div_operator, [2, 3, 4, 8, 16, 32]);

    fn div_assign<const N: usize>() {
        let (mut a, _) = generate_vector_pair::<N>();
        let b = 12.0;
        a /= b;

        let division_answer: Vec<f64> = (0..N).map(|x| (x as f64) / b).collect::<Vec<_>>();

        assert_eq!(a, Minkowski::try_from(division_answer).unwrap());
    }

    parameterize_vector_length!(div_assign, [2, 3, 4, 8, 16, 32]);

    fn compute_add_ref_ref<const N: usize>(a: &Minkowski<N>, b: &Minkowski<N>) -> Minkowski<N> {
        *a + *b
    }

    fn compute_add_ref_type<const N: usize>(a: &Minkowski<N>, b: Minkowski<N>) -> Minkowski<N> {
        *a + b
    }

    fn compute_add_type_ref<const N: usize>(a: Minkowski<N>, b: &Minkowski<N>) -> Minkowski<N> {
        a + *b
    }

    fn add_with_refs<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();

        let addition_answer = Minkowski::try_from(
            (0..(2 * N))
                .step_by(2)
                .map(|x| (x + N) as f64)
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let c = compute_add_ref_ref(&a, &b);
        assert_eq!(c, addition_answer);

        let c = compute_add_ref_type(&a, b);
        assert_eq!(c, addition_answer);

        let c = compute_add_type_ref(a, &b);
        assert_eq!(c, addition_answer);
    }
    parameterize_vector_length!(add_with_refs, [2, 3, 4, 8, 16, 32]);

    fn neg<const N: usize>() {
        let a: Minkowski<N> = Minkowski::try_from(0..N).unwrap();
        let b = -a;
        for (i, j) in zip(a.coordinates.iter(), b.coordinates.iter()) {
            assert_eq!(*i, -j);
        }
    }
    parameterize_vector_length!(neg, [2, 3, 4, 8, 16, 32]);

    #[test]
    fn display() {
        let a = Minkowski::from([1.5, 2.125, -3.875]);
        let s = format!("{a}");
        assert_eq!(s, "[1.5, 2.125, -3.875]");

        let a = Minkowski::from([10.0, 20.0, 30.0, 40.0]);
        let s = format!("{a}");
        assert_eq!(s, "[10, 20, 30, 40]");
    }

    #[test]
    fn from_2_tuple() {
        let a = Minkowski::from((3.0, 0.125));
        assert_eq!(a.coordinates, [3.0, 0.125]);
    }

    #[test]
    fn from_3_tuple() {
        let a = Minkowski::from((-0.5, 2.0, 18.125));
        assert_eq!(a.coordinates, [-0.5, 2.0, 18.125]);
    }

    fn from_vec<const N: usize>() {
        let mut vec = Vec::with_capacity(N);

        assert_eq!(
            Minkowski::<N>::try_from(vec.clone()),
            Err(Error::InvalidVectorLength)
        );

        for i in 0..N {
            vec.push(i as f64 * 0.5);
        }
        let a = Minkowski::<N>::try_from(vec.clone()).unwrap();

        assert_eq!(vec, Vec::from(a.coordinates));

        vec.push(1.0);
        assert_eq!(
            Minkowski::<N>::try_from(vec.clone()),
            Err(Error::InvalidVectorLength)
        );
    }
    parameterize_vector_length!(from_vec, [2, 3, 4, 8, 16, 32]);
}
