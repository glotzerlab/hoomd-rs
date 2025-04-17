// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement canonical vector types.
*/
use std::array;
use std::fmt;
use std::iter::zip;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use rand::Rng;
use rand::distr::{Distribution, StandardUniform, Uniform};

use crate::Angle;
use crate::Versor;
use crate::{Cross, Error, Rotate, Unit, Vector};

/** A [`Vector`] represented by `N` `f64` coordinates.

[`Cartesian`] is the canonical implementation of [`Vector`].

## Constructing vectors

Create a vector with an array of coordinates:
```
use hoomd_vector::Cartesian;

let v = Cartesian::from([1.0, 2.0, 3.0, 4.0, 5.0]);
```

2D and 3D vectors can also be initialized from tuples:
```
use hoomd_vector::Cartesian;

let a = Cartesian::from((1.0, 2.0, 3.0));
let b = Cartesian::from((4.0, 5.0));
```

Construct a random vector in the [-1, 1] hypercube:

```
use hoomd_vector::Cartesian;
use rand::{rngs::StdRng, Rng, SeedableRng};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut rng = StdRng::seed_from_u64(1);
let v: Cartesian::<3> = rng.random();
# Ok(())
# }
```

## Operating on vectors

Use vector math operations when you can:
```
use hoomd_vector::{Cartesian, Vector};

let a = Cartesian::from([1.0, 2.0]);
let b = Cartesian::from([4.0, 8.0]);
let c = (a + b).dot(&a);
```

Access the coordinates directly when needed:
```
use hoomd_vector::Cartesian;

let a = Cartesian::from((1.0, 2.0));
let b = Cartesian::from((a[1], 0.0));
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cartesian<const N: usize> {
    /// The vector's coordinates.
    pub coordinates: [f64; N],
}

impl<const N: usize> Default for Cartesian<N> {
    /** Create a 0 vector.

    # Example
    ```
    use hoomd_vector::Cartesian;

    let v = Cartesian::<3>::default();
    assert_eq!(v, [0.0; 3].into())
    ```
    */
    #[inline]
    #[must_use]
    fn default() -> Self {
        Cartesian::from([0.0; N])
    }
}

impl<const N: usize> From<[f64; N]> for Cartesian<N> {
    /** Create a Cartesian vector with the given coordinates.

    # Example
    ```
    use hoomd_vector::Cartesian;

    let v = Cartesian::from([4.0, 3.0]);
    ```
    */
    #[inline]
    fn from(coordinates: [f64; N]) -> Self {
        Self { coordinates }
    }
}

impl<const N: usize> IntoIterator for Cartesian<N> {
    type Item = f64;
    type IntoIter = <[f64; N] as IntoIterator>::IntoIter;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.coordinates.into_iter()
    }
}

impl From<(f64, f64)> for Cartesian<2> {
    #[inline]
    fn from(coordinates: (f64, f64)) -> Self {
        Self {
            coordinates: coordinates.into(),
        }
    }
}

impl From<(f64, f64, f64)> for Cartesian<3> {
    #[inline]
    fn from(coordinates: (f64, f64, f64)) -> Self {
        Self {
            coordinates: coordinates.into(),
        }
    }
}

impl<const N: usize> TryFrom<Vec<f64>> for Cartesian<N> {
    type Error = Error;

    /** Create a Cartesian vector with coordinates given by a [`Vec<f64>`]

    # Example
    ```
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v = Cartesian::<3>::try_from(vec![3.0, 4.0, 5.0])?;
    assert_eq!(v, [3.0, 4.0, 5.0].into());
    # Ok(())
    # }
    ```
    <div class="warning">

    Use `Cartesian::From<[f64; N]>` in performance critical code.

    </div>
    */
    #[inline]
    fn try_from(value: Vec<f64>) -> Result<Self, Self::Error> {
        let coordinates = value.try_into().map_err(|_| Error::InvalidVectorLength)?;
        Ok(Self { coordinates })
    }
}

impl<const N: usize> TryFrom<std::ops::Range<usize>> for Cartesian<N> {
    type Error = Error;

    /** Create a Cartesian vector with coordinates given by a range.

    # Example
    ```
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v = Cartesian::<3>::try_from(3..6)?;
    assert_eq!(v, [3.0, 4.0, 5.0].into());
    # Ok(())
    # }
    ```

    <div class="warning">

    Use `Cartesian::From<[f64; N]>` in performance critical code.

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

impl<const N: usize> TryFrom<[f64; N]> for Unit<Cartesian<N>> {
    type Error = Error;

    /** Create a unit Cartesian vector by normalizing the given coordinates.

    # Example
    ```
    use hoomd_vector::{Unit, Cartesian};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = Unit::<Cartesian<3>>::try_from([2.0, 0.0, 0.0])?;
    assert_eq!(*n.get(), [1.0, 0.0, 0.0].into());
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn try_from(value: [f64; N]) -> Result<Self, Self::Error> {
        Cartesian::from(value).to_unit().map(|t| t.0)
    }
}

impl<const N: usize> Vector for Cartesian<N> {
    #[inline]
    fn dot(&self, other: &Self) -> f64 {
        zip(self.coordinates.iter(), other.coordinates.iter())
            .fold(0.0, |product, x| product + x.0 * x.1)
    }
}

impl<const N: usize> Add for Cartesian<N> {
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

impl<const N: usize> Mul for Cartesian<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut coordinates = [0.0; N];

        for (result, (a, b)) in coordinates
            .iter_mut()
            .zip(self.coordinates.iter().zip(rhs.coordinates.iter()))
        {
            *result = a * b;
        }
        Self { coordinates }
    }
}

impl<const N: usize> Add<f64> for Cartesian<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: f64) -> Self {
        let mut result = self;
        result.coordinates.iter_mut().for_each(|x| *x += rhs);
        result
    }
}

impl<const N: usize> AddAssign for Cartesian<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for (result, a) in self.coordinates.iter_mut().zip(rhs.coordinates.iter()) {
            *result += a;
        }
    }
}

impl<const N: usize> Div<f64> for Cartesian<N> {
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

impl<const N: usize> DivAssign<f64> for Cartesian<N> {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        for result in &mut self.coordinates {
            *result /= rhs;
        }
    }
}

impl<const N: usize> Mul<f64> for Cartesian<N> {
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

impl<const N: usize> MulAssign<f64> for Cartesian<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        for result in &mut self.coordinates {
            *result *= rhs;
        }
    }
}

impl<const N: usize> Sub for Cartesian<N> {
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

impl<const N: usize> SubAssign for Cartesian<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for (result, a) in self.coordinates.iter_mut().zip(rhs.coordinates) {
            *result -= a;
        }
    }
}

impl<const N: usize> fmt::Display for Cartesian<N> {
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

impl<const N: usize> Neg for Cartesian<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        let mut result = self;
        result.coordinates.iter_mut().for_each(|x| *x = -*x);
        result
    }
}

impl Cross for Cartesian<3> {
    #[inline]
    fn cross(&self, other: &Self) -> Self {
        Cartesian::from((
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        ))
    }
}

impl<const N: usize> Distribution<Cartesian<N>> for StandardUniform {
    /** Sample a Cartesian vector from the uniform [-1, 1] hypercube.

    Each coordinate in the vector is in the closed range [-1, 1].

    # Example

    ```
    use hoomd_vector::Cartesian;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(1);
    let v: Cartesian::<3> = rng.random();
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Cartesian<N> {
        #[expect(
            clippy::expect_used,
            reason = "This constants chosen for this distribution are valid"
        )]
        let uniform = Uniform::new_inclusive(-1.0, 1.0).expect("a valid distribution");
        Cartesian {
            coordinates: array::from_fn(|_| uniform.sample(rng)),
        }
    }
}

impl<const N: usize, T> Index<T> for Cartesian<N>
where
    T: Into<usize> + std::slice::SliceIndex<[f64], Output = f64>,
{
    type Output = f64;
    /** Get the value of the vector at coordinate i.

    # Example
    ```
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v = Cartesian::<3>::try_from(3..6)?;
    assert_eq!((v[0], v[1], v[2]), (3.0, 4.0, 5.0));
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn index(&self, index: T) -> &Self::Output {
        &self.coordinates[index]
    }
}

impl Cartesian<2> {
    /// Compute a 2-vector perpendicular to self.
    #[inline]
    #[must_use]
    pub fn perp(self) -> Self {
        Cartesian::from([-self[1], self[0]])
    }
}

// TODO: Cartesian::multidot based on simd

impl<const N: usize, T> IndexMut<T> for Cartesian<N>
where
    T: Into<usize> + std::slice::SliceIndex<[f64], Output = f64>,
{
    /** Get a mutable reference to the value of the vector at coordinate i.

    # Example
    ```
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut v = Cartesian::<3>::try_from(3..6)?;
    assert_eq!((v[0], v[1], v[2]), (3.0, 4.0, 5.0));
    v[0] += 1.0;
    assert_eq!(v[0], 4.0);
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn index_mut(&mut self, index: T) -> &mut Self::Output {
        &mut self.coordinates[index]
    }
}

/** Rotate vectors efficiently.

Construct a [`RotationMatrix`] to efficiently rotate many vectors by the same rotation.

See:
* [`RotationMatrix::from<Angle>`]
* [`RotationMatrix::from<Versor>`]

[`RotationMatrix`] _intentionally_ does not implement [`Rotation`](crate::Rotation).
[`Angle`](crate::Angle) and [`Versor`](crate::Versor) are representations of
rotations that are often the most effective and numerically stable to
manipulate.
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotationMatrix<const N: usize> {
    /// Rows of the rotation matrix.
    pub(crate) rows: [Cartesian<N>; N],
}

impl<const N: usize> RotationMatrix<N> {
    /// View the rows of the rotation matrix.
    #[inline]
    #[must_use]
    pub fn rows(&self) -> [Cartesian<N>; N] {
        self.rows
    }
}

impl<const N: usize> Default for RotationMatrix<N> {
    /// Create an N by N identity matrix.
    #[inline]
    fn default() -> RotationMatrix<N> {
        RotationMatrix {
            rows: array::from_fn(|i| array::from_fn(|j| if i == j { 1.0 } else { 0.0 }).into()),
        }
    }
}
impl From<Versor> for RotationMatrix<3> {
    /// Create a 3 by 3 rotation matrix from a Versor.
    #[inline]
    fn from(v: Versor) -> RotationMatrix<3> {
        v.to_rotation_matrix()
    }
}
impl From<Angle> for RotationMatrix<2> {
    /// Create a 2 by 2 rotation matrix from an Angle.
    #[inline]
    fn from(theta: Angle) -> RotationMatrix<2> {
        theta.to_rotation_matrix()
    }
}

impl<const N: usize> fmt::Display for RotationMatrix<N> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.rows
                .iter()
                .map(Cartesian::<N>::to_string)
                .collect::<Vec<String>>()
                .join(",\n ")
        )
    }
}

impl<const N: usize> Rotate<Cartesian<N>> for RotationMatrix<N> {
    type Matrix = RotationMatrix<N>;

    #[inline]
    /** Rotate a [`Cartesian<N>`] by a [`RotationMatrix`]

    # Examples
    ```
    use hoomd_vector::{Angle, Rotate, RotationMatrix, Cartesian};
    use std::f64::consts::PI;

    let v = Cartesian::from([-1.0, 0.0]);
    let a = Angle::from(PI/2.0);

    let matrix = RotationMatrix::from(a);
    let rotated = matrix.rotate(&v);
    // rotated is approximately [0.0, -1.0]
    ```

    ```
    use hoomd_vector::{Versor, Rotate, RotationMatrix, Cartesian};
    use std::f64::consts::PI;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Cartesian::from([-1.0, 0.0, 0.0]);
    let v = Versor::from_axis_angle([0.0, 0.0, 1.0].try_into()?, PI/2.0);

    let matrix = RotationMatrix::from(v);
    let b = matrix.rotate(&a);
    // b is approximately [0.0, -1.0, 0.0]
    # Ok(())
    # }
    ```
    */
    fn rotate(&self, vector: &Cartesian<N>) -> Cartesian<N> {
        let mut coordinates = [0.0; N];

        for (result, row) in coordinates.iter_mut().zip(self.rows.iter()) {
            *result = row.dot(vector);
        }

        Cartesian { coordinates }
    }
}

#[cfg(test)]
mod approx {
    use approx::{AbsDiffEq, RelativeEq};
    use std::iter::zip;

    impl<const N: usize> AbsDiffEq for super::Cartesian<N> {
        type Epsilon = <f64 as AbsDiffEq>::Epsilon;

        fn default_epsilon() -> Self::Epsilon {
            f64::default_epsilon()
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            zip(self.coordinates.iter(), other.coordinates.iter())
                .all(|x| f64::abs_diff_eq(x.0, x.1, epsilon))
        }
    }

    impl<const N: usize> RelativeEq for super::Cartesian<N> {
        fn default_max_relative() -> Self::Epsilon {
            f64::default_max_relative()
        }

        fn relative_eq(
            &self,
            other: &Self,
            epsilon: Self::Epsilon,
            max_relative: Self::Epsilon,
        ) -> bool {
            zip(self.coordinates.iter(), other.coordinates.iter())
                .all(|x| f64::relative_eq(x.0, x.1, epsilon, max_relative))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paste::paste;
    use rand::{SeedableRng, rngs::StdRng};

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
    fn generate_vector_pair<const N: usize>() -> (Cartesian<N>, Cartesian<N>) {
        (
            Cartesian::try_from(0..N).unwrap(),
            Cartesian::try_from(N..N * 2).unwrap(),
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

        assert_eq!(c, Cartesian::try_from(addition_answer).unwrap());
    }
    parameterize_vector_length!(add_explicit, [2, 3, 4, 8, 16, 32]);

    fn add_operator<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();
        let c = a + b;

        let addition_answer: Vec<f64> = (0..(2 * N))
            .step_by(2)
            .map(|x| (x + N) as f64)
            .collect::<Vec<_>>();

        assert_eq!(c, Cartesian::try_from(addition_answer).unwrap());
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

        assert_eq!(c, Cartesian::try_from(addition_answer).unwrap());
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

        assert_eq!(c, Cartesian::try_from(multiplication_answer).unwrap());
    }
    parameterize_vector_length!(mul_operator, [2, 3, 4, 8, 16, 32]);

    fn mul_assign<const N: usize>() {
        let (mut a, _) = generate_vector_pair::<N>();
        let b = 12.0;
        a *= b;

        let multiplication_answer: Vec<f64> = (0..N).map(|x| (x as f64) * b).collect::<Vec<_>>();

        assert_eq!(a, Cartesian::try_from(multiplication_answer).unwrap());
    }

    parameterize_vector_length!(mul_assign, [2, 3, 4, 8, 16, 32]);

    fn div_operator<const N: usize>() {
        let (a, _) = generate_vector_pair::<N>();
        let b = 12.0;
        let c = a / b;

        let division_answer: Vec<f64> = (0..N).map(|x| (x as f64) / b).collect::<Vec<_>>();

        assert_eq!(c, Cartesian::try_from(division_answer).unwrap());
    }
    parameterize_vector_length!(div_operator, [2, 3, 4, 8, 16, 32]);

    fn div_assign<const N: usize>() {
        let (mut a, _) = generate_vector_pair::<N>();
        let b = 12.0;
        a /= b;

        let division_answer: Vec<f64> = (0..N).map(|x| (x as f64) / b).collect::<Vec<_>>();

        assert_eq!(a, Cartesian::try_from(division_answer).unwrap());
    }

    parameterize_vector_length!(div_assign, [2, 3, 4, 8, 16, 32]);

    fn compute_add_ref_ref<const N: usize>(a: &Cartesian<N>, b: &Cartesian<N>) -> Cartesian<N> {
        *a + *b
    }

    fn compute_add_ref_type<const N: usize>(a: &Cartesian<N>, b: Cartesian<N>) -> Cartesian<N> {
        *a + b
    }

    fn compute_add_type_ref<const N: usize>(a: Cartesian<N>, b: &Cartesian<N>) -> Cartesian<N> {
        a + *b
    }

    fn add_with_refs<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();

        let addition_answer = Cartesian::try_from(
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

    fn dot<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();
        let c = a.dot(&b);

        let n = N as f64;

        // Analytical solution to sum of k * (n+k), k = 0 to n-1
        let dot_ans = (5.0 * n.powi(3) - 6.0 * n.powi(2) + n) / 6.0;

        assert_eq!(c, dot_ans);
    }
    parameterize_vector_length!(dot, [2, 3, 4, 8, 16, 32]);

    fn neg<const N: usize>() {
        let a: Cartesian<N> = Cartesian::try_from(0..N).unwrap();
        let b = -a;
        for (i, j) in zip(a.coordinates.iter(), b.coordinates.iter()) {
            assert_eq!(*i, -j);
        }
    }
    parameterize_vector_length!(neg, [2, 3, 4, 8, 16, 32]);

    #[test]
    fn cross() {
        let (a, b) = generate_vector_pair::<3>();
        let c = a.cross(&b);

        // Analytical solution
        let cross_ans = Cartesian::from((-3.0, 6.0, -3.0));

        assert_eq!(c, cross_ans);

        let a = Cartesian::from([1.0, 0.0, 0.0]);
        let b = Cartesian::from([0.0, 1.0, 0.0]);
        assert_eq!(a.cross(&b), [0.0, 0.0, 1.0].into());

        let a = Cartesian::from([0.0, 3.0, 0.0]);
        let b = Cartesian::from([0.0, 0.0, 2.0]);
        assert_eq!(a.cross(&b), [6.0, 0.0, 0.0].into());

        let a = Cartesian::from([2.0, 0.0, 0.0]);
        let b = Cartesian::from([0.0, 0.0, 4.0]);
        assert_eq!(a.cross(&b), [0.0, -8.0, 0.0].into());
    }

    #[test]
    fn display() {
        let a = Cartesian::from([1.5, 2.125, -3.875]);
        let s = format!("{a}");
        assert_eq!(s, "[1.5, 2.125, -3.875]");

        let a = Cartesian::from([10.0, 20.0, 30.0, 40.0]);
        let s = format!("{a}");
        assert_eq!(s, "[10, 20, 30, 40]");
    }

    #[test]
    fn from_2_tuple() {
        let a = Cartesian::from((3.0, 0.125));
        assert_eq!(a.coordinates, [3.0, 0.125]);
    }

    #[test]
    fn from_3_tuple() {
        let a = Cartesian::from((-0.5, 2.0, 18.125));
        assert_eq!(a.coordinates, [-0.5, 2.0, 18.125]);
    }

    fn from_vec<const N: usize>() {
        let mut vec = Vec::with_capacity(N);

        assert_eq!(
            Cartesian::<N>::try_from(vec.clone()),
            Err(Error::InvalidVectorLength)
        );

        for i in 0..N {
            vec.push(i as f64 * 0.5);
        }
        let a = Cartesian::<N>::try_from(vec.clone()).unwrap();

        assert_eq!(vec, Vec::from(a.coordinates));

        vec.push(1.0);
        assert_eq!(
            Cartesian::<N>::try_from(vec.clone()),
            Err(Error::InvalidVectorLength)
        );
    }
    parameterize_vector_length!(from_vec, [2, 3, 4, 8, 16, 32]);

    fn random_in_range<const N: usize>() {
        // Loosely verify we are drawing from the correct distribution
        let mut rng = StdRng::seed_from_u64(1);
        let a: Cartesian<N> = rng.random();

        assert!(a.coordinates.iter().all(|&x| -1.0 < x && x < 1.0));

        // This test will fail ~1e-3008 percent of the time - it's probably fine
        if N == 10_000 {
            assert!(a.coordinates.iter().any(|&x| x < 0.0));
        }
    }

    parameterize_vector_length!(random_in_range, [2, 3, 4, 8, 16, 32, 10_000]);

    #[test]
    fn to_unit() {
        let a = Cartesian::from((2.0, 0.0, 0.0));
        let (Unit(unit_a), _) = a.to_unit().expect("non-zero vector");
        assert_eq!(unit_a, [1.0, 0.0, 0.0].into());

        let (Unit(unit_a), _) = a.to_unit_unchecked();
        assert_eq!(unit_a, [1.0, 0.0, 0.0].into());

        let Unit(unit_a) =
            Unit::<Cartesian<3>>::try_from([1.0, 0.0, 0.0]).expect("non-zero vector");
        assert_eq!(unit_a, [1.0, 0.0, 0.0].into());

        let a = Cartesian::from((3.0, 0.0, 4.0));
        let (Unit(unit_a), _) = a.to_unit().expect("non-zero vector");
        assert_eq!(unit_a, [3.0 / 5.0, 0.0, 4.0 / 5.0].into());

        let (Unit(unit_a), _) = a.to_unit_unchecked();
        assert_eq!(unit_a, [3.0 / 5.0, 0.0, 4.0 / 5.0].into());

        let Unit(unit_a) =
            Unit::<Cartesian<3>>::try_from([3.0, 0.0, 4.0]).expect("non-zero vector");
        assert_eq!(unit_a, [3.0 / 5.0, 0.0, 4.0 / 5.0].into());

        let a = Cartesian::from((0.0, 0.0, 0.0));
        assert!(matches!(a.to_unit(), Err(Error::InvalidMagnitude)));
        assert!(matches!(
            Unit::<Cartesian<3>>::try_from([0.0, 0.0, 0.0]),
            Err(Error::InvalidMagnitude)
        ));
    }
    #[allow(clippy::print_stdout)]
    #[test]
    fn test_display_rotationmatrix() {
        println!("\n{}", RotationMatrix::<2>::default());
        println!("\n{}", RotationMatrix::<3>::default());
    }
}
