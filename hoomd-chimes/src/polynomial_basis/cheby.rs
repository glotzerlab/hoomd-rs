// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Chebyshev`]
 */
use super::Basis;
use arrayvec::ArrayVec;

/** Evaluates the Chebyshev polynomials and its derivatives
  of the first kind $`T_i(s)`$ for orders $`i`$ equals 1 to
  $`n`$, given coordinate $`s`$.

*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chebyshev<const N: usize> {}

impl<const N: usize> Default for Chebyshev<N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Chebyshev<N> {
    /**  Creates a new `Chebyshev` instance with maximum order `N`.

    The struct computes `Chebyshev` polynomials $`T_1(s)`$ to $`T_{N}(s)`$.

    # Panics

    Will panic if N = 0.
    */
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        assert!(N > 0, "Chebyshev order must be at least 1 (N >= 1)");
        Chebyshev {}
    }
}

impl<const N: usize> Basis<N> for Chebyshev<N> {
    /** The `eval_cheby` fucntion returns a vector where
    the `i`-th element is $`T_i(s)`$, computed using the
    recurrence relation:

    ```math
    \begin{cases}
    T_0(s) = 1 \\
    T_1(s) = s \\
    T_i(s) = 2s T_{i-1}(s) - T_{i-2}(s) \text{ for } i \geq 2 \\
    \end{cases}
    ```

    # Examples

    ```
    use hoomd_chimes::polynomial_basis::{Basis, Chebyshev};

    let cheby = Chebyshev::<3>::new();
    let s = 0.5;
    let tn = cheby.evaluate(&s);
    // T_1=0.5, T_2=-0.5, T_3=-1.0
    assert_eq!(tn.as_slice(), [0.5, -0.5, -1.0]);
    ```
    */
    #[inline]
    fn evaluate(&self, s: &f64) -> ArrayVec<f64, N> {
        let mut tn = ArrayVec::<f64, N>::new();
        let t0_fn = 1.0; // T_0(s) = 1

        tn.push(*s); // T_1(s) = s
        if N > 1 {
            tn.push(2.0 * s * tn[0] - t0_fn); // T_1(s) = s
        }

        // Compute T_i(s) using recurrence: T_i = 2s * T_{i-1} - T_{i-2}
        for idx in 2..N {
            tn.push(2.0 * s * tn[idx - 1] - tn[idx - 2]);
        }
        tn
    }

    /**
    The fucntion returns a vector where the $`i`$-th element is
    $`\frac{dT_i(s)}{ds}`$, by first, computing the Chebyshev
    polynomials of the second kind $`U_i(s)`$ using the
    recurrence relation:

    ```math
    \begin{cases}
    U_0(s) = 1  \\
    U_1(s) = 2s \\
    U_i(s) = 2s U_{i-1}(s) - U_{i-2}(s) \text{ for } i \geq 2 \\
    \end{cases}
    ```

    Then use the relation:

    ```math
    \frac{d T_i}{ds} = i \times U_{i-1}(s) ,
    ```
    Particularly

    ```math
    \frac{d T_1}{ds} = 1
    ```

    # Examples

    ```
    use hoomd_chimes::polynomial_basis::{Basis, Chebyshev};

    let cheby = Chebyshev::<3>::new();
    let s = 0.5;
    let tnd = cheby.evaluate_derivative(&s);
    // dT_1/ds=1, dT_2/ds=2, dT_3/ds=0.0

    assert_eq!(tnd.as_slice(), [1.0, 2.0, 0.0]);
    ```
    */
    #[inline]
    fn evaluate_derivative(&self, s: &f64) -> ArrayVec<f64, N> {
        let mut tnd = ArrayVec::<f64, N>::new();
        let u0_fn = 1.0; // U_0(s) = 1

        tnd.push(2.0 * s); // U_1(s) = 2s
        if N > 1 {
            tnd.push(2.0 * s * tnd[0] - u0_fn);
        }

        // Compute U_i(s) using recurrence: U_i = 2s * U_{i-1} - U_{i-2}
        for idx in 2..N {
            tnd.push(2.0 * s * tnd[idx - 1] - tnd[idx - 2]);
        }

        // Convert to dT_i/ds = i * U_{i-1}
        for idx in (1..N).rev() {
            tnd[idx] = ((idx as f64) + 1.0) * tnd[idx - 1];
        }
        tnd[0] = 1.0;
        tnd
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    fn test_evaluate() {
        let cheby = Chebyshev::<3>::new();
        let s = 0.5;
        let tn = cheby.evaluate(&s);
        // Expected: T_0(s) = 1, T_1(s) = s, T_2(s) = 2s^2 - 1, T_3(s) = 4s^3 - 3s
        let expected = [
            0.5,                               // T_1 = 0.5
            2.0 * 0.5 * 0.5 - 1.0,             // T_2 = 2 * (0.5)^2 - 1 = -0.5
            4.0 * 0.5 * 0.5 * 0.5 - 3.0 * 0.5, // T_3 = 4 * (0.5)^3 - 3 * 0.5 = -1.0
        ];
        assert_eq!(tn.as_slice(), &expected);

        // Edge case: N = 1
        let cheby = Chebyshev::<1>::new();
        assert_eq!(cheby.evaluate(&0.5).as_slice(), &[s]);
    }

    #[rstest]
    fn test_evaluate_derivative() {
        let cheby = Chebyshev::<3>::new();
        let s = 0.5;
        let tnd = cheby.evaluate_derivative(&s);
        // Expected: dT_0/ds = 0, dT_1/ds = 1, dT_2/ds = 4s, dT_3/ds = 3(4s^2 - 1)
        let expected = [
            1.0,                           // dT_1/ds = 1 * U_0 = 1
            2.0 * 2.0 * 0.5,               // dT_2/ds = 2 * U_1 = 2 * 2 * 0.5 = 2
            3.0 * (4.0 * 0.5 * 0.5 - 1.0), // dT_3/ds = 3 * U_2 = 3 * (4 * (0.5)^2 - 1) = 0.0
        ];
        assert_eq!(tnd.as_slice(), &expected);

        // Edge case: N = 1
        let cheby = Chebyshev::<1>::new();
        assert_eq!(cheby.evaluate_derivative(&0.5).as_slice(), &[1.0]);
    }

    #[rstest]
    #[should_panic(expected = "Chebyshev order must be at least 1")]
    fn test_panic_n_zero() {
        let _ = Chebyshev::<0>::new();
    }
}
