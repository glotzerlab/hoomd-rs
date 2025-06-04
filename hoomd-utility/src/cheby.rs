// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Chebyshev`]
 */

/** Evaluates the Chebyshev polynomials and its derivatives
  of the first kind `T_O(s)` for orders O equals 0 to
  `n - 1`, given coordinate `s`.

*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chebyshev {
    /// Maximum order of the Chebyshev polynomials `T_O` (O = 0 to n-1).
    pub n: usize,
}

impl Chebyshev {
    /** The `eval_cheby` fucntion returns a vector where
    the `i`-th element is `T_i(s)`, computed using the
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
    use hoomd_utility::cheby::Chebyshev;

    let cheby = Chebyshev { n: 4 };
    let s = 0.5;
    let tn = cheby.eval_cheby(&s);
    assert_eq!(tn[0], 1.0); // T_0(0.5) = 1
    assert_eq!(tn[1], 0.5); // T_1(0.5) = 0.5
    assert_eq!(tn[2], -0.5); // T_2(0.5) = 2(0.5)^2 - 1 = -0.5
    assert_eq!(tn[3], -1.0); // T_3(0.5) = 4(0.5)^3 - 3(0.5) = -1.0
    ```

    # Notes

    If `n` is 0, an empty vector is returned.
    */
    #[must_use]
    #[inline]
    pub fn eval_cheby(&self, s: &f64) -> Vec<f64> {
        let mut tn: Vec<f64> = vec![0.0; self.n];

        if self.n == 0 {
            return tn;
        }

        tn[0] = 1.0; // T_0(s) = 1
        if self.n > 1 {
            tn[1] = *s; // T_1(s) = s
        }

        // Compute T_n(s) using recurrence: T_n = 2s * T_{n-1} - T_{n-2}
        for idx in 2..self.n {
            tn[idx] = 2.0 * s * tn[idx - 1] - tn[idx - 2];
        }
        tn
    }

    /**
    The `eval_dcheby_ds` fucntion returns a vector where
    the `i`-th element is `\dfrac{T_i(s)}{ds}`, by first, computing
    the Chebyshev polynomials of the second kind `U_i(s)` using the
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
    \frac{d T_0}{ds} = 0
    ```

    # Examples

    ```
    use hoomd_utility::cheby::Chebyshev;

    let cheby = Chebyshev { n: 4 };
    let s = 0.5;
    let tnd = cheby.eval_dcheby_ds(&s);
    assert_eq!(tnd[0], 0.0); // dT_0/ds = 0
    assert_eq!(tnd[1], 1.0); // dT_1/ds = 1 * U_0 = 1
    assert_eq!(tnd[2], 2.0); // dT_2/ds = 2 * U_1 = 2 * 2 * 0.5 = 2
    assert_eq!(tnd[3], 0.0); // dT_3/ds = 3 * U_2 = 3 * (4(0.5)^2 - 1) = 0.0
    ```

    # Notes

    If `n` is 0, an empty vector is returned.
    */
    #[must_use]
    #[inline]
    pub fn eval_dcheby_ds(&self, s: &f64) -> Vec<f64> {
        let mut tnd: Vec<f64> = vec![0.0; self.n];

        if self.n == 0 {
            return tnd;
        }

        tnd[0] = 1.0; // U_0(s) = 1
        if self.n > 1 {
            tnd[1] = 2.0 * s; // U_1(s) = 2s
        }

        // Compute U_n(s) using recurrence: U_n = 2s * U_{n-1} - U_{n-2}
        for idx in 2..self.n {
            tnd[idx] = 2.0 * s * tnd[idx - 1] - tnd[idx - 2];
        }

        // Convert to dT_n/ds = n * U_{n-1}
        for idx in (1..self.n).rev() {
            tnd[idx] = (idx as f64) * tnd[idx - 1]; // Cast idx to f64
        }
        tnd[0] = 0.0; // dT_0/ds = 0
        tnd
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    fn test_eval_cheby() {
        let cheby = Chebyshev { n: 4 };
        let s = 0.5;
        let tn = cheby.eval_cheby(&s);

        // Expected values for T_n(0.5)
        assert_eq!(tn.len(), 4);
        assert_eq!(tn[0], 1.0); // T_0(0.5) = 1
        assert_eq!(tn[1], 0.5); // T_1(0.5) = 0.5
        assert_eq!(tn[2], -0.5); // T_2(0.5) = 2(0.5)^2 - 1 = -0.5
        assert_eq!(tn[3], -1.0); // T_3(0.5) = 4(0.5)^3 - 3(0.5) = -1.0

        // Edge case: n = 0
        let cheby_empty = Chebyshev { n: 0 };
        assert_eq!(cheby_empty.eval_cheby(&s), vec![]);

        // Edge case: n = 1
        let cheby_one = Chebyshev { n: 1 };
        assert_eq!(cheby_one.eval_cheby(&s), vec![1.0]);
    }

    #[rstest]
    fn test_eval_dcheby_ds() {
        let cheby = Chebyshev { n: 4 };
        let s = 0.5;
        let tnd = cheby.eval_dcheby_ds(&s);

        // Expected values for dT_n/ds = n * U_{n-1}(0.5)
        // U_0(0.5) = 1, U_1(0.5) = 2*0.5 = 1, U_2(0.5) = 4(0.5)^2 - 1 = -0.5
        assert_eq!(tnd.len(), 4);
        assert_eq!(tnd[0], 0.0); // dT_0/ds = 0
        assert_eq!(tnd[1], 1.0); // dT_1/ds = 1 * U_0 = 1
        assert_eq!(tnd[2], 2.0); // dT_2/ds = 2 * U_1 = 2 * 1 = 2
        assert_eq!(tnd[3], 0.0); // dT_3/ds = 3 * U_2 = 3 * (4(0.5)^2 - 1) = 0.0

        // Edge case: n = 0
        let cheby_empty = Chebyshev { n: 0 };
        assert_eq!(cheby_empty.eval_dcheby_ds(&s), vec![]);

        // Edge case: n = 1
        let cheby_one = Chebyshev { n: 1 };
        assert_eq!(cheby_one.eval_dcheby_ds(&s), vec![0.0]);
    }
}
