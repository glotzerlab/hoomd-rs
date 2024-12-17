use hoomd_rs_vector::vector::Cartesian;

#[derive(Debug)]
pub struct Simplex<const N: usize>
where
    [(); N + 1]:,
{
    coordinates: [Cartesian<N>; N + 1],
}

impl<const N: usize> Default for Simplex<N>
where
    [(); N + 1]:,
{
    /** Create a regular N-Simplex

    ```
    # use hoomd_rs_geom::simplex;
    let tet = simplex::Simplex::<3>::default();
    // assert_eq!(tet, [0.0; 3].into())
    ```
    */
    #[inline]
    fn default() -> Simplex<N> {
        let mut coordinates = std::array::from_fn(|i| {
            Cartesian::from(std::array::from_fn(|j| if i == j { 1.0 } else { 0.0 }))
        });

        let c = (1.0 + (1.0 + N as f64).sqrt()) / (N as f64);
        coordinates[coordinates.len() - 1] = Cartesian::from([c; N]);
        Simplex { coordinates }
    }
}
