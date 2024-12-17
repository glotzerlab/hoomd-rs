use hoomd_rs_vector::vector::Cartesian;
// TODO: many shape properties are computable with 2d matrcies - imlpement these 
// as generics



#[derive(Debug)]
pub struct Simplex<const N: usize>
where
    [(); N + 1]:,
{
    vertices: [Cartesian<N>; N + 1],
}

impl<const N: usize> Default for Simplex<N>
where
    [(); N + 1]:,
{
    /** Create a regular N-Simplex.

    The default simplex is the convex hull of the basis vectors of ℝ^N and a point 
    at TODO:.

    ```
    # use hoomd_rs_geom::simplex;
    let tet = simplex::Simplex::<3>::default();
    // assert_eq!(tet, [0.0; 3].into())
    ```
    */
    #[inline]
    fn default() -> Simplex<N> {
        let mut vertices = std::array::from_fn(|i| {
            Cartesian::from(std::array::from_fn(|j| if i == j { 1.0 } else { 0.0 }))
        });

        let c = (1.0 + (1.0 + N as f64).sqrt()) / (N as f64);
        vertices[N] = Cartesian::from([c; N]);
        Simplex { vertices }
    }
}

impl<const N: usize> Simplex<N> 
    where
        [(); N+1]:
{
    // fn centroid(self) -> Cartesian::<N> {
        
    // }
}
