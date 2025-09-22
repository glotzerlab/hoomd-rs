use hoomd_utility::valid::PositiveReal;
use crate::shape::{Capsule, ConvexPolytope, Cuboid, Cylinder, Hyperellipsoid, Hyperparallelepiped, Hypersphere, Simplex3};

pub trait Scale{
    fn scale(&mut self, scale_factor: PositiveReal);
}

pub<const N: usize> trait Shear{
    fn shear(&mut self, angle: f64 , parallel_axis: Cartesian<N>, perpendicular_axis: Cartesian<N>);
}

// pub trait Elongate{
//     fn scale(&mut self, ,scale_factor: PositiveReal);
// }

impl<const N: usize> Scale for Capsule<N>{
    fn scale(&mut self, scale_factor: PositiveReal){
        self.height *= scale_factor;
        self.radius *= scale_factor;
    }
}

impl Scale for Cylinder{
    fn scale(&mut self, scale_factor: PositiveReal){
        self.height *= scale_factor;
        self.radius *= scale_factor;
    }
}

impl<const N: usize> Scale for Cuboid<N>{
    fn scale(&mut self, scale_factor: PositiveReal){
        self.edge_lengths = self.edge_lengths.map(|v| v * scale_factor);
    }
}

impl<const N: usize> Scale for Hyperparallelepiped<N>{
    fn scale(&mut self, scale_factor: PositiveReal){
        self.edge_vectors = self.edge_vectors.map(|v| v *scale_factor );
    }
}

impl<const N: usize> Scale for Hypersphere<N>{
    fn scale(&mut self, scale_factor: PositiveReal){
        self.radius *= scale_factor;
    }
}

impl<const N: usize> Scale for Hyperellipsoid<N>{
    fn scale(&mut self, scale_factor: PositiveReal){
        self.semi_axes = self.semi_axes.map(|v| v *scale_factor);
    }
}

impl<const N: usize> Scale for ConvexPolytope<N>{
    fn scale(&mut self, scale_factor: PositiveReal){
        for vertex in &mut self.vertices{
            *vertex *= scale_factor;
        }
    }
}

impl Scale for Simplex3{
    fn scale(&mut self, scale_factor: PositiveReal){
        for vertex in &mut self.vertices{
            *vertex *= scale_factor;
        }
        self.bounding_radius *= scale_factor;
    }
}

impl<const N: usize> Shear for Hyperparallelepiped<N>{
    fn shear(&mut self, angle: f64 , parallel_axis: &Cartesian<N>, perpendicular_axis: &Cartesian<N>){
        shear_matrix = eye() + perpendicular_axis.to_column().matmul(parallel_axis.to_row());
        self.edge_vectors = self.edge_vectors.map(|v| shear_matrix.matmul(v.to_column()).into());
    }
}

#[cfg(test)]
#[expect(clippy::used_underscore_binding, reason = "Required for const tests.")]
mod tests {
    use super::*;
    #[test]
    fn test_cuboid_scale(){
        let scale_factor: PositiveReal = 5.0.try_into().unwrap();
        let mut my_cuboid = Cuboid::<3>{edge_lengths: [1.,2.,1.].map(|x| x.try_into().unwrap())};
        my_cuboid.scale(scale_factor);

        assert_eq!(my_cuboid.edge_lengths, [5.,10.,5.].map(|x| x.try_into().unwrap()));
    }
}