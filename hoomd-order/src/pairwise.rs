use hoomd_vector::{InnerProduct, Rotate, Unit, Vector};

pub trait UnivariateCollectiveVariables {

    fn cv(&self, r: f64) -> f64;
}

pub trait CollectiveVariables<V: Vector, R: Rotate<V>> {

    fn cv(&self, r: V) -> f64;
}

pub struct SimpleMask {
    pub r_cut: f64
}

impl UnivariateCollectiveVariables for SimpleMask {
    fn cv(&self, r: f64) -> f64 {
        if r <= self.r_cut{
            1.0
        }
        else {
            0.0
        }
    }
}
