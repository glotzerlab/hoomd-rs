

pub trait Scale{
    fn Scale(self, scale_factor: PositiveReal) -> Self;
}

impl Scale for Hyperparallelepiped<N>{
    fn Scale(self, ){
        Hyperparallelepiped.edge_vectors.map(|v| v.map(|x| x * scale_factor.get() ))
    }
}