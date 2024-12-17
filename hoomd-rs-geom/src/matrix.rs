use hoomd_rs_vector::vector::Cartesian;
use std::ops::Index;

pub trait Matrix<const M: usize, const N: usize> {
    fn shape(&self) -> [usize; 2] {
        [M, N] // M rows and N columns
    }
    fn len(&self) -> usize { M }
    fn is_square(&self) -> bool { N==M }
}

pub trait SquareMatrix<const N: usize>: Matrix<N, N> {
    fn is_square(&self) -> bool {true}
}

pub trait Det<const N: usize>: SquareMatrix<N> {
    fn det(&self) -> f64;
}


#[derive(Debug)]
pub struct ArrayXX<const N: usize, const M: usize> {
    // M rows and N columns
    coordinates: [Cartesian<N>; M],
}

impl<const N: usize, const M: usize> Index<usize> for ArrayXX<N, M> {
    type Output = Cartesian::<N>;

    fn index(&self, i: usize) -> &Self::Output {
        &self.coordinates[i]
    }
}



impl<const N: usize, const M: usize> Matrix<N, M> for ArrayXX<N, M> {}
impl<const N: usize> SquareMatrix<N> for ArrayXX<N, N> {}

impl Det<2> for ArrayXX<2, 2> {
    fn det(&self) -> f64 where Self: Sized {
        let m = self.coordinates;
        m[0][0]*m[1][1] - m[0][1]*m[1][0]
    }
}

// impl<const N: usize, const M: usize> ArrayXX<N, M>
//     where
//         (): SquareMatrix<N, M>,
//     {
//     pub fn det(self) -> f64 {
//         0.0   
//     }
// }
