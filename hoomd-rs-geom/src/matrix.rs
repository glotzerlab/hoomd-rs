use hoomd_rs_vector::vector::Cartesian;
use std::ops::{Add, Index};
use std::array;

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
    rows: [Cartesian<N>; M],
}

impl<const N: usize, const M: usize> Index<usize> for ArrayXX<N, M> {
    type Output = Cartesian::<N>;

    fn index(&self, i: usize) -> &Self::Output {
        &self.rows[i]
    }
}



impl<const N: usize, const M: usize> Matrix<N, M> for ArrayXX<N, M> {}
impl<const N: usize> SquareMatrix<N> for ArrayXX<N, N> {}

impl<const N: usize> ArrayXX<N, N> {
    fn diag(&self, k: usize) -> Cartesian::<N>
    {
        array::from_fn(|i| (*self)[i].coordinates[i+k]).into()
    }

}

impl Det<2> for ArrayXX<2, 2> {
    fn det(&self) -> f64 where Self: Sized {
        self[0].coordinates[0]*self[1].coordinates[1] - self[0].coordinates[1]*self[1].coordinates[0]
    }
}

impl Det<3> for ArrayXX<3, 3> {
    fn det(&self) -> f64 where Self: Sized {
        self.diag(0).coordinates.iter().product::<f64>() + 
        self[0].coordinates[1] * self[1].coordinates[2] * self[2].coordinates[0]
        self[0].coordinates[2] * self[1].coordinates[0] * self[2].coordinates[0]
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
