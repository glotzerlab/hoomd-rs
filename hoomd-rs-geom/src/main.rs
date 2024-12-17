#![feature(generic_const_exprs)] // See https://hackmd.io/OZG_XiLFRs2Xmw5s39jRzA?view

pub mod simplex;
pub mod matrix;

fn main() {
    println!("Hello, world!");

    const N: usize = 3;

    let s = simplex::Simplex::<N>::default();
    println!("{:?}", s)
}
