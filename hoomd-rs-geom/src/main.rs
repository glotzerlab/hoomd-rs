#![feature(generic_const_exprs)]

pub mod simplex;

fn main() {
    println!("Hello, world!");

    const N: usize = 3;

    let s = simplex::Simplex::<N>::default();
    println!("{:?}", s)
}
