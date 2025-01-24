// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

pub mod intersects;
pub mod matrix;
pub mod shape;
pub mod simplex;
pub mod sphere;
use crate::intersects::Intersects;
use crate::shape::{Convex, Shape, Volume};
use crate::sphere::Sphere;
use hoomd_vector::Cartesian;

fn main() {
    println!("Hello, world!");

    const N: usize = 3;
    let s = Sphere::<N, Cartesian<N>>::default();
    let other = Sphere::<N, Cartesian<N>>::from((1.0, [2.0000001, 0.0, 0.0]));

    println!("{:?}", s);
    println!("{:?}", s.volume());
    println!("{:?}", s.centroid());
    println!("{:?}", s.is_convex());

    // println!("{:?}", double_factorial(35));
    println!("Intersects: {:?}", s.intersects(other));
}
