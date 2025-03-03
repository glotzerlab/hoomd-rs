// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

pub mod cuboid;
pub mod intersects;
pub mod matrix;
pub mod poly;
pub mod modifiers;
pub mod shape;
pub mod simplex;
pub mod sphere;
use crate::cuboid::Cuboid;
use crate::intersects::{Intersects, IntersectsAt};
use crate::shape::{Shape, Volume};
use crate::sphere::Sphere;
use hoomd_vector::{Cartesian, Rotation, Versor};

fn main() {
    println!("Hello, world!");

    const N: usize = 3;
    let s = Sphere::<N>::default();
    let other = Sphere::<N>::from(1.0);

    println!("{:?}", s);
    println!("{:?}", s.volume());
    // println!("{:?}", s.centroid());

    let aab = Cuboid::<N> {
        edge_lengths: [1.0, 1.0, 1.0].into(),
        center: [0.0; 3].into(),
    };
    let aab1 = Cuboid::<N> {
        edge_lengths: [1.0, 2.0, 2.0].into(),
        center: [9.0; 3].into(),
    };
    println!("{:?}: {},{},{}", aab, aab.a(), aab.b(), aab.c());
    // println!("Cuboid intersects {:?}", aab.intersects(aab1));

    // println!("{:?}", double_factorial(35));
    println!(
        "Intersects: {:?}",
        s.intersects_at(
            &other,
            &(Cartesian::from([2.01, 0.0, 0.0])),
            &Versor::identity()
        )
    );
}
