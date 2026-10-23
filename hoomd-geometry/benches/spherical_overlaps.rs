// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]

//! Benchmark hyperbolic overlaps

use divan::{self, Bencher, black_box, counter::ItemsCount};
use hoomd_geometry::{IntersectsAtGlobal, shape::SphericalConvexPolygon};
use hoomd_manifold::Spherical;
use hoomd_vector::Angle;
use rand::{
    Rng, SeedableRng,
    distr::{Distribution, Uniform},
    rngs::StdRng,
};
use std::f64::consts::PI;

fn main() {
    divan::main();
}

fn deck_transform(zenith: f64, azimuth: f64, position: &Spherical<3>) -> f64 {
    let theta_over_two = zenith / 2.0;
    let stereo = position.stereographic_projection_from_south_pole();
    -2.0 * (((theta_over_two.sin()) * ((azimuth.sin()) * stereo[0] - (azimuth.cos()) * stereo[1]))
        .atan2(
            (theta_over_two.cos())
                - (theta_over_two.sin())
                    * ((azimuth.cos()) * stereo[0] + (azimuth.sin()) * stereo[1]),
        ))
}

fn translate_then_gen_vertices<R: Rng>(
    rng: &mut R,
    shape: &SphericalConvexPolygon,
) -> Vec<Spherical<3>> {
    let origin = Spherical::<3>::from_polar_coordinates(0.3, 0.4);
    let trial_zenith = Uniform::new(0.0, PI / 3.0).expect("r is positive and real");
    let trial_azimuth = Uniform::new(-PI, PI).expect("hard-coded distribution should be valid");
    let theta = trial_zenith.sample(rng);
    let phi: f64 = trial_azimuth.sample(rng);
    let (sn_th, cs_th, sn_ph, cs_ph) = (theta.sin(), theta.cos(), phi.sin(), phi.cos());
    let pt = origin.coordinates();

    let del_phi = deck_transform(theta, phi, &origin);
    let translated_origin = Spherical::<3>::from_cartesian_coordinates(
        [
            pt[0] * (cs_th * cs_ph * cs_ph + sn_ph * sn_ph)
                + pt[1] * sn_ph * cs_ph * (cs_th - 1.0)
                + pt[2] * sn_th * cs_ph,
            pt[0] * sn_ph * cs_ph * (cs_th - 1.0)
                + pt[1] * (cs_th * sn_ph * sn_ph + cs_ph * cs_ph)
                + pt[2] * sn_th * sn_ph,
            -pt[0] * sn_th * cs_ph - pt[1] * sn_th * sn_ph + pt[2] * cs_th,
        ]
        .into(),
    );
    shape
        .vertices()
        .iter()
        .map(|v| {
            SphericalConvexPolygon::vertex_to_system_frame(
                v,
                Angle::from(del_phi),
                &translated_origin,
            )
        })
        .collect::<Vec<Spherical<3>>>()
}

fn gen_vertices_then_translate<R: Rng>(
    rng: &mut R,
    shape: &SphericalConvexPolygon,
) -> Vec<Spherical<3>> {
    let origin = Spherical::<3>::from_polar_coordinates(0.3, 0.4);
    let trial_zenith = Uniform::new(0.0, PI / 3.0).expect("r is positive and real");
    let trial_azimuth = Uniform::new(-PI, PI).expect("hard-coded distribution should be valid");
    let theta = trial_zenith.sample(rng);
    let phi: f64 = trial_azimuth.sample(rng);
    let (sn_th, cs_th, sn_ph, cs_ph) = (theta.sin(), theta.cos(), phi.sin(), phi.cos());

    let vertices = shape
        .vertices()
        .iter()
        .map(|v| {
            let v_transformed =
                SphericalConvexPolygon::vertex_to_system_frame(v, Angle::from(0.0), &origin);
            v_transformed.point().coordinates
        })
        .collect::<Vec<[f64; 3]>>();

    let translate = |pt: &[f64; 3]| -> Spherical<3> {
        Spherical::<3>::from_cartesian_coordinates(
            [
                pt[0] * (cs_th * cs_ph * cs_ph + sn_ph * sn_ph)
                    + pt[1] * sn_ph * cs_ph * (cs_th - 1.0)
                    + pt[2] * sn_th * cs_ph,
                pt[0] * sn_ph * cs_ph * (cs_th - 1.0)
                    + pt[1] * (cs_th * sn_ph * sn_ph + cs_ph * cs_ph)
                    + pt[2] * sn_th * sn_ph,
                -pt[0] * sn_th * cs_ph - pt[1] * sn_th * sn_ph + pt[2] * cs_th,
            ]
            .into(),
        )
    };

    vertices
        .iter()
        .map(translate)
        .collect::<Vec<Spherical<3>>>()
}

fn gnomonic_overlap<R: Rng>(rng: &mut R, shape: &SphericalConvexPolygon) -> bool {
    let pos_1 = Spherical::<3>::from_polar_coordinates(0.01, 0.2);
    let trial_zenith = Uniform::new(0.0, PI / 3.0).expect("r is positive and real");
    let trial_azimuth = Uniform::new(-PI, PI).expect("hard-coded distribution should be valid");
    let theta = trial_zenith.sample(rng);
    let phi: f64 = trial_azimuth.sample(rng);
    let pos_2 = Spherical::<3>::from_polar_coordinates(theta, phi);
    let (orientation_1, orientation_2) = (
        Angle::from(trial_azimuth.sample(rng)),
        Angle::from(trial_azimuth.sample(rng)),
    );

    shape.intersects_at_global(shape, &pos_1, &orientation_1, &pos_2, &orientation_2)
}

const NUM_VERTICES: &[usize] = &[3, 4, 5, 8, 16];

#[divan::bench(consts = NUM_VERTICES, sample_count = 1000)]
fn deck_transform_then_generate_vertices<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);
    let shape = SphericalConvexPolygon::regular(N, 0.5);

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(translate_then_gen_vertices(&mut rng, &shape)));
}

#[divan::bench(consts = NUM_VERTICES, sample_count = 1000)]
fn generate_vertices_then_transform<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);
    let shape = SphericalConvexPolygon::regular(N, 0.5);

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(gen_vertices_then_translate(&mut rng, &shape)));
}

#[divan::bench(consts = NUM_VERTICES, sample_count = 1000)]
fn overlap_check<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);
    let shape = SphericalConvexPolygon::regular(N, 0.25);

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(gnomonic_overlap(&mut rng, &shape)));
}
