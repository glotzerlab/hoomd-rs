// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]

//! Benchmark hyperbolic overlaps

use divan::{self, Bencher, black_box, counter::ItemsCount};
use hoomd_geometry::{IntersectsAtGlobal, shape::HyperbolicConvexPolygon};
use hoomd_manifold::Hyperbolic;
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

fn deck_transform(boost: f64, rotation: f64, position: &Hyperbolic<3>) -> f64 {
    let tau_over_two = boost / 2.0;
    let poincare = position.to_poincare();
    -2.0 * (((tau_over_two.sinh())
        * ((rotation.cos()) * poincare[1] - (rotation.sin()) * poincare[0]))
        .atan2(
            (tau_over_two.cosh())
                + (tau_over_two.sinh())
                    * ((rotation.cos()) * poincare[0] + (rotation.sin()) * poincare[1]),
        ))
}

fn translate_then_gen_vertices<R: Rng>(
    rng: &mut R,
    shape: &HyperbolicConvexPolygon,
) -> Vec<Hyperbolic<3>> {
    let origin = Hyperbolic::<3>::from_polar_coordinates(0.3, 0.4);
    let trial_boost = Uniform::new(0.0, 1.0).expect("r is positive and real");
    let trial_rotation = Uniform::new(-PI, PI).expect("hard-coded distribution should be valid");
    let theta = trial_rotation.sample(rng);
    let v: f64 = trial_boost.sample(rng);
    let (sh_v, ch_v, sn_th, cs_th) = (v.sinh(), v.cosh(), theta.sin(), theta.cos());
    let pt = origin.coordinates();

    let del_phi = deck_transform(v, theta, &origin);
    let translated_origin = Hyperbolic::<3>::from_minkowski_coordinates(
        [
            pt[0] * (ch_v * cs_th * cs_th + sn_th * sn_th)
                + pt[1] * sn_th * cs_th * (ch_v - 1.0)
                + pt[2] * sh_v * cs_th,
            pt[0] * sn_th * cs_th * (ch_v - 1.0)
                + pt[1] * (ch_v * sn_th * sn_th + cs_th * cs_th)
                + pt[2] * sh_v * sn_th,
            pt[0] * sh_v * cs_th + pt[1] * sh_v * sn_th + pt[2] * ch_v,
        ]
        .into(),
    );
    shape
        .vertices()
        .iter()
        .map(|v| {
            HyperbolicConvexPolygon::vertex_to_system_frame(
                v,
                Angle::from(del_phi),
                &translated_origin,
            )
        })
        .collect::<Vec<Hyperbolic<3>>>()
}

fn gen_vertices_then_translate<R: Rng>(
    rng: &mut R,
    shape: &HyperbolicConvexPolygon,
) -> Vec<Hyperbolic<3>> {
    let origin = Hyperbolic::<3>::from_polar_coordinates(0.3, 0.4);
    let trial_boost = Uniform::new(0.0, 1.0).expect("r is positive and real");
    let trial_rotation = Uniform::new(-PI, PI).expect("hard-coded distribution should be valid");
    let theta = trial_rotation.sample(rng);
    let v: f64 = trial_boost.sample(rng);
    let (sh_v, ch_v, sn_th, cs_th) = (v.sinh(), v.cosh(), theta.sin(), theta.cos());
    let vertices = shape
        .vertices()
        .iter()
        .map(|v| {
            let v_transformed =
                HyperbolicConvexPolygon::vertex_to_system_frame(v, Angle::from(0.0), &origin);
            v_transformed.point().coordinates
        })
        .collect::<Vec<[f64; 3]>>();

    let translate = |pt: &[f64; 3]| -> Hyperbolic<3> {
        Hyperbolic::<3>::from_minkowski_coordinates(
            [
                pt[0] * (ch_v * cs_th * cs_th + sn_th * sn_th)
                    + pt[1] * sn_th * cs_th * (ch_v - 1.0)
                    + pt[2] * sh_v * cs_th,
                pt[0] * sn_th * cs_th * (ch_v - 1.0)
                    + pt[1] * (ch_v * sn_th * sn_th + cs_th * cs_th)
                    + pt[2] * sh_v * sn_th,
                pt[0] * sh_v * cs_th + pt[1] * sh_v * sn_th + pt[2] * ch_v,
            ]
            .into(),
        )
    };
    vertices
        .iter()
        .map(translate)
        .collect::<Vec<Hyperbolic<3>>>()
}

fn klein_overlap<R: Rng>(rng: &mut R, shape: &HyperbolicConvexPolygon) -> bool {
    let pos_1 = Hyperbolic::<3>::from_polar_coordinates(0.01, 0.2);
    let trial_boost = Uniform::new(0.0, 1.0).expect("r is positive and real");
    let trial_rotation = Uniform::new(-PI, PI).expect("hard-coded distribution should be valid");
    let theta = trial_rotation.sample(rng);
    let v: f64 = trial_boost.sample(rng);
    let pos_2 = Hyperbolic::<3>::from_polar_coordinates(v, theta);
    let (orientation_1, orientation_2) = (
        Angle::from(trial_rotation.sample(rng)),
        Angle::from(trial_rotation.sample(rng)),
    );

    shape.intersects_at_global(shape, &pos_1, &orientation_1, &pos_2, &orientation_2)
}

const NUM_VERTICES: &[usize] = &[3, 4, 5, 8, 16];

#[divan::bench(consts = NUM_VERTICES, sample_count = 1000)]
fn deck_transform_then_generate_vertices<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);
    let shape = HyperbolicConvexPolygon::regular(N, 0.5);

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(translate_then_gen_vertices(&mut rng, &shape)));
}

#[divan::bench(consts = NUM_VERTICES, sample_count = 1000)]
fn generate_vertices_then_transform<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);
    let shape = HyperbolicConvexPolygon::regular(N, 0.5);

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(gen_vertices_then_translate(&mut rng, &shape)));
}

#[divan::bench(consts = NUM_VERTICES, sample_count = 1000)]
fn overlap_check<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);
    let shape = HyperbolicConvexPolygon::regular(N, 0.25);

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(klein_overlap(&mut rng, &shape)));
}
