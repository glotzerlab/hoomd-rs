// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]
#![expect(clippy::wildcard_imports, reason = "simplifies code")]
#![expect(
    clippy::needless_pass_by_value,
    reason = "divan takes Bencher by value"
)]

//! Benchmark periodic ghost generation for [`Hypercuboid`].
//!
//! The crate ships a single generic `GenerateGhosts` impl for
//! `Periodic<Hypercuboid<N>>` that enumerates periodic images with a bitmask
//! loop. This benchmark compares it to the hand-unrolled, dimension-specialized
//! 2D/3D implementations it replaced.
//!
//! The buffer capacity is the crate's real `MAX_GHOSTS` (set at build time via
//! `HOOMD_MAX_GHOSTS`), read here from the same environment variable so the
//! self-contained `specific_*` variants allocate the *same* buffer the shipped
//! `generate_ghosts` does. Because `MAX_GHOSTS` is a single global constant,
//! each dimension is measured at its own optimal capacity by running the bench
//! separately with `HOOMD_MAX_GHOSTS = 2^N - 1` and filtering to that
//! dimension's rows (e.g. `HOOMD_MAX_GHOSTS=7 cargo bench --bench cuboid -- 3d`).
//!
//! * `trait_method` — the shipped `generate_ghosts` (the new generic bitmask
//!   algorithm). Run for the dimension matching the configured `MAX_GHOSTS`.
//! * `specific` — the original hand-unrolled dimension-specialized algorithm,
//!   reimplemented here for direct comparison.
//!
//! Each is measured on a `corner` site (the worst case, `2^N - 1` ghosts) and an
//! `interior` site (the common case, no ghosts).

use arrayvec::ArrayVec;
use divan::{Bencher, black_box};
use hoomd_geometry::{IsPointInside, shape::Hypercuboid};
use hoomd_microstate::{
    boundary::{GenerateGhosts, Periodic},
    property::Point,
};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Cartesian;

fn main() {
    divan::main();
}

/// The crate's `MAX_GHOSTS`, read from the same `HOOMD_MAX_GHOSTS` build-time
/// environment variable so the bench and the shipped code agree on the buffer
/// size.
const MAX_GHOSTS: usize = match option_env!("HOOMD_MAX_GHOSTS") {
    // `from_str` is not const, so manually parse like the crate does.
    Some(val) => match usize::from_str_radix(val, 10) {
        Ok(n) => n,
        Err(_) => panic!("HOOMD_MAX_GHOSTS must be a non-negative integer"),
    },
    None => 12,
};

/// Edge length and interaction range used by every benchmark.
const EDGE: f64 = 20.0;
const RANGE: f64 = 1.0;

/// Build a `PositiveReal` from a hard-coded constant.
fn pos(value: f64) -> PositiveReal {
    value
        .try_into()
        .expect("hard-coded constant should be positive")
}

/// Build a cubic periodic box.
fn periodic_box<const N: usize>(edge: f64, range: f64) -> Periodic<Hypercuboid<N>> {
    let shape = Hypercuboid::with_equal_edges(pos(edge));
    Periodic::new(range, shape).expect("hard-coded range should be valid")
}

// -------------------------------------------------------------------------------------------------
// The original hand-unrolled, dimension-specialized implementations.
//
// These are verbatim reimplementations of the pre-generalization `generate_ghosts`
// bodies, so they measure exactly the algorithm the generic impl replaced.
// `min`/`max` are hoisted once, matching the originals.
// -------------------------------------------------------------------------------------------------

/// The original hand-unrolled 2D implementation.
fn specific_2d(
    shape: &Hypercuboid<2>,
    range: f64,
    site: &Point<Cartesian<2>>,
) -> ArrayVec<Point<Cartesian<2>>, MAX_GHOSTS> {
    let mut result = ArrayVec::new();
    let r = site.position;
    if !shape.is_point_inside(&r) {
        return result;
    }

    let max = shape.maximal_extents();
    let min = shape.minimal_extents();

    let new_site = |x: f64, y: f64| {
        let mut new_site = *site;
        new_site.position[0] += x * shape.edge_lengths[0].get();
        new_site.position[1] += y * shape.edge_lengths[1].get();
        new_site
    };

    let near_left = r[0] < min[0] + range;
    let near_right = r[0] > max[0] - range;
    let near_top = r[1] > max[1] - range;
    let near_bottom = r[1] < min[1] + range;

    if near_right {
        result.push(new_site(-1.0, 0.0));
    }
    if near_left {
        result.push(new_site(1.0, 0.0));
    }
    if near_top {
        result.push(new_site(0.0, -1.0));
    }
    if near_bottom {
        result.push(new_site(0.0, 1.0));
    }
    if near_right && near_top {
        result.push(new_site(-1.0, -1.0));
    }
    if near_right && near_bottom {
        result.push(new_site(-1.0, 1.0));
    }
    if near_left && near_top {
        result.push(new_site(1.0, -1.0));
    }
    if near_left && near_bottom {
        result.push(new_site(1.0, 1.0));
    }

    result
}

/// The original hand-unrolled 3D implementation.
#[allow(
    clippy::too_many_lines,
    reason = "mirrors the original hand-unrolled code"
)]
fn specific_3d(
    shape: &Hypercuboid<3>,
    range: f64,
    site: &Point<Cartesian<3>>,
) -> ArrayVec<Point<Cartesian<3>>, MAX_GHOSTS> {
    let mut result = ArrayVec::new();
    let r = site.position;
    if !shape.is_point_inside(&r) {
        return result;
    }

    let max = shape.maximal_extents();
    let min = shape.minimal_extents();

    let new_site = |x: f64, y: f64, z: f64| {
        let mut new_site = *site;
        new_site.position[0] += x * shape.edge_lengths[0].get();
        new_site.position[1] += y * shape.edge_lengths[1].get();
        new_site.position[2] += z * shape.edge_lengths[2].get();
        new_site
    };

    let near_left = r[0] < min[0] + range;
    let near_right = r[0] > max[0] - range;
    let near_top = r[1] > max[1] - range;
    let near_bottom = r[1] < min[1] + range;
    let near_front = r[2] > max[2] - range;
    let near_back = r[2] < min[2] + range;

    if near_right {
        result.push(new_site(-1.0, 0.0, 0.0));
    }
    if near_left {
        result.push(new_site(1.0, 0.0, 0.0));
    }
    if near_top {
        result.push(new_site(0.0, -1.0, 0.0));
    }
    if near_bottom {
        result.push(new_site(0.0, 1.0, 0.0));
    }
    if near_front {
        result.push(new_site(0.0, 0.0, -1.0));
    }
    if near_back {
        result.push(new_site(0.0, 0.0, 1.0));
    }

    if near_right && near_top {
        result.push(new_site(-1.0, -1.0, 0.0));
    }
    if near_right && near_bottom {
        result.push(new_site(-1.0, 1.0, 0.0));
    }
    if near_right && near_front {
        result.push(new_site(-1.0, 0.0, -1.0));
    }
    if near_right && near_back {
        result.push(new_site(-1.0, 0.0, 1.0));
    }
    if near_left && near_top {
        result.push(new_site(1.0, -1.0, 0.0));
    }
    if near_left && near_bottom {
        result.push(new_site(1.0, 1.0, 0.0));
    }
    if near_left && near_front {
        result.push(new_site(1.0, 0.0, -1.0));
    }
    if near_left && near_back {
        result.push(new_site(1.0, 0.0, 1.0));
    }

    if near_top && near_front {
        result.push(new_site(0.0, -1.0, -1.0));
    }
    if near_bottom && near_front {
        result.push(new_site(0.0, 1.0, -1.0));
    }
    if near_top && near_back {
        result.push(new_site(0.0, -1.0, 1.0));
    }
    if near_bottom && near_back {
        result.push(new_site(0.0, 1.0, 1.0));
    }

    if near_right && near_top && near_front {
        result.push(new_site(-1.0, -1.0, -1.0));
    }
    if near_right && near_top && near_back {
        result.push(new_site(-1.0, -1.0, 1.0));
    }
    if near_right && near_bottom && near_front {
        result.push(new_site(-1.0, 1.0, -1.0));
    }
    if near_right && near_bottom && near_back {
        result.push(new_site(-1.0, 1.0, 1.0));
    }
    if near_left && near_top && near_front {
        result.push(new_site(1.0, -1.0, -1.0));
    }
    if near_left && near_top && near_back {
        result.push(new_site(1.0, -1.0, 1.0));
    }
    if near_left && near_bottom && near_front {
        result.push(new_site(1.0, 1.0, -1.0));
    }
    if near_left && near_bottom && near_back {
        result.push(new_site(1.0, 1.0, 1.0));
    }

    result
}

// -------------------------------------------------------------------------------------------------
// Benchmarks.
// -------------------------------------------------------------------------------------------------

/// The shipped `generate_ghosts` (the new generic bitmask algorithm).
#[divan::bench_group]
mod trait_method {
    use super::*;

    macro_rules! corner {
        ($($name:ident at $n:literal),+ $(,)?) => {
            $(#[divan::bench]
            fn $name(bencher: Bencher) {
                let periodic = periodic_box::<$n>(EDGE, RANGE);
                let coord = EDGE / 2.0 - RANGE / 2.0;
                bencher
                    .with_inputs(|| Cartesian::<$n>::from([coord; $n]))
                    .bench_local_values(|pt| {
                        black_box(periodic.generate_ghosts(&Point::new(pt)))
                    });
            })+
        };
    }

    macro_rules! interior {
        ($($name:ident at $n:literal),+ $(,)?) => {
            $(#[divan::bench]
            fn $name(bencher: Bencher) {
                let periodic = periodic_box::<$n>(EDGE, RANGE);
                bencher
                    .with_inputs(|| Cartesian::<$n>::from([0.0; $n]))
                    .bench_local_values(|pt| {
                        black_box(periodic.generate_ghosts(&Point::new(pt)))
                    });
            })+
        };
    }

    corner! {
        corner_1d at 1, corner_2d at 2, corner_3d at 3, corner_4d at 4, corner_5d at 5,
    }

    interior! {
        interior_1d at 1, interior_2d at 2, interior_3d at 3, interior_4d at 4, interior_5d at 5,
    }
}

/// The original hand-unrolled dimension-specialized implementations.
#[divan::bench_group]
mod specific {
    use super::*;

    macro_rules! corner {
        ($name:ident, $n:literal, $alg:path) => {
            #[divan::bench]
            fn $name(bencher: Bencher) {
                let shape = Hypercuboid::<$n>::with_equal_edges(pos(EDGE));
                let coord = EDGE / 2.0 - RANGE / 2.0;
                bencher
                    .with_inputs(|| Cartesian::<$n>::from([coord; $n]))
                    .bench_local_values(|pt| black_box($alg(&shape, RANGE, &Point::new(pt))));
            }
        };
    }

    macro_rules! interior {
        ($name:ident, $n:literal, $alg:path) => {
            #[divan::bench]
            fn $name(bencher: Bencher) {
                let shape = Hypercuboid::<$n>::with_equal_edges(pos(EDGE));
                bencher
                    .with_inputs(|| Cartesian::<$n>::from([0.0; $n]))
                    .bench_local_values(|pt| black_box($alg(&shape, RANGE, &Point::new(pt))));
            }
        };
    }

    corner! { corner_2d, 2, specific_2d }
    corner! { corner_3d, 3, specific_3d }
    interior! { interior_2d, 2, specific_2d }
    interior! { interior_3d, 3, specific_3d }
}
