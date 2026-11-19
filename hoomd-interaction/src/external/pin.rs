// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`PinOrientations`]

use hoomd_microstate::{
    Body, Microstate,
    boundary::GenerateMinimumImageCandidates,
    property::{Orientation, Position},
};
use hoomd_vector::{Metric, RotationalDistance};

use crate::{DeltaEnergyOne, MaximumInteractionRange, TotalEnergy};

/// Connect each body orientation to a reference with a harmonic spring.
///
/// ```math
/// U_\mathrm{body} = \begin{cases}
/// \min_j \frac{1}{2} k (\Delta \theta_j)^2 & n_\mathrm{references} > 0 \\
/// 0 & n_\mathrm{references} = 0 \\
/// \end{cases}
/// ```
///
/// Apply a harmonic external potential to the angle between the body's
/// orientation and a given reference orientation. Take the minimum over all
/// symmetric references *j*. $`\Delta \theta_j`$ is the smallest angle that
/// rotates the body to reference *j* (as given by `R`'s
/// [`RotationalDistance`] implementation).
///
/// If there is no reference orientation for a body, the energy is 0.
///
/// Use [`PinOrientations`] for Frenkel-Ladd free energy computations or any
/// time you want to loosely pin body orientations in place. To fix
/// body orientations exactly at the initial condition, do not apply rotational
/// trial moves.
///
/// [`PinOrientations`] stores references in *body tag order*. It is thus not
/// suitable for simulations that remove bodies from the microstate. If you
/// need to pin orientations *and* remove bodies, write a custom external
/// potential that uses a custom `reference` stored in the body properties.
pub struct PinOrientations<R> {
    /// The spring constant $`[\mathrm{energy}]`$.
    pub k: f64,
    /// Set of reference orientations for each body (in tag order).
    pub reference: Vec<Vec<R>>,
}

impl<R> PinOrientations<R> {
    /// Compute the energy for a given body.
    ///
    /// ```math
    /// U_\mathrm{body} = \begin{cases}
    /// \min_j \frac{1}{2} k (\Delta \theta_j)^2 & n_\mathrm{references} > 0 \\
    /// 0 & n_\mathrm{references} = 0 \\
    /// \end{cases}
    /// ```
    ///
    /// The body must exist in the microstate with the given `body_tag`.
    /// The `body_properties` may match those in the microstate or may be
    /// modified (i.e. a trial move).
    #[inline]
    pub fn body_energy<B>(&self, body_tag: usize, body_properties: &B) -> f64
    where
        R: RotationalDistance,
        B: Orientation<Rotation = R>,
    {
        self.reference[body_tag]
            .iter()
            .map(|reference_orientation| {
                0.5 * self.k
                    * body_properties
                        .orientation()
                        .rotational_distance(reference_orientation)
                        .powi(2)
            })
            .reduce(f64::min)
            .unwrap_or(0.0)
    }
}

impl<B, S, X, C, R> DeltaEnergyOne<B, S, X, C> for PinOrientations<R>
where
    R: RotationalDistance,
    B: Orientation<Rotation = R>,
{
    #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
        let tagged_body = &initial_microstate.bodies()[body_index];
        let body_tag = tagged_body.tag;

        let energy_initial = self.body_energy(body_tag, &tagged_body.item.properties);
        let energy_final = self.body_energy(body_tag, &final_body.properties);

        energy_final - energy_initial
    }
}

impl<B, S, X, C, R> TotalEnergy<Microstate<B, S, X, C>> for PinOrientations<R>
where
    R: RotationalDistance,
    B: Orientation<Rotation = R>,
{
    #[inline]
    fn total_energy(&self, microstate: &Microstate<B, S, X, C>) -> f64 {
        microstate
            .bodies()
            .iter()
            .map(|tagged_body| self.body_energy(tagged_body.tag, &tagged_body.item.properties))
            .sum()
    }
}

impl<R> MaximumInteractionRange for PinOrientations<R> {
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        // External interactions are not applied between pairs of particles.
        0.0
    }
}

/// Connect each body position to a reference with a harmonic spring.
///
/// ```math
/// U_\mathrm{body} = \begin{cases}
/// \min_j \frac{1}{2} k (\Delta \vec{r}_j)^2 & n_\mathrm{references} > 0 \\
/// 0 & n_\mathrm{references} = 0 \\
/// \end{cases}
/// ```
///
/// Apply a harmonic external potential to the separation distance between a
/// body's position and a reference position. Take the minimum over all
/// symmetric references *j* given by any periodic boundary conditions
/// in effect.
///
/// Use [`PinPositions`] for Frenkel-Ladd free energy computations or any
/// time you want to loosely pin body positions in place. To fix
/// body positions exactly at the initial condition, do not apply translational
/// trial moves.
///
/// [`PinPositions`] stores references in *body tag order*. It is thus not
/// suitable for simulations that remove bodies from the microstate. If you
/// need to pin positions *and* remove bodies, write a custom external
/// potential that uses a custom `reference` stored in the body properties.
pub struct PinPositions<V> {
    /// The spring constant $`[\mathrm{energy}] \cdot [\mathrm{length}]^{-2}`$.
    pub k: f64,
    /// Reference position for each body (in tag order).
    pub reference: Vec<V>,
}

impl<P> PinPositions<P>
where
    P: Metric,
{
    /// Compute the energy for a given body.
    ///
    /// ```math
    /// U_\mathrm{body} = \begin{cases}
    /// \min_j \frac{1}{2} k (\Delta \vec{r}_j)^2 & n_\mathrm{references} > 0 \\
    /// 0 & n_\mathrm{references} = 0 \\
    /// \end{cases}
    /// ```
    ///
    /// The body must exist in the microstate with the given `body_tag`.
    /// The `body_properties` may match those in the microstate or may be
    /// modified (i.e. a trial move). Pass the microstate's boundary
    /// condition to `boundary`. It will be used to generate the candidate
    /// minimum images.
    #[inline]
    pub fn body_energy<B, C>(&self, body_tag: usize, body_properties: &B, boundary: &C) -> f64
    where
        B: Position<Position = P>,
        C: GenerateMinimumImageCandidates<B>,
    {
        let reference = &self.reference[body_tag];
        let ghosts = boundary.generate_minimum_image_candidates(body_properties);
        std::iter::once(body_properties.position())
            .chain(
                ghosts
                    .iter()
                    .map(hoomd_microstate::property::Position::position),
            )
            .map(|position| 0.5 * self.k * position.distance_squared(reference))
            .reduce(f64::min)
            .unwrap_or(0.0)
    }
}

impl<B, S, X, C, P> DeltaEnergyOne<B, S, X, C> for PinPositions<P>
where
    B: Position<Position = P>,
    C: GenerateMinimumImageCandidates<B>,
    P: Metric,
{
    #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
        let tagged_body = &initial_microstate.bodies()[body_index];
        let body_tag = tagged_body.tag;
        let boundary = initial_microstate.boundary();

        let energy_initial = self.body_energy(body_tag, &tagged_body.item.properties, boundary);
        let energy_final = self.body_energy(body_tag, &final_body.properties, boundary);

        energy_final - energy_initial
    }
}

impl<B, S, X, C, P> TotalEnergy<Microstate<B, S, X, C>> for PinPositions<P>
where
    B: Position<Position = P>,
    C: GenerateMinimumImageCandidates<B>,
    P: Metric,
{
    #[inline]
    fn total_energy(&self, microstate: &Microstate<B, S, X, C>) -> f64 {
        let boundary = microstate.boundary();
        microstate
            .bodies()
            .iter()
            .map(|tagged_body| {
                self.body_energy(tagged_body.tag, &tagged_body.item.properties, boundary)
            })
            .sum()
    }
}

impl<V> MaximumInteractionRange for PinPositions<V> {
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        // External interactions are not applied between pairs of particles.
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use assert2::check;
    use hoomd_geometry::shape::Rectangle;
    use hoomd_microstate::{
        boundary::{Open, Periodic},
        property::{OrientedPoint, Point},
    };
    use hoomd_vector::{Angle, Cartesian, Rotation, Versor};
    use rstest::*;
    use std::f64::consts::PI;

    mod pin_orientations {
        use super::*;

        fn oriented_body<R>(
            orientation: R,
        ) -> Body<OrientedPoint<Cartesian<2>, R>, Point<Cartesian<2>>> {
            Body {
                properties: OrientedPoint {
                    position: Cartesian::default(),
                    orientation,
                },
                sites: [Point::new(Cartesian::default())].into(),
            }
        }

        #[test]
        fn body_energy_no_reference() {
            let pin = PinOrientations::<Angle> {
                k: 2.0,
                reference: vec![vec![]],
            };
            let body = oriented_body(Angle::from(1.0));

            check!(pin.body_energy(0, &body.properties) == 0.0);
        }

        #[rstest]
        #[case(0.0, 0.0)]
        #[case(0.5, 0.25)]
        #[case(-0.5, 0.25)]
        #[case(2.0 * PI - 0.1, 0.01)]
        #[case(PI, PI * PI)]
        fn body_energy_angle(#[case] theta: f64, #[case] expected: f64) {
            let pin = PinOrientations {
                k: 2.0,
                reference: vec![vec![Angle::from(0.0)]],
            };
            let body = oriented_body(Angle::from(theta));

            assert_relative_eq!(
                pin.body_energy(0, &body.properties),
                expected,
                epsilon = 1e-12
            );
        }

        #[test]
        fn body_energy_minimum_over_references() {
            let pin = PinOrientations {
                k: 3.0,
                reference: vec![vec![
                    Angle::from(0.0),
                    Angle::from(PI / 2.0),
                    Angle::from(PI),
                    Angle::from(3.0 * PI / 2.0),
                ]],
            };
            let body = oriented_body(Angle::from(PI / 2.0 + 0.2));

            assert_relative_eq!(
                pin.body_energy(0, &body.properties),
                0.5 * 3.0 * 0.2 * 0.2,
                epsilon = 1e-12
            );
        }

        #[test]
        fn body_energy_versor() {
            let pin = PinOrientations {
                k: 1.0,
                reference: vec![vec![Versor::identity()]],
            };
            let orientation = Versor::from_axis_angle(
                [0.0, 0.0, 1.0]
                    .try_into()
                    .expect("hard-coded vector should have non-zero length"),
                PI / 2.0,
            );
            let properties = OrientedPoint {
                position: Cartesian::<3>::default(),
                orientation,
            };

            assert_relative_eq!(
                pin.body_energy(0, &properties),
                0.5 * (PI / 2.0).powi(2),
                epsilon = 1e-12
            );
        }

        #[test]
        fn total_energy() -> anyhow::Result<()> {
            let pin = PinOrientations {
                k: 2.0,
                reference: vec![vec![Angle::from(0.0)], vec![Angle::from(1.0)]],
            };
            let microstate = Microstate::builder()
                .bodies([
                    oriented_body(Angle::from(0.5)),
                    oriented_body(Angle::from(0.7)),
                ])
                .try_build()?;

            assert_relative_eq!(
                pin.total_energy(&microstate),
                0.5 * 0.5 + 0.3 * 0.3,
                epsilon = 1e-12
            );
            Ok(())
        }

        #[test]
        fn total_energy_uses_tag() -> anyhow::Result<()> {
            let pin = PinOrientations {
                k: 2.0,
                reference: vec![
                    vec![Angle::from(0.0)],
                    vec![Angle::from(1.0)],
                    vec![Angle::from(2.0)],
                ],
            };
            let mut microstate = Microstate::builder()
                .bodies([
                    oriented_body(Angle::from(0.0)),
                    oriented_body(Angle::from(0.9)),
                    oriented_body(Angle::from(2.2)),
                ])
                .try_build()?;
            microstate.remove_body(0);
            check!(microstate.bodies()[0].tag != 0);

            assert_relative_eq!(
                pin.total_energy(&microstate),
                0.1 * 0.1 + 0.2 * 0.2,
                epsilon = 1e-12
            );
            Ok(())
        }

        #[test]
        fn delta_energy_one() -> anyhow::Result<()> {
            let pin = PinOrientations {
                k: 2.0,
                reference: vec![
                    vec![Angle::from(0.0)],
                    vec![Angle::from(1.0)],
                    vec![Angle::from(2.0)],
                ],
            };
            let mut microstate = Microstate::builder()
                .bodies([
                    oriented_body(Angle::from(0.1)),
                    oriented_body(Angle::from(0.9)),
                    oriented_body(Angle::from(2.2)),
                ])
                .try_build()?;

            let final_body = oriented_body(Angle::from(0.3));
            assert_relative_eq!(
                pin.delta_energy_one(&microstate, 0, &final_body),
                0.3 * 0.3 - 0.1 * 0.1,
                epsilon = 1e-12
            );

            // After removal, index 0 holds a body with a different tag.
            microstate.remove_body(0);
            let tag = microstate.bodies()[0].tag;
            let initial_theta = [0.1, 0.9, 2.2][tag];
            let final_theta = 2.0 + 0.4;
            let final_body = oriented_body(Angle::from(final_theta));
            let reference = [0.0, 1.0, 2.0][tag];
            assert_relative_eq!(
                pin.delta_energy_one(&microstate, 0, &final_body),
                (final_theta - reference).powi(2) - (initial_theta - reference).powi(2),
                epsilon = 1e-12
            );
            Ok(())
        }

        #[test]
        fn maximum_interaction_range() {
            let pin = PinOrientations::<Angle> {
                k: 1.0,
                reference: vec![],
            };
            check!(pin.maximum_interaction_range() == 0.0);
        }
    }

    mod pin_positions {
        use super::*;

        #[test]
        fn body_energy_open() {
            let pin = PinPositions {
                k: 2.0,
                reference: vec![Cartesian::from([0.0, 0.0])],
            };
            let body = Body::point(Cartesian::from([3.0, 4.0]));

            check!(pin.body_energy(0, &body.properties, &Open) == 25.0);
        }

        #[test]
        fn body_energy_periodic_minimum_image() -> anyhow::Result<()> {
            let boundary = Periodic::new(1.0, Rectangle::with_equal_edges(10.0.try_into()?))?;
            let pin = PinPositions {
                k: 2.0,
                reference: vec![Cartesian::from([4.5, 0.0]), Cartesian::from([4.5, 4.5])],
            };

            let body = Body::point(Cartesian::from([-4.5, 0.0]));
            assert_relative_eq!(
                pin.body_energy(0, &body.properties, &boundary),
                1.0,
                epsilon = 1e-12
            );
            // Without the periodic boundary, the full distance applies.
            assert_relative_eq!(
                pin.body_energy(0, &body.properties, &Open),
                81.0,
                epsilon = 1e-12
            );

            let body = Body::point(Cartesian::from([3.5, 0.0]));
            assert_relative_eq!(
                pin.body_energy(0, &body.properties, &boundary),
                1.0,
                epsilon = 1e-12
            );

            // Wrap across a corner.
            let body = Body::point(Cartesian::from([-4.5, -4.5]));
            assert_relative_eq!(
                pin.body_energy(1, &body.properties, &boundary),
                2.0,
                epsilon = 1e-12
            );
            Ok(())
        }

        #[test]
        fn total_energy() -> anyhow::Result<()> {
            let pin = PinPositions {
                k: 2.0,
                reference: vec![Cartesian::from([0.0, 0.0]), Cartesian::from([1.0, 1.0])],
            };
            let microstate = Microstate::builder()
                .bodies([
                    Body::point(Cartesian::from([3.0, 4.0])),
                    Body::point(Cartesian::from([1.0, 2.0])),
                ])
                .try_build()?;

            check!(pin.total_energy(&microstate) == 26.0);
            Ok(())
        }

        #[test]
        fn total_energy_uses_tag() -> anyhow::Result<()> {
            let pin = PinPositions {
                k: 2.0,
                reference: vec![
                    Cartesian::from([0.0, 0.0]),
                    Cartesian::from([1.0, 0.0]),
                    Cartesian::from([2.0, 0.0]),
                ],
            };
            let mut microstate = Microstate::builder()
                .bodies([
                    Body::point(Cartesian::from([0.0, 0.0])),
                    Body::point(Cartesian::from([1.0, 1.0])),
                    Body::point(Cartesian::from([2.0, 2.0])),
                ])
                .try_build()?;
            microstate.remove_body(0);
            check!(microstate.bodies()[0].tag != 0);

            check!(pin.total_energy(&microstate) == 5.0);
            Ok(())
        }

        #[test]
        fn delta_energy_one_open() -> anyhow::Result<()> {
            let pin = PinPositions {
                k: 2.0,
                reference: vec![Cartesian::from([0.0, 0.0]), Cartesian::from([1.0, 0.0])],
            };
            let mut microstate = Microstate::builder()
                .bodies([
                    Body::point(Cartesian::from([1.0, 0.0])),
                    Body::point(Cartesian::from([1.0, 1.0])),
                ])
                .try_build()?;

            let final_body = Body::point(Cartesian::from([3.0, 4.0]));
            check!(pin.delta_energy_one(&microstate, 0, &final_body) == 24.0);

            // After removal, index 0 holds the body with tag 1.
            microstate.remove_body(0);
            check!(microstate.bodies()[0].tag == 1);
            let final_body = Body::point(Cartesian::from([1.0, 3.0]));
            check!(pin.delta_energy_one(&microstate, 0, &final_body) == 8.0);
            Ok(())
        }

        #[test]
        fn delta_energy_one_periodic() -> anyhow::Result<()> {
            let boundary = Periodic::new(1.0, Rectangle::with_equal_edges(10.0.try_into()?))?;
            let pin = PinPositions {
                k: 2.0,
                reference: vec![Cartesian::from([4.5, 0.0])],
            };
            let microstate = Microstate::builder()
                .boundary(boundary)
                .bodies([Body::point(Cartesian::from([3.5, 0.0]))])
                .try_build()?;

            // Move across the boundary: the minimum image is 0.5 from the reference.
            let final_body = Body::point(Cartesian::from([-5.0, 0.0]));
            assert_relative_eq!(
                pin.delta_energy_one(&microstate, 0, &final_body),
                0.25 - 1.0,
                epsilon = 1e-12
            );
            Ok(())
        }

        #[test]
        fn maximum_interaction_range() {
            let pin = PinPositions::<Cartesian<2>> {
                k: 1.0,
                reference: vec![],
            };
            check!(pin.maximum_interaction_range() == 0.0);
        }
    }
}
