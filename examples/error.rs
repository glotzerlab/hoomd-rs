use hoomd_microstate::{Body, Microstate, MicrostateBuilder, boundary::Square, property::Point};
use hoomd_vector::Cartesian;

use anyhow::Context;

fn my_method(
    microstate: &mut Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Square>,
) -> anyhow::Result<()> {
    microstate
        .add_body(Body::point(Cartesian::from([5.0, 0.0])))
        .context("Adding body in my_method")?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut microstate = MicrostateBuilder::with_boundary(Square {
        l: 10.0.try_into()?,
    })
    .bodies([Body::point(Cartesian::from([0.0, 0.0]))])
    .try_build()?;

    my_method(&mut microstate)?;

    Ok(())
}
