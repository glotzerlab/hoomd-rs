use hoomd_interaction::univariate::{UnivariateEnergy, LennardJones};

fn main() {
    let lennard_jones: LennardJones = LennardJones::default();
    println!("lennard_jones(1.5): {}", lennard_jones.energy(1.5));
}
