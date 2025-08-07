/*! This is an example
*/
#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]
use arrayvec::ArrayVec;
use hoomd_chimes::potential::{Chimes2b, ChimesPenalty, TersoffSmooth};
use hoomd_chimes::transformation::MorseTransformation;
use hoomd_interaction::pairwise::{IsotropicEnergy, IsotropicForce};
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() -> std::io::Result<()> {
    run()
}
fn run() -> std::io::Result<()> {
    let mut r_test: Vec<f64> = Vec::new(); // Or Vec<f32> for single precision
    let start = 2.0;
    let end = 4.5;
    let step = 0.1;

    let mut current_value = start;
    while current_value <= end {
        r_test.push(current_value);
        current_value += step;
    }

    let lambda = 3.0;
    let r_out = 4.3;
    let r_in = 2.5;
    let fo = 0.5;
    const N_COEFF: usize = 12;
    let coeff_2b: ArrayVec<f64, N_COEFF> = [
        12.18210812696601,
        -2.4736277383012033,
        8.236322683724822,
        -5.857960598882468,
        7.09430467818287,
        -3.228348403842029,
        4.459762350244618,
        -1.7428518526761505,
        1.8351757021581792,
        -0.6583907417871219,
        0.5610649662686235,
        -0.10076735150819065,
    ]
    .into_iter()
    .collect();

    let morse_trans: MorseTransformation = MorseTransformation {
        lambda,
        r_out,
        r_in,
    };

    let chimes2b_cheby: Chimes2b<MorseTransformation, N_COEFF> =
        Chimes2b::new(morse_trans, coeff_2b, r_in);

    let chimes2b = TersoffSmooth {
        f: chimes2b_cheby,
        r_out,
        r_in,
        fo,
    };

    let a = 1E+4;
    let dt = 0.01;

    let chimes_penalty = ChimesPenalty { r_in, a, dt };

    // Create output file
    let file = File::create("./hoomd-chimes/examples/rusty-chimes-TiO2-Ti.txt")?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "r\tenergy\tforce")?;

    // Write results
    for r in r_test.iter() {
        let energy = chimes_penalty.energy(*r) + chimes2b.energy(*r);
        let force = chimes_penalty.force(*r) + chimes2b.force(*r);
        writeln!(writer, "{:.18}\t{:.18}\t{:.18}", r, energy, force)?;
    }

    // Ensure all data is written
    writer.flush()?;

    Ok(())
}
