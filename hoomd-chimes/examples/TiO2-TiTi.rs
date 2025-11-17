/*! This is an example
*/
#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]
use hoomd_chimes::potential::{ChimesChebyshevExpansion, ChimesPenalty, TersoffSmooth};
use hoomd_chimes::transformation::MorseTransformation;
use hoomd_interaction::univariate::{UnivariateEnergy, UnivariateForce};
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
    let coeff_2b = vec![
        12.182_108_126_966_01,
        -2.473_627_738_301_203_3,
        8.236_322_683_724_822,
        -5.857_960_598_882_468,
        7.094_304_678_182_87,
        -3.228_348_403_842_029,
        4.459_762_350_244_618,
        -1.742_851_852_676_150_5,
        1.835_175_702_158_179_2,
        -0.658_390_741_787_121_9,
        0.561_064_966_268_623_5,
        -0.100_767_351_508_190_65,
    ];

    let morse_trans: MorseTransformation = MorseTransformation {
        lambda,
        r_out,
        r_in,
    };

    let chimes2b_cheby: ChimesChebyshevExpansion<MorseTransformation, 12> =
        ChimesChebyshevExpansion::new(morse_trans, coeff_2b, r_in);

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
    for r in &r_test {
        let energy = chimes_penalty.energy(*r) + chimes2b.energy(*r);
        let force = chimes_penalty.force(*r) + chimes2b.force(*r);
        writeln!(writer, "{r:.18}\t{energy:.18}\t{force:.18}")?;
    }

    // Ensure all data is written
    writer.flush()?;

    Ok(())
}
