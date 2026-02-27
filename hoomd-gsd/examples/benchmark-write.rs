#![allow(clippy::print_stdout, reason = "Demonstration purposes")]

//! This is an example

use clap::Parser;
use hoomd_gsd::file_layer::GsdFile;
use std::time::Instant;

/// Arguments
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Number of keys to write per frame
    #[arg(short, long, default_value_t = 2)]
    n_keys: usize,

    /// f64 values per key
    #[arg(short, long, default_value_t = 2048)]
    key_size: usize,

    /// File size to write (in MB)
    #[arg(short, long, default_value_t = 256)]
    file_size: usize,

    /// Initial buffer size (in bytes)
    #[arg(long, default_value_t = 1024)]
    initial_buffer: usize,
}

/// Measure the performance of writing a file.
fn benchmark(
    buffer: usize,
    n_keys: usize,
    key_size: usize,
    file_size_mb: usize,
) -> Result<f64, anyhow::Error> {
    let target_file_size: usize = file_size_mb * 1024 * 1024;

    let n_frames = target_file_size / key_size / n_keys / size_of::<f64>();

    let data: Vec<f64> = (0..key_size).map(|x| x as f64).collect();
    let names: Vec<String> = (0..n_keys).map(|k| format!("key {k}")).collect();

    let mut gsd_file = GsdFile::create("test.gsd", "app", "schema", (0, 0))?;
    *gsd_file.maximum_write_buffer_size_mut() = buffer;

    let t1 = Instant::now();
    for _ in 0..n_frames {
        for name in &names {
            gsd_file.write_scalars(name, data.iter().copied() )?;
        }
        gsd_file.end_frame()?;
    }
    gsd_file.sync_all()?;

    let time_span = t1.elapsed().as_secs_f64();

    let time_per_key = time_span / n_keys as f64 / n_frames as f64;

    let mb_per_second = (key_size * 8 + 32) as f64 / 1_048_576.0 / time_per_key;

    drop(gsd_file);

    Ok(mb_per_second)
}

fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();
    let mut buffer = args.initial_buffer;

    println!("[");
    while buffer <= 64 * 1024 * 1024 {
        let mb_per_sec = benchmark(buffer, args.n_keys, args.key_size, args.file_size)?;
        println!("[{buffer}, {mb_per_sec}],");
        buffer *= 2;
    }
    println!("]");

    Ok(())
}
