#![allow(clippy::print_stdout, reason = "Demonstration purposes")]

/*! This is an example
*/

use hoomd_gsd::file_layer::{GsdFile, Mode};
use std::time::Instant;

fn main() -> Result<(), anyhow::Error> {
    let n_keys = 2;
    let max_frames = 100;

    let names: Vec<String> = (0..n_keys).map(|k| format!("key {k}")).collect();

    let open_time = Instant::now();
    let gsd_file = GsdFile::open("test.gsd", Mode::Read)?;
    let open_us = open_time.elapsed().as_secs_f64() / 1e-6;
    println!("Time to open: {open_us} microseconds");

    let n_frames_read = gsd_file.n_frames().min(max_frames);

    let read_time = Instant::now();

    let mut total_bytes = 0;
    for frame in 0..n_frames_read {
        for name in &names {
            #[expect(
                clippy::needless_collect,
                reason = "to measure the time it takes to read the whole array"
            )]
            let array = gsd_file
                .iter_scalars::<f64>(frame, name)?
                .collect::<Vec<_>>();
            total_bytes += array.len() * size_of::<f64>();
        }
    }
    let read_sec = read_time.elapsed().as_secs_f64();

    let time_per_key = read_sec / (n_keys * n_frames_read) as f64;
    let us_per_key = time_per_key / 1e-6;

    println!("Sequential latency   : {us_per_key} microseconds/key.");

    let mb_sec = total_bytes as f64 / 1024.0 / 1024.0 / read_sec;
    println!("Sequential throughput: {mb_sec} MB/s");

    Ok(())
}
