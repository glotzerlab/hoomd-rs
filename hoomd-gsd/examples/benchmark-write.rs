#![allow(clippy::print_stdout, reason = "Demonstration purposes")]

/*! This is an example
*/

use hoomd_gsd::file_layer::GsdFile;
use std::time::Instant;

/// Measure the performance of writing a file.
fn benchmark(buffer: usize) -> Result<f64, anyhow::Error> {
    let n_keys: usize = 2;
    let key_size: usize = 2048;
    let target_file_size: usize = 256 * 1024 * 1024;

    let n_frames = target_file_size / key_size / n_keys / size_of::<f64>();

    let data: Vec<f64> = (0..key_size).map(|x| x as f64).collect();
    let names: Vec<String> = (0..n_keys).map(|k| format!("key {k}")).collect();

    let mut gsd_file = GsdFile::create("test.gsd", "app", "schema", (1,0))?;
    *gsd_file.sync_threshold_mut() = buffer;

    let t1 = Instant::now();
    for _ in 0..n_frames {
        for name in &names {
            gsd_file.write_array(name, 1, &data)?;
            }
        gsd_file.end_frame()?;
        }
    gsd_file.sync_all()?;

    let time_span = t1.elapsed().as_secs_f64();

    let time_per_key = time_span / n_keys as f64 / n_frames as f64;

    let mb_per_second
        = (key_size * 8 + 32) as f64 / 1_048_576.0 / time_per_key;

    drop(gsd_file);

    Ok(mb_per_second)
    }

fn main() -> Result<(), anyhow::Error> {
    let mut buffer = 1024 * 1024;

    println!("[");
    while buffer <= 64 * 1024 * 1024
        {
        let mb_per_sec = benchmark(buffer)?;
        println!("[{buffer}, {mb_per_sec}],");
        buffer *= 2;
        }
    println!("]");

    Ok(())
    }

