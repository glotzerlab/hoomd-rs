// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![allow(clippy::print_stdout, reason = "Provide status updates in tool output")]

/*! Tool that builds all examples with WASM for use in the web docs */

use std::fs;
use std::process::Command;

use anyhow::{Context, anyhow};
use clap::Parser;
use toml::{Table, Value};

/// Arguments
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Examples to build.
    examples: Vec<String>,

    /// Optimize for size.
    #[arg(long)]
    reduce_size: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let manifest_file =
        fs::read_to_string("examples/Cargo.toml").context("error reading examples/Cargo.toml")?;
    let cargo = manifest_file
        .parse::<Table>()
        .context("error parsing examples/Cargo.toml")?;
    let examples = cargo
        .get("package")
        .ok_or(anyhow!("package key not present"))?
        .get("metadata")
        .ok_or(anyhow!("package.metadata key not present"))?
        .get("example")
        .ok_or(anyhow!("package.metadata.example key not present"))?;

    if let Value::Table(table) = examples {
        build_examples(table, &args).context("failed to build examples")?;
    } else {
        return Err(anyhow!("package.metadata.example is not a table"));
    }

    Ok(())
}

/// Build the examples.
fn build_examples(table: &Table, args: &Args) -> anyhow::Result<()> {
    let profile = if args.reduce_size {
        "release-wasm"
    } else {
        "release"
    };

    // Issue one cargo build command to build examples in parallel.
    let mut cargo_build = Command::new("cargo");

    cargo_build
        .arg("build")
        .args(["--profile", profile])
        .args(["--features", "doc-example"])
        .args(["--target", "wasm32-unknown-unknown"]);

    for (name, _) in table {
        if args.examples.is_empty() || args.examples.contains(name) {
            cargo_build.args(["--example", name]);
        }
    }

    println!("\nBuilding examples...");
    let status = cargo_build.status().context("failed to build examples")?;

    if !status.success() {
        return Err(anyhow!("failed to build examples"));
    }

    println!("\nGenerating and optimizing bindings for...");
    for (name, configuration) in table {
        println!("{name}");
        let configuration = configuration
            .as_table()
            .ok_or(anyhow!("package.metadata.example.{name} is not a table"))?;
        let subpath = configuration
            .get("path")
            .ok_or(anyhow!("package.metadata.example.{name}.path not found"))?
            .as_str()
            .ok_or(anyhow!(
                "package.metadata.example.{name}.path is not a string"
            ))?;

        let path = format!("doc/src/{subpath}");
        let target_wasm = format!("./target/wasm32-unknown-unknown/{profile}/examples/{name}.wasm");

        let status = Command::new("wasm-bindgen")
            .arg("--no-typescript")
            .args(["--target", "web"])
            .args(["--out-dir", path.as_str()])
            .args(["--out-name", name])
            .arg(target_wasm)
            .status()
            .context(format!("failed to generate bindings for {name}"))?;

        if !status.success() {
            return Err(anyhow!("failed to generate bindings for {name}"));
        }

        if args.reduce_size {
            let output_wasm = format!("{path}/{name}_bg.wasm");
            let optimized_wasm = "optimized.wasm";

            let status = Command::new("wasm-opt")
                .arg("-Oz")
                .args(["--output", optimized_wasm])
                .arg(output_wasm.as_str())
                .status()
                .context(format!("failed to optimize {output_wasm}"))?;

            if !status.success() {
                return Err(anyhow!("failed to optimize {output_wasm}"));
            }

            fs::rename(optimized_wasm, output_wasm.as_str())
                .context(format!("failed to move {optimized_wasm} to {output_wasm}"))?;
        }
    }

    Ok(())
}
