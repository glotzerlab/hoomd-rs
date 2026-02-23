// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![allow(
    clippy::missing_inline_in_public_items,
    reason = "GSD methods are not meant to be customized"
)]

//! Read and write GSD files.
//!
//! # GSD files
//!
//! A GSD file stores 2D arrays of integer and floating point types in named chunks
//! that are associated with trajectory frames. The [GSD Python package] can read
//! and write these files. `hoomd-gsd` implements GSD file I/O in native Rust.
//!
//! [GSD Python package]: https://gsd.readthedocs.io
//!
//! # The file layer
//!
//! [`GsdFile`](file_layer::GsdFile) provides direct access to read and write GSD
//! formatted files. Call [`create_new`](file_layer::GsdFile::create_new) to create
//! a new GSD file:
//!
//! ```
//! use hoomd_gsd::file_layer::GsdFile;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use tempfile::tempdir;
//! # let tmp_dir = tempdir().expect("temp dir should be created");
//! # let path = tmp_dir.path().join("test.gsd");
//! // let path = "file.gsd";
//! let mut gsd_file = GsdFile::create_new(path, "example", "hoomd", (1, 4))?;
//! # Ok(())
//! # }
//! ```
//!
//! Add new arrays to the current frame with
//! [`write_scalars`](file_layer::GsdFile::write_scalars) and
//! [`write_arrays`](file_layer::GsdFile::write_arrays). You **must** end the frame
//! with [`end_frame`](file_layer::GsdFile::end_frame) or no data will be written to
//! the file!
//! ```
//! use hoomd_gsd::file_layer::GsdFile;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use tempfile::tempdir;
//! # let tmp_dir = tempdir().expect("temp dir should be created");
//! # let path = tmp_dir.path().join("test.gsd");
//! let position = vec![[5.0_f32, 3.0, -4.0], [-2.0, 3.0, -6.0]];
//!
//! let mut gsd_file = GsdFile::create_new(path, "example", "hoomd", (1, 4))?;
//! gsd_file.write_scalars("configuration/step", &[100_000_u64])?;
//! gsd_file.write_scalars(
//!     "configuration/box",
//!     &[10.0_f32, 20.0, 15.0, 0.0, 0.0, 0.0],
//! )?;
//! gsd_file.write_arrays("particles/position", &position)?;
//! gsd_file.end_frame()?;
//! # Ok(())
//! # }
//! ```
//! Each array in the file in stored in a specific type. `write_scalars` and
//! `write_arrays` automatically infer that type from the argument given.
//!
//! # HOOMD schema
//!
//! See the [GSD Python package] documentation for a full specification of the HOOMD
//! schema. Files written with this schema will interoperate with [HOOMD-blue],
//! [OVITO], and other applications.
//!
//! [HOOMD-blue]: https://hoomd-blue.readthedocs.io
//! [Ovito]: https://www.ovito.org
//!
//! At this time, the `hoomd-gsd` crate does not provide any high level API for
//! reading or writing the HOOMD schema. See the code examples throughout the
//! [`file_layer`] module for minimal examples that write files with the HOOMD
//! schema. Note that the HOOMD schema uses f32 data types, so convert appropriately
//! when mapping vectors from `hoomd-vector`.

pub mod file_layer;
