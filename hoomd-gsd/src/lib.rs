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
//! # HOOMD schema
//!
//! Use [`HoomdGsdFile`] to write to GSD files with the HOOMD schema that can
//! be read by the [Ovito], [HOOMD-blue], the [GSD Python package], and other applications.
//!
//! [`HoomdGsdFile`]: hoomd::HoomdGsdFile
//! [GSD Python package]: https://gsd.readthedocs.io
//! [HOOMD-blue]: https://hoomd-blue.readthedocs.io
//! [Ovito]: https://www.ovito.org
//!
//! Create a new GSD file with the hoomd schema:
//!
//! ```
//! use hoomd_gsd::hoomd::HoomdGsdFile;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use tempfile::tempdir;
//! # let tmp_dir = tempdir().expect("temp dir should be created");
//! # let path = tmp_dir.path().join("test.gsd");
//! // let path = "file.gsd";
//! let hoomd_gsd_file = HoomdGsdFile::create(path)?;
//! # Ok(())
//! # }
//! ```
//!
//! Call [`append_frame`] to add a new frame to file. Chain any number of method calls
//! On the return value of [`append_frame`] to write those data chunks to the frame:
//!
//! ```
//! use hoomd_gsd::hoomd::HoomdGsdFile;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use tempfile::tempdir;
//! # let tmp_dir = tempdir().expect("temp dir should be created");
//! # let path = tmp_dir.path().join("test.gsd");
//! // let path = "file.gsd";
//! let mut hoomd_gsd_file = HoomdGsdFile::create(path)?;
//! hoomd_gsd_file
//!     .append_frame(1_000)?
//!     .configuration_box([100.0, 50.0, 80.0, 0.0, 0.0, 0.0])?
//!     .particles_position([[0.0, 1.0, 2.0].into(), [3.0, 6.0, 12.0].into()])?
//!     .end()?;
//! # Ok(())
//! # }
//! ```
//!
//! See the [`Frame`] documentation for a complete list of data chunks that you can write.
//!
//! The file is automatically synchronized and closed when the [`HoomdGsdFile`] is dropped.
//! Call [`open`] to open an existing file and append more frames:
//!
//! ```
//! use hoomd_gsd::hoomd::HoomdGsdFile;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use tempfile::tempdir;
//! # let tmp_dir = tempdir().expect("temp dir should be created");
//! # let path = tmp_dir.path().join("test.gsd");
//! // let path = "file.gsd";
//! # HoomdGsdFile::create(&path)?;
//! let mut hoomd_gsd_file = HoomdGsdFile::open(path)?;
//! hoomd_gsd_file
//!     .append_frame(2000)?
//!     .configuration_box([105.0, 48.0, 72.0, 0.0, 0.0, 0.0])?
//!     .particles_position([
//!         [2.0, 3.0, -1.0].into(),
//!         [18.0, 4.0, -6.0].into(),
//!     ]);
//! # Ok(())
//! # }
//! ```
//!
//! [`Frame`]: hoomd::Frame
//! [`append_frame`]: hoomd::HoomdGsdFile::append_frame
//! [`open`]: hoomd::HoomdGsdFile::open
//!
//! ## The file layer
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
//! gsd_file.write_scalars("configuration/step", [100_000_u64])?;
//! gsd_file.write_scalars(
//!     "configuration/box",
//!     [10.0_f32, 20.0, 15.0, 0.0, 0.0, 0.0],
//! )?;
//! gsd_file.write_arrays("particles/position", position.iter().copied())?;
//! gsd_file.end_frame()?;
//! # Ok(())
//! # }
//! ```
//! Each array in the file in stored in a specific type. `write_scalars` and
//! `write_arrays` automatically infer that type from the argument given.
//!
//! # Complete documentation
//!
//! `hoomd-gsd` is is a part of *hoomd-rs*. Read the [complete documentation]
//! for more information.
//!
//! [complete documentation]: https://hoomd-rs.readthedocs.io

pub mod file_layer;
pub mod hoomd;
