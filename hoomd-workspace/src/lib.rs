// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! `hoomd_workspace` allows you to create and access state point directories
//! in a workspace that is compatible with the [signac framework].
//! This crate offers a minimal API that covers only the most common use-cases
//! of the signac framework for simulation workflows. For a much more
//! comprehensive interface, use the [signac Python package].
//!
//! # The workspace
//!
//! The workspace is a directory on the filesystem named `workspace`.
//! `hoomd_workspace` *always* assumes that the `workspace` directory
//! is in the current working directory.
//!
//! # State points
//!
//! Each entry in the workspace has a unique **state point** associated with it.
//! In the signac framework, a state point is a dictionary. In `hoomd-workspace`,
//! a state point is any type that can be serialized to JSON. Derive serde's
//! `Deserialize` and `Serialize` traits to make any struct a valid state point:
//!
//! ```
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Deserialize, Serialize)]
//! struct MyStatePoint {
//!     temperature: f64,
//!     pressure: f64,
//! }
//! ```
//!
//! If you want to work with unstructured data, you can use [`serde_json::Map`]
//! directly.
//!
//! `hoomd-workspace` gives every state point an [`identifier`] and a [`path`].
//! The identifier is a hash of the state point's JSON representation. The path
//! is `workspace/{identifier}`.
//!
//! ```
//! use hoomd_workspace::Entry;
//! use serde::Serialize;
//! use std::path::Path;
//!
//! #[derive(Serialize)]
//! struct MyStatePoint {
//!     temperature: f64,
//!     pressure: f64,
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let my_state_point = MyStatePoint {
//!     temperature: 1.0,
//!     pressure: 2.0,
//! };
//!
//! let identifier = my_state_point.identifier()?;
//! let path = my_state_point.path()?;
//! # Ok(())
//! # }
//! ```
//!
//! # Create a state point directory
//!
//! Call [`add`] to create a state point directory.
//! ```
//! use serde::{Deserialize, Serialize};
//! use hoomd_workspace::Entry;
//!
//! #[derive(Deserialize, Serialize)]
//! struct MyStatePoint {
//!     temperature: f64,
//!     pressure: f64,
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use tempfile::tempdir;
//! # let tmp_dir = tempdir().expect("temp dir should be created");
//! # std::env::set_current_dir(&tmp_dir).expect("should be able to switch to temporary directory");
//! let my_state_point = MyStatePoint { temperature: 1.0, pressure: 2.0 };
//!
//! hoomd_workspace::add(&my_state_point)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Read the state point from a directory
//!
//! In simulation workflows, you often start with the *directory name* itself and not the
//! state point. Call [`state_point`] to read the state point that is associated with
//! the given directory (identifier).
//! ```
//! use serde::{Deserialize, Serialize};
//! use std::path::Path;
//! use anyhow::anyhow;
//!
//! #[derive(Deserialize, Serialize)]
//! struct MyStatePoint {
//!     temperature: f64,
//!     pressure: f64,
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use tempfile::tempdir;
//! # let tmp_dir = tempdir().expect("temp dir should be created");
//! # std::env::set_current_dir(&tmp_dir).expect("should be able to switch to temporary directory");
//! # let create_state_point = MyStatePoint { temperature: 1.0, pressure: 2.0 };
//! # hoomd_workspace::add(&create_state_point)?;
//!
//! let identifier = Path::new("bb97883a3a70ccfc0840d49a8c794342");
//! let my_state_point: MyStatePoint = hoomd_workspace::state_point(identifier)?
//!     .ok_or(anyhow!("state point not found"))?;
//! # Ok(())
//! # }
//! ```
//!
//! [signac framework]: https://signac.readthedocs.io
//! [signac Python package]: https://signac.readthedocs.io/projects/core/en/latest/
//! [`identifier`]: Entry::identifier
//! [`path`]: Entry::path

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

mod entry;

pub use entry::Entry;

/// Enumerate possible sources of error in fallible workspace methods.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum Error {
    /// Failed to serialize a state point to JSON.
    #[error("Failed to serialize state point to JSON")]
    Serialize(#[source] serde_json::Error),

    /// Failed to format state point JSON as a string.
    #[error("Failed to format the state point JSON as a string")]
    Format(#[source] serde_json::Error),

    /// Failed to read `signac_statepoint.json`
    #[error("Failed to read {0}")]
    Read(PathBuf, #[source] io::Error),

    /// Failed to parse `signac_statepoint.json`
    #[error("Failed to parse {0}")]
    Parse(PathBuf, #[source] serde_json::Error),

    /// Failed to create `workspace/{identifier}`
    #[error("Failed to create {0}")]
    Create(PathBuf, #[source] io::Error),

    /// Failed to write `signac_statepoint.json`
    #[error("Failed to create or write {0}")]
    Write(PathBuf, #[source] io::Error),
}

/// Add a new state point to the workspace.
///
/// `add` creates the directory `workspace/{state_point.identifier()}` and
/// serializes `state_point` to `signac_statepoint.json` in that directory. The
/// state point's identifier and JSON representation meet the specifications of
/// the [signac framework], making the workspace fully interoperable with the
/// [signac Python package].
///
/// [signac framework]: https://signac.readthedocs.io
/// [signac Python package]: https://signac.readthedocs.io/projects/core/en/latest/
///
/// # Example
///
/// ```
/// use serde::{Deserialize, Serialize};
/// use hoomd_workspace::Entry;
///
/// #[derive(Deserialize, Serialize)]
/// struct MyStatePoint {
///     temperature: f64,
///     pressure: f64,
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use tempfile::tempdir;
/// # let tmp_dir = tempdir().expect("temp dir should be created");
/// # std::env::set_current_dir(&tmp_dir).expect("should be able to switch to temporary directory");
/// let my_state_point = MyStatePoint { temperature: 1.0, pressure: 2.0 };
///
/// hoomd_workspace::add(&my_state_point)?;
///
/// assert!(my_state_point.path()?.exists());
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// * [`Error::Serialize`] when `serde_json` cannot serialize `state_point` to JSON.
/// * [`Error::Format`] when `serde_json` cannot format the JSON as a string.
/// * [`Error::Create`] when `workspace/{state_point.identifier()}` cannot be created.
/// * [`Error::Write`] when `serde_json` cannot serialize `state_point` to JSON or there is
///   an IO error writing `workspace/{state_point.identifier()}/signac_statepoint.json`.
#[inline]
pub fn add<T: Entry + Serialize>(state_point: &T) -> Result<(), Error> {
    let identifier_path = state_point.path()?;

    fs::create_dir_all(&identifier_path).map_err(|e| Error::Create(identifier_path.clone(), e))?;

    let state_point_json = entry::formatted(state_point)?;

    let state_point_path = identifier_path.join("signac_statepoint.json");
    fs::write(&state_point_path, state_point_json)
        .map_err(|e| Error::Write(state_point_path.clone(), e))?;

    Ok(())
}

/// Determine the state point of a given identifier pointing to a directory in `workspace/`.
///
/// When the file `workspace/{identifier}/signac_statepoint.json` exists, [`state_point`]
/// reads it, deserializes the JSON and returns `Ok(Some(state_point))`. When the file
/// does not exist, [`state_point`] returns `Ok(None)`.
///
/// # Examples
///
/// Read an existing state point:
/// ```
/// use serde::{Deserialize, Serialize};
/// use std::path::Path;
/// use anyhow::anyhow;
///
/// #[derive(Deserialize, Serialize)]
/// struct MyStatePoint {
///     temperature: f64,
///     pressure: f64,
/// }
/// #
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use tempfile::tempdir;
/// # let tmp_dir = tempdir().expect("temp dir should be created");
/// # std::env::set_current_dir(&tmp_dir).expect("should be able to switch to temporary directory");
/// # let create_state_point = MyStatePoint { temperature: 1.0, pressure: 2.0 };
/// # hoomd_workspace::add(&create_state_point)?;
///
/// // This hash matches the state point created in the example for `add`
/// let identifier = Path::new("bb97883a3a70ccfc0840d49a8c794342");
/// let my_state_point: MyStatePoint = hoomd_workspace::state_point(identifier)?
///     .ok_or(anyhow!("state point not found"))?;
///
/// assert_eq!(my_state_point.pressure, 2.0);
/// assert_eq!(my_state_point.temperature, 1.0);
/// # Ok(())
/// # }
/// ```
///
/// Attempt to access a state point that does not exist:
/// ```
/// use serde::{Deserialize, Serialize};
/// use std::path::Path;
/// use anyhow::anyhow;
///
/// #[derive(Deserialize, Serialize)]
/// struct MyStatePoint {
///     temperature: f64,
///     pressure: f64,
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use tempfile::tempdir;
/// # let tmp_dir = tempdir().expect("temp dir should be created");
/// # std::env::set_current_dir(&tmp_dir).expect("should be able to switch to temporary directory");
///
/// let identifier = Path::new("not a state point");
/// let maybe_state_point: Option<MyStatePoint> = hoomd_workspace::state_point(identifier)?;
///
/// assert!(maybe_state_point.is_none());
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// * [`Error::Read`] when `workspace/{identifier}/signac_statepoint.json` exists and cannot be read.
/// * [`Error::Parse`] when `serde_json` cannot deserialize
///   `workspace/{identifier}/signac_statepoint.json` to `T`
#[inline]
pub fn state_point<T: for<'a> Deserialize<'a>>(identifier: &Path) -> Result<Option<T>, Error> {
    let state_point_path = [
        Path::new("workspace"),
        identifier,
        Path::new("signac_statepoint.json"),
    ]
    .iter()
    .collect();
    let state_point_bytes = match fs::read(&state_point_path) {
        Ok(state_point_bytes) => state_point_bytes,
        Err(error) => match error.kind() {
            io::ErrorKind::NotFound => return Ok(None),
            _ => return Err(Error::Read(state_point_path, error)),
        },
    };

    let state_point: T = serde_json::from_slice(&state_point_bytes)
        .map_err(|e| Error::Parse(state_point_path, e))?;
    Ok(Some(state_point))
}
