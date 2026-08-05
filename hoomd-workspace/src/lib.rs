// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! TODO: Docs
use std::{fs, io, path::{Path, PathBuf}};

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

/// Add a new state point to the on-disk workspace.
///
/// [`add`] creates the directory `workspace/{state_point.identifier()}` and serializes
/// `state_point` to `signac_statepoint.json`. Once a state point has been added
/// to the workspace, [`state_point`] may be called to read the state point and
/// the [`signac`] Python package will find it.
///
/// [`signac`]: https://signac.readthedocs.io
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

    fs::create_dir_all(&identifier_path)
        .map_err(|e| Error::Create(identifier_path.clone(), e))?;

    let state_point_json = entry::formatted(state_point)?;
    
    let state_point_path = identifier_path.join("signac_statepoint.json");
    fs::write(&state_point_path, state_point_json)
        .map_err(|e| Error::Write(state_point_path.clone(), e))?;

    Ok(())
}

/// Determine the state point of a given identifier (a directory in `workspace/`).
///
/// When the file `workspace/{identifier}/signac_statepoint.json` exists, [`state_point`]
/// reads it, deserializes the JSON and returns `Ok(Some(state_point))`. When the file
/// does not exist, [`state_point`] returns `Ok(None)`.
///
/// # Errors
///
/// * [`Error::Read`] when `workspace/{identifier}/signac_statepoint.json` exists and cannot be read.
/// * [`Error::Parse`] when `serde_json` cannot deserialize
///   `workspace/{identifier}/signac_statepoint.json` to `T`
#[inline]
pub fn state_point<T: for<'a> Deserialize<'a>>(identifier: &Path) -> Result<Option<T>, Error> {
    let state_point_path = [Path::new("workspace"), identifier, Path::new("signac_statepoint.json")].iter().collect();
    let state_point_bytes = match fs::read(&state_point_path) {
        Ok(state_point_bytes) => state_point_bytes,
        Err(error) => match error.kind() {
            io::ErrorKind::NotFound => return Ok(None),
            _ => return Err(Error::Read(state_point_path, error)),
        }
    };
       
    let state_point: T = serde_json::from_slice(&state_point_bytes)
        .map_err(|e| Error::Parse(state_point_path, e))?;
    Ok(Some(state_point))
}


