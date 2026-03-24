// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Write HOOMD schema GSD files.
//!
//! Use [`HoomdGsdFile`] to write GSD files with the HOOMD schema.

use std::{
    array,
    num::TryFromIntError,
    path::Path,
    time::{Duration, Instant},
};
use thiserror::Error;

use hoomd_vector::{Cartesian, Versor};

use crate::file_layer::{EncodeError, GsdFile, Mode, OpenError, SyncError, Type, WriteError};

/// Longest type name size (including the null terminator).
const MAX_NAME_LENGTH: usize = 64;

/// Default delay between automatic calls to `sync_all`.
const DEFAULT_AUTO_SYNC_DELAY: Duration = Duration::new(10, 0);

/// Create and write to GSD files with the HOOMD schema.
///
/// Files written by [`HoomdGsdFile`] can be read by the [Ovito], [HOOMD-blue],
/// the [GSD Python package], and other applications.
///
/// [GSD Python package]: https://gsd.readthedocs.io
/// [HOOMD-blue]: https://hoomd-blue.readthedocs.io
/// [Ovito]: https://www.ovito.org
///
/// # Buffering
///
/// GSD aggressively buffers data in memory before synchronizing it to the
/// file. Your appended frames *may* not be present in the file until it is
/// closed or you call [`sync_all`]. [`append_frame`] will automatically call
/// [`sync_all`] when the last sync was more than 10 seconds prior (by default).
/// Set `auto_sync_delay` to change the timeout.
///
/// # Example
///
/// ```
/// use hoomd_gsd::hoomd::HoomdGsdFile;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use tempfile::tempdir;
/// # let tmp_dir = tempdir().expect("temp dir should be created");
/// # let path = tmp_dir.path().join("test.gsd");
/// // let path = "file.gsd";
/// let mut hoomd_gsd_file = HoomdGsdFile::create(path)?;
/// hoomd_gsd_file
///     .append_frame(2_000)?
///     .configuration_box([105.0, 48.0, 72.0, 0.0, 0.0, 0.0])?
///     .particles_position([
///         [2.0, 3.0, -1.0].into(),
///         [18.0, 4.0, -6.0].into(),
///     ])?
///     .end()?;
/// # Ok(())
/// # }
/// ```
///
/// [`append_frame`]: HoomdGsdFile::append_frame
/// [`sync_all`]: HoomdGsdFile::sync_all
pub struct HoomdGsdFile {
    /// The wrapped GSD file.
    gsd_file: GsdFile,

    /// Time to auto synchronize.
    auto_sync_delay: Duration,

    /// Time of the last auto synchronization.
    last_auto_sync: Instant,
}

/// The number of dimensions in the GSD file.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dimensions {
    /// All particle z coordinates and the box z should be zero.
    Two,
    /// Particle positions and the box shape can have non-zero z values.
    Three,
}

/// In-progress frame in a HOOMD GSD file.
///
/// Call [`HoomdGsdFile::append_frame`] to create a new frame in the file. The
/// [`Frame`] it returns has chainable methods you can call to add data
/// chunks to the frame in the file. The frame is complete when the
/// [`Frame`] is dropped.
///
/// `append_frame` always writes `configuration/step` with the given step.
/// Each of the following methods writes the data chunk of the same name
/// (where '/' has been replaced with '_'):
///
/// # Configuration
///
/// * [`configuration_dimensions`](Self::configuration_dimensions)
/// * [`configuration_box`](Self::configuration_box)
///
/// # Particles
///
/// * [`particles_diameter`](Self::particles_diameter)
/// * [`particles_position`](Self::particles_position)
/// * [`particles_orientation`](Self::particles_orientation)
/// * [`particles_type_id`](Self::particles_type_id)
/// * [`particles_types`](Self::particles_types)
///
/// The first call to any `particles_*` method (except `particles_types`) implicitly
/// writes the chunk `particles/N`. Any subsequent calls to other `particles_*`
/// methods will return an error if `N` does not match.
///
/// # Log
///
/// All `log_*` methods write the chunk "log/{name}".
///
/// * [`log_scalar`](Self::log_scalar)
/// * [`log_scalars`](Self::log_scalars)
/// * [`log_arrays`](Self::log_arrays)
pub struct Frame<'a> {
    /// The GSD file this frame is part of.
    hoomd_gsd_file: &'a mut HoomdGsdFile,

    /// Stores the number of particles for error checking `particles_*` calls.
    particles_n: Option<u32>,

    /// Flag when the frame has ended.
    ended: bool,
}

/// Errors that can occur while appending a frame to a HOOMD GSD file.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum AppendError {
    /// This data chunk does not match the dimensions of those previously written.
    #[error("The length of data chunk {0} does not match previously written {1} chunks")]
    InconsistentLength(String, String),

    /// Cannot write to the file.
    #[error("cannot write to the file")]
    Write(#[from] WriteError),

    /// Cannot encode data to write.
    #[error("cannot encode data to write")]
    Encode(#[from] EncodeError),

    /// Cannot synchronize data to the file.
    #[error("cannot synchronize data to the file")]
    Sync(#[from] SyncError),

    /// Too many entries to write.
    #[error("cannot write {0} entries to data chunk {1}")]
    ChunkTooLarge(usize, String, #[source] TryFromIntError),
}

impl HoomdGsdFile {
    /// Overwrite an existing HOOMD GSD file (or create a new file).
    ///
    /// Creates a GSD file at the given path, overwriting any file that may already
    /// exist. When successful, return a [`HoomdGsdFile`] opened in write mode.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_gsd::hoomd::HoomdGsdFile;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("test.gsd");
    /// // let path = "file.gsd";
    /// let hoomd_gsd_file = HoomdGsdFile::create(path)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`OpenError`] when any of the following occur:
    /// * The file cannot be created.
    /// * The file is corrupt, unreadable, or there is an I/O error (see
    ///   [`DecodeError`]).
    ///
    /// [`DecodeError`]: crate::file_layer::DecodeError
    #[inline]
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self, OpenError> {
        let version = env!("CARGO_PKG_VERSION");
        let application = format!("hoomd-rs {version}");
        let gsd_file = GsdFile::create(path, &application, "hoomd", (1, 4))?;

        let auto_sync_delay = DEFAULT_AUTO_SYNC_DELAY;
        let last_auto_sync = Instant::now();
        Ok(Self {
            gsd_file,
            auto_sync_delay,
            last_auto_sync,
        })
    }

    /// Create a new HOOMD GSD file.
    ///
    /// Creates a new GSD file at the given path, returning an error when the
    /// path already exists. When successful, return a [`HoomdGsdFile`] opened in
    /// write mode.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_gsd::hoomd::HoomdGsdFile;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("test.gsd");
    /// // let path = "file.gsd";
    /// let hoomd_gsd_file = HoomdGsdFile::create_new(path)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`OpenError`] when any of the following occur:
    /// * The file cannot be created.
    /// * The file already exists.
    /// * The file is corrupt, unreadable, or there is an I/O error (see
    ///   [`DecodeError`]).
    ///
    /// [`DecodeError`]: crate::file_layer::DecodeError
    #[inline]
    pub fn create_new<P: AsRef<Path>>(path: P) -> Result<Self, OpenError> {
        let version = env!("CARGO_PKG_VERSION");
        let application = format!("hoomd-rs {version}");
        let gsd_file = GsdFile::create_new(path, &application, "hoomd", (1, 4))?;

        let auto_sync_delay = DEFAULT_AUTO_SYNC_DELAY;
        let last_auto_sync = Instant::now();
        Ok(Self {
            gsd_file,
            auto_sync_delay,
            last_auto_sync,
        })
    }

    /// Open an existing HOOMD GSD file in write mode.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_gsd::hoomd::HoomdGsdFile;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("test.gsd");
    /// // let path = "file.gsd";
    /// # HoomdGsdFile::create_new(&path)?;
    /// let hoomd_gsd_file = HoomdGsdFile::open(path)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`OpenError`] when any of the following occur:
    /// * The file does not exist.
    /// * The file is corrupt, unreadable, or there is an I/O error (see
    ///   [`DecodeError`]).
    ///
    /// [`DecodeError`]: crate::file_layer::DecodeError
    #[inline]
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, OpenError> {
        let gsd_file = GsdFile::open(path, Mode::Write)?;

        let auto_sync_delay = DEFAULT_AUTO_SYNC_DELAY;
        let last_auto_sync = Instant::now();
        Ok(Self {
            gsd_file,
            auto_sync_delay,
            last_auto_sync,
        })
    }

    /// Write buffered data to the filesystem.
    ///
    /// `sync_all` ensures that the data and indices for all complete frames is
    /// written to the filesystem.
    ///
    /// In most cases, callers should not invoke `sync_all` manually. It will
    /// be called automatically when a [`HoomdGsdFile`] is dropped and after any
    /// call to `append_frame` at least 10 seconds (by default) after a previous
    /// call to `sync_all`.
    ///
    /// Call `sync_all` only want to ensure that all data up to a specific frame
    /// are present in the file.
    ///
    /// # Errors
    ///
    /// Returns a [`WriteError`] when any of the following occur:
    /// * The file is not opened in a write mode.
    /// * An I/O error writing to the file.
    pub fn sync_all(&mut self) -> Result<(), SyncError> {
        self.gsd_file.sync_all()?;
        self.last_auto_sync = Instant::now();
        Ok(())
    }

    /// Append a new frame to the file.
    ///
    /// Use chained method calls to concisely write all the data chunks needed
    /// for a single frame (see the example):
    ///
    /// # Ownership
    ///
    /// The returned [`Frame`] holds a mutable reference to this
    /// [`HoomdGsdFile`], so the compiler forces you to complete writing the
    /// frame and drop the [`Frame`] (either implicitly or explicitly) before
    /// you can take any other actions that would mutate the [`HoomdGsdFile`].
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_gsd::hoomd::HoomdGsdFile;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("test.gsd");
    /// // let path = "file.gsd";
    /// let mut hoomd_gsd_file = HoomdGsdFile::create(path)?;
    /// hoomd_gsd_file
    ///     .append_frame(1_000)?
    ///     .configuration_box([100.0, 50.0, 80.0, 0.0, 0.0, 0.0])?
    ///     .particles_position([[0.0, 1.0, 2.0].into(), [3.0, 6.0, 12.0].into()])?
    ///     .end()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`WriteError`] when any of the following occur:
    /// * The file is not opened in a write mode.
    /// * An I/O error writing to the file.
    #[inline]
    pub fn append_frame(&mut self, step: u64) -> Result<Frame<'_>, WriteError> {
        self.gsd_file.write_scalars("configuration/step", [step])?;
        Ok(Frame {
            hoomd_gsd_file: self,
            particles_n: None,
            ended: false,
        })
    }

    /// Get the auto sync delay.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_gsd::hoomd::HoomdGsdFile;
    /// use std::time::Duration;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("test.gsd");
    /// // let path = "file.gsd";
    /// let mut hoomd_gsd_file = HoomdGsdFile::create(path)?;
    /// let auto_sync_delay = hoomd_gsd_file.auto_sync_delay();
    ///
    /// assert_eq!(*auto_sync_delay, Duration::new(10, 0));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn auto_sync_delay(&self) -> &Duration {
        &self.auto_sync_delay
    }

    /// Get a mutable reference to the auto sync delay.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_gsd::hoomd::HoomdGsdFile;
    /// use std::time::Duration;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("test.gsd");
    /// // let path = "file.gsd";
    /// let mut hoomd_gsd_file = HoomdGsdFile::create(path)?;
    /// *hoomd_gsd_file.auto_sync_delay_mut() = Duration::new(60, 0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn auto_sync_delay_mut(&mut self) -> &mut Duration {
        &mut self.auto_sync_delay
    }
}

impl Frame<'_> {
    /// Write [`configuration/dimensions`] to the current frame in the GSD file.
    ///
    /// [`configuration/dimensions`]: https://gsd.readthedocs.io/en/v4.2.0/schema-hoomd.html#chunk-configuration-dimensions
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_gsd::hoomd::{Dimensions, HoomdGsdFile};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("test.gsd");
    /// // let path = "file.gsd";
    /// let mut hoomd_gsd_file = HoomdGsdFile::create(path)?;
    /// hoomd_gsd_file
    ///     .append_frame(1_000)?
    ///     .configuration_dimensions(Dimensions::Two)?
    ///     .end()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`AppendError`] when any of the following occur:
    /// * The file is not opened in a write mode.
    /// * An I/O error writing to the file.
    pub fn configuration_dimensions(self, dimensions: Dimensions) -> Result<Self, AppendError> {
        let chunk_name = "configuration/dimensions";

        let dimensions: u8 = match dimensions {
            Dimensions::Two => 2,
            Dimensions::Three => 3,
        };

        self.hoomd_gsd_file
            .gsd_file
            .write_scalars(chunk_name, [dimensions])?;

        Ok(self)
    }

    /// Write [`configuration/box`] to the current frame in the GSD file.
    ///
    /// The input `f64` values are converted to `f32`.
    ///
    /// [`configuration/box`]: https://gsd.readthedocs.io/en/v4.2.0/schema-hoomd.html#chunk-configuration-box
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_gsd::hoomd::HoomdGsdFile;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("test.gsd");
    /// // let path = "file.gsd";
    /// let mut hoomd_gsd_file = HoomdGsdFile::create(path)?;
    /// hoomd_gsd_file
    ///     .append_frame(1_000)?
    ///     .configuration_box([10.0, 20.0, 30.0, 0.0, 0.0, 0.0])?
    ///     .end()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`AppendError`] when any of the following occur:
    /// * The file is not opened in a write mode.
    /// * An I/O error writing to the file.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "truncating to match the GSD specification"
    )]
    pub fn configuration_box(self, values: [f64; 6]) -> Result<Self, AppendError> {
        let chunk_name = "configuration/box";

        let values: [f32; 6] = array::from_fn(|i| values[i] as f32);
        self.hoomd_gsd_file
            .gsd_file
            .write_scalars(chunk_name, values)?;

        Ok(self)
    }

    /// Write the length of the data to `particles/N` or return an error when
    /// `particles/N` has already been written and this `N` does not match.
    fn validate_particles_chunk<T>(
        &mut self,
        data: &[T],
        chunk_name: &str,
    ) -> Result<(), AppendError> {
        if let Some(n) = self.particles_n {
            if data.len() != n as usize {
                return Err(AppendError::InconsistentLength(
                    chunk_name.to_string(),
                    "particles".to_string(),
                ));
            }
        } else {
            let n = data
                .len()
                .try_into()
                .map_err(|e| AppendError::ChunkTooLarge(data.len(), chunk_name.to_string(), e))?;
            self.hoomd_gsd_file
                .gsd_file
                .write_scalars("particles/N", [n])?;
            self.particles_n = Some(n);
        }

        Ok(())
    }

    /// Write [`particles/position`] to the current frame in the GSD file.
    ///
    /// The input `Cartesian<3>` argument is converted to `[f32; 3]`.
    ///
    /// [`particles/position`]: https://gsd.readthedocs.io/en/v4.2.0/schema-hoomd.html#chunk-particles-position
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_gsd::hoomd::HoomdGsdFile;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("test.gsd");
    /// // let path = "file.gsd";
    /// let mut hoomd_gsd_file = HoomdGsdFile::create(path)?;
    /// hoomd_gsd_file
    ///     .append_frame(1_000)?
    ///     .particles_position([
    ///         [2.0, 3.0, -1.0].into(),
    ///         [18.0, 4.0, -6.0].into(),
    ///     ])?
    ///     .end()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`AppendError`] when any of the following occur:
    /// * The file is not opened in a write mode.
    /// * An I/O error writing to the file.
    /// * *N* does not match a previous `particles_*` data chunk in this frame.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "truncating to match the GSD specification"
    )]
    pub fn particles_position<I>(mut self, position: I) -> Result<Self, AppendError>
    where
        I: IntoIterator<Item = Cartesian<3>>,
    {
        let chunk_name = "particles/position";
        let data: Vec<_> = position
            .into_iter()
            .map(|v| -> [f32; 3] { array::from_fn(|i| v[i] as f32) })
            .collect();

        self.validate_particles_chunk(&data, chunk_name)?;

        self.hoomd_gsd_file
            .gsd_file
            .write_arrays(chunk_name, data)?;

        Ok(self)
    }

    /// Write [`particles/orientation`] to the current frame in the GSD file.
    ///
    /// [`particles/orientation`]: https://gsd.readthedocs.io/en/v4.2.0/schema-hoomd.html#chunk-particles-orientation
    ///
    /// The input `Versor` argument is converted to `[f32; 4]`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_gsd::hoomd::HoomdGsdFile;
    /// use hoomd_vector::Versor;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("test.gsd");
    /// // let path = "file.gsd";
    /// let mut hoomd_gsd_file = HoomdGsdFile::create(path)?;
    /// hoomd_gsd_file
    ///     .append_frame(1_000)?
    ///     .particles_orientation([
    ///         Versor::from_axis_angle([0.0, 0.0, 1.0].try_into()?, 1.2),
    ///         Versor::from_axis_angle([0.0, 1.0, 0.0].try_into()?, -0.3),
    ///     ])?
    ///     .end()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`AppendError`] when any of the following occur:
    /// * The file is not opened in a write mode.
    /// * An I/O error writing to the file.
    /// * *N* does not match a previous `particles_*` data chunk in this frame.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "truncating to match the GSD specification"
    )]
    pub fn particles_orientation<I>(mut self, orientation: I) -> Result<Self, AppendError>
    where
        I: IntoIterator<Item = Versor>,
    {
        let chunk_name = "particles/orientation";
        let data: Vec<_> = orientation
            .into_iter()
            .map(|v| {
                [
                    v.get().scalar as f32,
                    v.get().vector[0] as f32,
                    v.get().vector[1] as f32,
                    v.get().vector[2] as f32,
                ]
            })
            .collect();

        self.validate_particles_chunk(&data, chunk_name)?;

        self.hoomd_gsd_file
            .gsd_file
            .write_arrays(chunk_name, data)?;

        Ok(self)
    }

    /// Write [`particles/typeid`] to the current frame in the GSD file.
    ///
    /// [`particles/typeid`]: https://gsd.readthedocs.io/en/v4.2.0/schema-hoomd.html#chunk-particles-typeid
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_gsd::hoomd::HoomdGsdFile;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("test.gsd");
    /// // let path = "file.gsd";
    /// let mut hoomd_gsd_file = HoomdGsdFile::create(path)?;
    /// hoomd_gsd_file
    ///     .append_frame(1_000)?
    ///     .particles_type_id([0, 0, 0, 1, 1, 1])?
    ///     .end()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`AppendError`] when any of the following occur:
    /// * The file is not opened in a write mode.
    /// * An I/O error writing to the file.
    /// * *N* does not match a previous `particles_*` data chunk in this frame.
    pub fn particles_type_id<I>(mut self, type_id: I) -> Result<Self, AppendError>
    where
        I: IntoIterator<Item = u32>,
    {
        let chunk_name = "particles/typeid";
        let data: Vec<_> = type_id.into_iter().collect();

        self.validate_particles_chunk(&data, chunk_name)?;

        self.hoomd_gsd_file
            .gsd_file
            .write_scalars(chunk_name, data)?;

        Ok(self)
    }

    /// Write [`particles/types`] to the current frame in the GSD file.
    ///
    /// [`particles/types`]: https://gsd.readthedocs.io/en/v4.2.0/schema-hoomd.html#chunk-particles-types
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_gsd::hoomd::HoomdGsdFile;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("test.gsd");
    /// // let path = "file.gsd";
    /// let mut hoomd_gsd_file = HoomdGsdFile::create(path)?;
    /// hoomd_gsd_file
    ///     .append_frame(1_000)?
    ///     .particles_types(["A", "B", "linker"])?
    ///     .end()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`AppendError`] when any of the following occur:
    /// * The file is not opened in a write mode.
    /// * An I/O error writing to the file.
    ///
    /// # Panics
    ///
    /// Panics when any type name string is longer than 63 characters. This is a limitation
    /// in the *hoomd-rs* implementation, not the GSD file format. The limitation may be
    /// removed in a future release.
    pub fn particles_types<'a, I>(self, types: I) -> Result<Self, AppendError>
    where
        I: IntoIterator<Item = &'a str>,
    {
        let chunk_name = "particles/types";
        let types: Vec<_> = types.into_iter().map(str::to_string).collect();
        let max_len = types.iter().map(String::len).fold(0, Ord::max);

        assert!(max_len < MAX_NAME_LENGTH, "type name length too long");

        self.hoomd_gsd_file.gsd_file.write_arrays(
            chunk_name,
            types.into_iter().map(|s| -> [u8; MAX_NAME_LENGTH] {
                array::from_fn(|i| if i < s.len() { s.as_bytes()[i] } else { 0 })
            }),
        )?;

        Ok(self)
    }

    /// Write [`particles/diameter`] to the current frame in the GSD file.
    ///
    /// [`particles/diameter`]: https://gsd.readthedocs.io/en/v4.2.0/schema-hoomd.html#chunk-particles-diameter
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_gsd::hoomd::HoomdGsdFile;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("test.gsd");
    /// // let path = "file.gsd";
    /// let mut hoomd_gsd_file = HoomdGsdFile::create(path)?;
    /// hoomd_gsd_file
    ///     .append_frame(1_000)?
    ///     .particles_diameter([1.0, 0.5, 0.25, 2.0])?
    ///     .end()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`AppendError`] when any of the following occur:
    /// * The file is not opened in a write mode.
    /// * An I/O error writing to the file.
    /// * *N* does not match a previous `particles_*` data chunk in this frame.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "truncating to match the GSD specification"
    )]
    pub fn particles_diameter<I>(mut self, diameter: I) -> Result<Self, AppendError>
    where
        I: IntoIterator<Item = f64>,
    {
        let chunk_name = "particles/diameter";
        let data: Vec<_> = diameter.into_iter().map(|x| x as f32).collect();

        self.validate_particles_chunk(&data, chunk_name)?;

        self.hoomd_gsd_file
            .gsd_file
            .write_scalars(chunk_name, data)?;

        Ok(self)
    }

    /// Write [`log/{name}`] to the current frame in the GSD file as a 1x1 array of type
    /// `T`.
    ///
    /// [`log/{name}`]: https://gsd.readthedocs.io/en/v4.2.0/schema-hoomd.html#logged-data
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_gsd::hoomd::HoomdGsdFile;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("test.gsd");
    /// // let path = "file.gsd";
    /// let mut hoomd_gsd_file = HoomdGsdFile::create(path)?;
    /// hoomd_gsd_file
    ///     .append_frame(1_000)?
    ///     .log_scalar("height", 10.0_f64)?
    ///     .end()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`AppendError`] when any of the following occur:
    /// * The file is not opened in a write mode.
    /// * An I/O error writing to the file.
    pub fn log_scalar<T>(self, name: &str, scalar: T) -> Result<Self, AppendError>
    where
        T: Type,
    {
        let chunk_name = ["log", name].join("/");

        self.hoomd_gsd_file
            .gsd_file
            .write_scalars(&chunk_name, [scalar])?;

        Ok(self)
    }

    /// Write [`log/{name}`] to the current frame in the GSD file as a *N*x1 array of type
    /// `T` where *N* is the number of items produced by the `scalars` iterator.
    ///
    /// [`log/{name}`]: https://gsd.readthedocs.io/en/v4.2.0/schema-hoomd.html#logged-data
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_gsd::hoomd::HoomdGsdFile;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("test.gsd");
    /// // let path = "file.gsd";
    /// let mut hoomd_gsd_file = HoomdGsdFile::create(path)?;
    /// hoomd_gsd_file
    ///     .append_frame(1_000)?
    ///     .log_scalars("energy", [1.0_f64, 2.0, 3.0])?
    ///     .end()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`AppendError`] when any of the following occur:
    /// * The file is not opened in a write mode.
    /// * An I/O error writing to the file.
    pub fn log_scalars<T, I>(self, name: &str, scalars: I) -> Result<Self, AppendError>
    where
        T: Type,
        I: IntoIterator<Item = T>,
    {
        let chunk_name = ["log", name].join("/");

        self.hoomd_gsd_file
            .gsd_file
            .write_scalars(&chunk_name, scalars)?;

        Ok(self)
    }

    /// Write [`log/{name}`] to the current frame in the GSD file as a *N*x*M* array of type
    /// `T` where *N* is the number of items produced by the `arrays` iterator.
    ///
    /// [`log/{name}`]: https://gsd.readthedocs.io/en/v4.2.0/schema-hoomd.html#logged-data
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_gsd::hoomd::HoomdGsdFile;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("test.gsd");
    /// // let path = "file.gsd";
    /// let mut hoomd_gsd_file = HoomdGsdFile::create(path)?;
    /// hoomd_gsd_file
    ///     .append_frame(1_000)?
    ///     .log_arrays("points", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])?
    ///     .end()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`AppendError`] when any of the following occur:
    /// * The file is not opened in a write mode.
    /// * An I/O error writing to the file.
    pub fn log_arrays<T, I, const M: usize>(
        self,
        name: &str,
        arrays: I,
    ) -> Result<Self, AppendError>
    where
        T: Type,
        I: IntoIterator<Item = [T; M]>,
    {
        let chunk_name = ["log", name].join("/");

        self.hoomd_gsd_file
            .gsd_file
            .write_arrays(&chunk_name, arrays)?;

        Ok(self)
    }

    /// End the frame.
    ///
    /// Once the frame is complete, no more data chunks may be added to it. The next call
    /// to [`HoomdGsdFile::append_frame`] will add a new frame to the file.
    ///
    /// Calling `end` is *optional*. The frame will automatically end when [`Frame`] is
    /// dropped. `Drop` ignores any errors. Call `end` explicitly to check for
    /// errors.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when there is a problem writing to the file.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_gsd::hoomd::HoomdGsdFile;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("test.gsd");
    /// // let path = "file.gsd";
    /// let mut hoomd_gsd_file = HoomdGsdFile::create(path)?;
    /// hoomd_gsd_file.append_frame(1_000)?.end()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn end(mut self) -> Result<(), AppendError> {
        self.hoomd_gsd_file.gsd_file.end_frame()?;
        if self.hoomd_gsd_file.last_auto_sync.elapsed() >= self.hoomd_gsd_file.auto_sync_delay {
            self.hoomd_gsd_file.sync_all()?;
        }

        self.ended = true;

        Ok(())
    }
}

/// End the frame.
///
/// Once the frame is complete, no more data chunks may be added to it. The next call
/// to [`HoomdGsdFile::append_frame`] will add a new frame to the file.
///
/// `drop` checks the amount of time since the last call to [`HoomdGsdFile::sync_all`].
/// If it has been more than the auto sync delay, `drop` calls `sync_all`.
/// `drop` ignores all errors. Call [`Frame::end`] to end the frame and
/// check on potential I/O errors.
///
/// # Example
///
/// ```
/// use hoomd_gsd::hoomd::HoomdGsdFile;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use tempfile::tempdir;
/// # let tmp_dir = tempdir().expect("temp dir should be created");
/// # let path = tmp_dir.path().join("test.gsd");
/// // let path = "file.gsd";
/// let mut hoomd_gsd_file = HoomdGsdFile::create(path)?;
/// hoomd_gsd_file.append_frame(1_000)?;
/// // ... some I/O errors might be ignored during the implicit drop.
/// # Ok(())
/// # }
/// ```
impl Drop for Frame<'_> {
    fn drop(&mut self) {
        if !self.ended {
            let _ = self.hoomd_gsd_file.gsd_file.end_frame();
            if self.hoomd_gsd_file.last_auto_sync.elapsed() >= self.hoomd_gsd_file.auto_sync_delay {
                let _ = self.hoomd_gsd_file.sync_all();
            }
        }
    }
}

#[cfg(test)]
mod test {
    use assert2::{assert, check};
    use std::thread;
    use tempfile::tempdir;

    use super::*;
    use hoomd_vector::Versor;

    #[test]
    fn create() -> anyhow::Result<()> {
        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");

        let hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        check!(*hoomd_gsd_file.gsd_file.mode() == Mode::Write);
        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path.clone(), Mode::Read)?;
        check!(gsd_file.application().starts_with("hoomd-rs"));
        check!(gsd_file.schema() == "hoomd");
        check!(gsd_file.schema_version() == (1, 4));
        check!(gsd_file.n_frames() == 0);
        check!(gsd_file.name_id().is_empty());

        Ok(())
    }

    #[test]
    fn create_new() -> anyhow::Result<()> {
        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let hoomd_gsd_file = HoomdGsdFile::create_new(path.clone())?;
        check!(*hoomd_gsd_file.gsd_file.mode() == Mode::Write);
        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path.clone(), Mode::Read)?;
        check!(gsd_file.application().starts_with("hoomd-rs"));
        check!(gsd_file.schema() == "hoomd");
        check!(gsd_file.schema_version() == (1, 4));
        check!(gsd_file.n_frames() == 0);
        check!(gsd_file.name_id().is_empty());

        check!(matches!(
            HoomdGsdFile::create_new(path.clone()),
            Err(OpenError::IO(_, _))
        ));

        Ok(())
    }

    #[test]
    fn open() -> anyhow::Result<()> {
        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");

        let hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        check!(*hoomd_gsd_file.gsd_file.mode() == Mode::Write);
        drop(hoomd_gsd_file);

        let hoomd_gsd_file = HoomdGsdFile::open(path.clone())?;
        check!(*hoomd_gsd_file.gsd_file.mode() == Mode::Write);
        // TODO: test that data chunks can be appended to an open file
        drop(hoomd_gsd_file);

        Ok(())
    }

    #[test]
    fn auto_sync_delay() -> anyhow::Result<()> {
        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");

        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        check!(*hoomd_gsd_file.auto_sync_delay() == DEFAULT_AUTO_SYNC_DELAY);

        *hoomd_gsd_file.auto_sync_delay_mut() = Duration::new(20, 0);
        check!(*hoomd_gsd_file.auto_sync_delay() == Duration::new(20, 0));

        let previous_auto_sync = hoomd_gsd_file.last_auto_sync;

        thread::sleep(Duration::from_millis(40));
        hoomd_gsd_file.sync_all()?;

        check!(previous_auto_sync != hoomd_gsd_file.last_auto_sync);

        Ok(())
    }

    #[test]
    fn configuration_dimensions() -> anyhow::Result<()> {
        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        hoomd_gsd_file.append_frame(0)?;
        hoomd_gsd_file
            .append_frame(1)?
            .configuration_dimensions(Dimensions::Two)?;
        hoomd_gsd_file
            .append_frame(2)?
            .configuration_dimensions(Dimensions::Two)?;
        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path.clone(), Mode::Read)?;
        check!(gsd_file.find_chunk(0, "configuration/dimensions") == None);

        let data = gsd_file.iter_scalars::<u8>(1, "configuration/dimensions")?;
        itertools::assert_equal(data, [2]);

        let data = gsd_file.iter_scalars::<u8>(2, "configuration/dimensions")?;
        itertools::assert_equal(data, [2]);

        Ok(())
    }

    #[test]
    fn configuration_box() -> anyhow::Result<()> {
        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        hoomd_gsd_file.append_frame(0)?;
        hoomd_gsd_file
            .append_frame(1)?
            .configuration_box([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path.clone(), Mode::Read)?;
        check!(gsd_file.find_chunk(0, "configuration/box") == None);

        let data = gsd_file.iter_scalars::<f32>(1, "configuration/box")?;
        itertools::assert_equal(data, [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        Ok(())
    }

    #[test]
    fn particles_position() -> anyhow::Result<()> {
        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        hoomd_gsd_file.append_frame(0)?;
        hoomd_gsd_file.append_frame(1)?.particles_position([
            [1.0, 2.0, 4.0].into(),
            [3.0, 4.0, 8.0].into(),
            [5.0, 6.0, 16.0].into(),
        ])?;
        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path.clone(), Mode::Read)?;
        check!(gsd_file.find_chunk(0, "particles/position") == None);
        check!(gsd_file.find_chunk(0, "particles/N") == None);

        let data = gsd_file.iter_arrays::<f32, 3>(1, "particles/position")?;
        itertools::assert_equal(data, [[1.0, 2.0, 4.0], [3.0, 4.0, 8.0], [5.0, 6.0, 16.0]]);
        let data = gsd_file.iter_scalars::<u32>(1, "particles/N")?;
        itertools::assert_equal(data, [3_u32]);

        Ok(())
    }

    #[test]
    fn particles_orientation() -> anyhow::Result<()> {
        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        hoomd_gsd_file.append_frame(0)?;
        hoomd_gsd_file
            .append_frame(1)?
            .particles_orientation([Versor::default(); 3])?;
        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path.clone(), Mode::Read)?;
        check!(gsd_file.find_chunk(0, "particles/orientation") == None);
        check!(gsd_file.find_chunk(0, "particles/N") == None);

        let data = gsd_file.iter_arrays::<f32, 4>(1, "particles/orientation")?;
        itertools::assert_equal(
            data,
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
        );
        let data = gsd_file.iter_scalars::<u32>(1, "particles/N")?;
        itertools::assert_equal(data, [3_u32]);

        Ok(())
    }

    #[test]
    fn particles_type_id() -> anyhow::Result<()> {
        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        hoomd_gsd_file.append_frame(0)?;
        hoomd_gsd_file
            .append_frame(1)?
            .particles_type_id([0, 1, 1, 2])?;
        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path.clone(), Mode::Read)?;
        check!(gsd_file.find_chunk(0, "particles/typeid") == None);
        check!(gsd_file.find_chunk(0, "particles/N") == None);

        let data = gsd_file.iter_scalars::<u32>(1, "particles/typeid")?;
        itertools::assert_equal(data, [0_u32, 1, 1, 2]);
        let data = gsd_file.iter_scalars::<u32>(1, "particles/N")?;
        itertools::assert_equal(data, [4_u32]);

        Ok(())
    }

    #[test]
    fn particles_len_check() -> anyhow::Result<()> {
        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        check!(let Err(AppendError::InconsistentLength(_, _)) = hoomd_gsd_file
            .append_frame(1)?
            .particles_type_id([0, 1, 1, 2])?
            .particles_orientation([Versor::default(); 3]));

        Ok(())
    }

    #[test]
    fn particles_types() -> anyhow::Result<()> {
        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        hoomd_gsd_file.append_frame(0)?;
        hoomd_gsd_file
            .append_frame(1)?
            .particles_types(["A", "B", "linker"])?;
        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path.clone(), Mode::Read)?;
        check!(gsd_file.find_chunk(0, "particles/types") == None);

        let data: Vec<_> = gsd_file
            .iter_arrays::<u8, 64>(1, "particles/types")?
            .collect();
        assert!(data.len() == 3);
        check!(data[0][0] == "A".as_bytes()[0]);
        check!(data[0][1] == 0);

        check!(data[1][0] == "B".as_bytes()[0]);
        check!(data[1][1] == 0);

        check!(data[2][0..6] == *"linker".as_bytes());
        check!(data[0][6] == 0);

        Ok(())
    }

    #[test]
    fn log_scalar() -> anyhow::Result<()> {
        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        hoomd_gsd_file.append_frame(0)?;
        hoomd_gsd_file
            .append_frame(1)?
            .log_scalar("test", 4.2_f64)?;
        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path.clone(), Mode::Read)?;
        check!(gsd_file.find_chunk(0, "log/test") == None);

        let data = gsd_file.iter_scalars::<f64>(1, "log/test")?;
        itertools::assert_equal(data, [4.2_f64]);

        Ok(())
    }

    #[test]
    fn log_scalars() -> anyhow::Result<()> {
        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        hoomd_gsd_file.append_frame(0)?;
        hoomd_gsd_file
            .append_frame(1)?
            .log_scalars("test", [4.2_f64, 8.4])?;
        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path.clone(), Mode::Read)?;
        check!(gsd_file.find_chunk(0, "log/test") == None);

        let data = gsd_file.iter_scalars::<f64>(1, "log/test")?;
        itertools::assert_equal(data, [4.2_f64, 8.4]);

        Ok(())
    }

    #[test]
    fn log_arrays() -> anyhow::Result<()> {
        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        hoomd_gsd_file.append_frame(0)?;
        hoomd_gsd_file
            .append_frame(1)?
            .log_arrays("test", [[4.2_f64, 8.4], [1.0, 2.0]])?;
        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path.clone(), Mode::Read)?;
        check!(gsd_file.find_chunk(0, "log/test") == None);

        let data = gsd_file.iter_arrays::<f64, 2>(1, "log/test")?;
        itertools::assert_equal(data, [[4.2_f64, 8.4], [1.0, 2.0]]);

        Ok(())
    }
}
