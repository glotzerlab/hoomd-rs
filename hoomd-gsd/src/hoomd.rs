use std::{array, num::TryFromIntError, path::Path};
use thiserror::Error;

use hoomd_vector::{Cartesian, Versor};

use crate::file_layer::{GsdFile, OpenError, WriteError};

pub struct HoomdGsdFile {
    gsd_file: GsdFile,
    // TODO: auto flush timeout, time of last flush
}

pub struct Frame<'a> {
    hoomd_gsd_file: &'a mut HoomdGsdFile,

    particles_n: Option<u32>,
}

/// Errors that can occur while appending to a HOOMD GSD frame.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum AppendError {
    /// This data chunk does not match the dimensions of those previously written.
    #[error("The length of data chunk {0} does not match those previously written")]
    InconsistentLength(String),
    
    /// Write to the file.
    #[error("cannot write to the file")]
    Write(#[from] WriteError),

    /// Too many entries to write.
    #[error("cannot write {0} entries to data chunk {1}")]
    ChunkTooLarge(usize, String, #[source] TryFromIntError),

    /// Dimension is out of bounds.
    #[error("invalid dimensions {0}: expected 2 or 3")]
    InvalidDimensions(u8),
}

impl HoomdGsdFile {
    
    /// Overwrite an existing HOOMD GSD file (or create a new file).
    ///
    /// Creates a GSD file at the given path, overwriting any file that may already
    /// exist. When successful, return a [`HoomdGsdFile`] opened in write mode.
    ///
    /// TODO: Document default autoflush settings.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_gsd::hoomd::HoomdGsdFile;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("test.gsd");
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
    #[inline]
    pub fn create<P: AsRef<Path>>(
        path: P,
    ) -> Result<Self, OpenError> {
        let version = env!("CARGO_PKG_VERSION");
        let application = format!("hoomd-rs {version}");
        let gsd_file = GsdFile::create(path, &application, "hoomd", (1, 4))?;

        Ok(Self { gsd_file })
    }

    #[inline]
    pub fn append_frame(&mut self, step: u64) -> Result<Frame<'_>, AppendError> {
        self.gsd_file.write_scalars("configuration/step", [step])?;
        Ok(Frame {
            hoomd_gsd_file: self,
            particles_n: None,
        })
    }
}

impl Frame<'_> {
    pub fn configuration_dimensions(mut self, dimensions: u8) -> Result<Self, AppendError>
    {
    let chunk_name = "configuration/dimensions";

    if dimensions < 2 || dimensions > 3 {
        return Err(AppendError::InvalidDimensions(dimensions));
    }

    self.hoomd_gsd_file.gsd_file.write_scalars(chunk_name, [dimensions])?;
    
    Ok(self)
    }

    pub fn configuration_box(mut self, values: [f64; 6]) -> Result<Self, AppendError>
    {
    let chunk_name = "configuration/box";

    let values: [f32; 6] = array::from_fn(|i| values[i] as f32);
    self.hoomd_gsd_file.gsd_file.write_scalars(chunk_name, values)?;
    
    Ok(self)
    }

    pub fn particles_position<I>(mut self, position: I) -> Result<Self, AppendError>
    where
        I: IntoIterator<Item = Cartesian<3>>,
        I::IntoIter: ExactSizeIterator,
    {
    let chunk_name = "particles/position";
    let iter = position.into_iter();

    if let Some(n) = self.particles_n {
        if iter.len() != n as usize {
            return Err(AppendError::InconsistentLength(chunk_name.to_string()));
        }
    } else {
        let n = iter.len().try_into().map_err(|e|
            AppendError::ChunkTooLarge(iter.len(), chunk_name.to_string(), e))?;
        self.hoomd_gsd_file.gsd_file.write_scalars("particles/N", [n])?;
        self.particles_n = Some(n);
    }
    
    self.hoomd_gsd_file.gsd_file.write_arrays(chunk_name,
        iter.map(|v| -> [f32; 3] { array::from_fn(|i| v[i] as f32)}))?;
    
    Ok(self)
    }

    pub fn particles_orientation<I>(mut self, orientation: I) -> Result<Self, AppendError>
    where
        I: IntoIterator<Item = Versor>,
        I::IntoIter: ExactSizeIterator,
    {
    let chunk_name = "particles/orientation";
    let iter = orientation.into_iter();

    if let Some(n) = self.particles_n {
        if iter.len() != n as usize {
            return Err(AppendError::InconsistentLength(chunk_name.to_string()));
        }
    } else {
        let n = iter.len().try_into().map_err(|e|
            AppendError::ChunkTooLarge(iter.len(), chunk_name.to_string(), e))?;
        self.hoomd_gsd_file.gsd_file.write_scalars("particles/N", [n])?;
        self.particles_n = Some(n);
    }
    
    self.hoomd_gsd_file.gsd_file.write_arrays(chunk_name,
        iter.map(|v| {
            [v.get().scalar as f32, v.get().vector[0] as f32, v.get().vector[1] as f32, v.get().vector[2] as f32]
        }
            ))?;
   
    Ok(self)
    }
}

impl Drop for Frame<'_> {
    fn drop(&mut self) {
        let _ = self.hoomd_gsd_file.gsd_file.end_frame();
        // TODO: auto flush
    }
}
