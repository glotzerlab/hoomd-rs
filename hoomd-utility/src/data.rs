// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Data logging methods.

use parquet::{
    file::{properties::WriterProperties, writer::SerializedFileWriter},
    record::RecordWriter,
};
use std::{fs::File, io, path::Path, sync::Arc};
use thiserror::Error;

/// The default maximum buffer size.
const DEFAULT_MAXIMUM_BUFFER_SIZE: usize = 2_usize.pow(17);

/// Enumerate possible sources of error when writing log files.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum Error {
    /// Encountered an IO error.
    #[error("I/O error")]
    IO(#[from] io::Error),

    /// Encountered an IO error.
    #[error("Parquet error")]
    Parquet(#[from] parquet::errors::ParquetError),
}

/// Create a unique file by appending an integer index.
///
/// The first file created will be "{path}.1". If that exists,
/// the function will create "{path}.2" and so on.
fn create_unique_file<P: AsRef<Path>>(path: P) -> io::Result<File> {
    let mut index: u32 = 0;

    loop {
        let numbered_path = path.as_ref().with_added_extension(index.to_string());

        match File::create_new(numbered_path) {
            Ok(file) => return Ok(file),
            Err(error) => match error.kind() {
                io::ErrorKind::AlreadyExists => (),
                _ => return Err(error),
            },
        }

        index += 1;
    }
}

/// Write log records to a Parquet data file.
///
/// Use `ParquetLogger` to open a parquet file and write one log record at a
/// time. `ParquetLogger` buffers up to [`maximum_buffer_size`] log records in
/// memory and then synchronizes the buffer to the file.
///
/// Derive `ParquetRecordWriter` on your log record struct, [`create`] a
/// `ParquetLogger`, and [`log`] records to the file.
///
/// [`create`]: Self::create
/// [`log`]: Self::log
/// [`maximum_buffer_size`]: Self::maximum_buffer_size
///
/// # Example
/// ```
/// use hoomd_utility::data::ParquetLogger;
/// use parquet_derive::ParquetRecordWriter;
///
/// #[derive(ParquetRecordWriter)]
/// pub struct LogRecord {
///     /// The simulation step.
///     step: u64,
///
///     /// Total system potential energy.
///     potential_energy: f64,
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use tempfile::tempdir;
/// # let tmp_dir = tempdir().expect("temp dir should be created");
/// # let path = tmp_dir.path().join("log.parquet");
/// // let path = "log.parquet";
/// let mut parquet_logger = ParquetLogger::<LogRecord>::create(path)?;
/// parquet_logger.log(LogRecord {
///     step: 0,
///     potential_energy: 1.0,
/// })?;
/// parquet_logger.log(LogRecord {
///     step: 1,
///     potential_energy: 2.0,
/// })?;
///
/// # Ok(())
/// # }
/// ```
pub struct ParquetLogger<T>
where
    for<'a> &'a [T]: RecordWriter<T>,
{
    /// Parquet writer.
    writer: SerializedFileWriter<File>,

    /// Logged records that have not been written to the file.
    buffer: Vec<T>,

    /// Buffer at most this many records.
    maximum_buffer_size: usize,
}

impl<T> ParquetLogger<T>
where
    for<'a> &'a [T]: RecordWriter<T>,
{
    /// Create a new parquet file.
    ///
    /// `create` *overwrites* any existing file at `path`. The default
    /// [`maximum_buffer_size`] is $` 2^{17} `$ records.
    ///
    /// [`maximum_buffer_size`]: Self::maximum_buffer_size
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when there is I/O error opening the file
    /// or `parquet` fails to initialize the file.
    ///
    /// [`Error`]: enum@crate::data::Error
    ///
    /// # Example
    /// ```
    /// use hoomd_utility::data::ParquetLogger;
    /// use parquet_derive::ParquetRecordWriter;
    ///
    /// #[derive(ParquetRecordWriter)]
    /// pub struct LogRecord {
    ///     /// The simulation step.
    ///     step: u64,
    ///
    ///     /// Total system potential energy.
    ///     potential_energy: f64,
    /// }
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("log.parquet");
    /// // let path = "log.parquet";
    /// let mut parquet_logger = ParquetLogger::<LogRecord>::create(path)?;
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let buffer = Vec::<T>::new();
        let schema = buffer.as_slice().schema()?;
        let props = Arc::new(WriterProperties::builder().build());
        let log_file = File::create(path)?;
        let writer = SerializedFileWriter::new(log_file, schema, props)?;

        Ok(Self {
            writer,
            buffer,
            maximum_buffer_size: DEFAULT_MAXIMUM_BUFFER_SIZE,
        })
    }

    /// Create a unique parquet file.
    ///
    /// Parquet files cannot be appended to once written. The accepted solution
    /// by the Parquet developers is to create many files (`file.parquet.0`,
    /// `file.parquet.1`, and so on) that you concatenate on read.
    ///
    /// `create_unique` facilitates this process by appending `.0`, `.1`, ... `.{N}`.
    /// to the given `path`. `create_unique` attempts all extensions in order,
    /// continuing to the next increment each time it finds an existing file.
    ///
    /// The default [`maximum_buffer_size`] is $` 2^{17} `$ records.
    ///
    /// [`maximum_buffer_size`]: Self::maximum_buffer_size
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when there is I/O error opening the file
    /// or `parquet` fails to initialize the file.
    ///
    /// [`Error`]: enum@crate::data::Error
    ///
    /// # Example
    /// ```
    /// use hoomd_utility::data::ParquetLogger;
    /// use parquet_derive::ParquetRecordWriter;
    ///
    /// #[derive(ParquetRecordWriter)]
    /// pub struct LogRecord {
    ///     /// The simulation step.
    ///     step: u64,
    ///
    ///     /// Total system potential energy.
    ///     potential_energy: f64,
    /// }
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("log.parquet");
    /// // let path = "log.parquet";
    /// let mut parquet_logger = ParquetLogger::<LogRecord>::create_unique(path)?;
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn create_unique<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let buffer = Vec::<T>::new();
        let schema = buffer.as_slice().schema()?;
        let props = Arc::new(WriterProperties::builder().build());
        let log_file = create_unique_file(path)?;
        let writer = SerializedFileWriter::new(log_file, schema, props)?;

        Ok(Self {
            writer,
            buffer,
            maximum_buffer_size: DEFAULT_MAXIMUM_BUFFER_SIZE,
        })
    }

    /// Log a record to the file.
    ///
    /// `log` buffers up to [`maximum_buffer_size`] records in memory before
    /// synchronizing them to the file. Call `log` once for each record you
    /// produce during a simulation.
    ///
    /// [`maximum_buffer_size`]: Self::maximum_buffer_size
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when there is I/O error writing to the file
    /// or `parquet` is unable to write the buffer.
    ///
    /// [`Error`]: enum@crate::data::Error
    ///
    /// # Example
    /// ```
    /// use hoomd_utility::data::ParquetLogger;
    /// use parquet_derive::ParquetRecordWriter;
    ///
    /// #[derive(ParquetRecordWriter)]
    /// pub struct LogRecord {
    ///     /// The simulation step.
    ///     step: u64,
    ///
    ///     /// Total system potential energy.
    ///     potential_energy: f64,
    /// }
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use tempfile::tempdir;
    /// # let tmp_dir = tempdir().expect("temp dir should be created");
    /// # let path = tmp_dir.path().join("log.parquet");
    /// // let path = "log.parquet";
    /// let mut parquet_logger = ParquetLogger::<LogRecord>::create(path)?;
    /// parquet_logger.log(LogRecord {
    ///     step: 0,
    ///     potential_energy: 1.0,
    /// })?;
    /// parquet_logger.log(LogRecord {
    ///     step: 1,
    ///     potential_energy: 2.0,
    /// })?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn log(&mut self, record: T) -> Result<(), Error> {
        self.buffer.push(record);

        if self.buffer.len() >= self.maximum_buffer_size {
            self.sync()?;
        }

        Ok(())
    }

    /// Maximum number of log records to buffer in memory.
    #[inline]
    #[must_use]
    pub fn maximum_buffer_size(&self) -> usize {
        self.maximum_buffer_size
    }

    /// Mutable reference to the maximum number of log records to buffer in memory.
    #[inline]
    pub fn maximum_buffer_size_mut(&mut self) -> &mut usize {
        &mut self.maximum_buffer_size
    }

    /// Synchronize the buffer to the file.
    ///
    /// `sync` writes all currently buffered log records to a row group. A single
    /// parquet file can contain at most $` 2^{15} `$ row groups, so call `sync`
    /// only at specific points when data must be on disk.
    ///
    /// All buffered data is *automatically* synchronized when the logger is
    /// dropped.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when there is I/O error writing to the file
    /// or `parquet` is unable to write the buffer.
    ///
    /// [`Error`]: enum@crate::data::Error
    #[inline]
    pub fn sync(&mut self) -> Result<(), Error> {
        if !self.buffer.is_empty() {
            let mut row_group = self.writer.next_row_group()?;
            self.buffer.as_slice().write_to_row_group(&mut row_group)?;
            row_group.close()?;
            self.buffer.clear();
            self.writer.flush()?;
        }
        Ok(())
    }
}

/// Synchronize the buffer and close the parquet file.
impl<T> Drop for ParquetLogger<T>
where
    for<'a> &'a [T]: RecordWriter<T>,
{
    #[inline]
    fn drop(&mut self) {
        let _ = self.sync();
        let _ = self.writer.finish();
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use assert2::check;
    use parquet_derive::ParquetRecordWriter;
    use tempfile::tempdir;

    #[derive(ParquetRecordWriter)]
    struct LogRecord {
        a: f64,
    }

    #[test]
    fn test_create_unique() -> anyhow::Result<()> {
        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.parquet");
        let _ = ParquetLogger::create_unique(path.clone())?;

        check!(fs::exists(path.with_added_extension("0"))?);
        check!(!fs::exists(path.with_added_extension("1"))?);
        check!(!fs::exists(path.with_added_extension("2"))?);

        let _ = ParquetLogger::create_unique(path.clone())?;

        check!(fs::exists(path.with_added_extension("0"))?);
        check!(fs::exists(path.with_added_extension("1"))?);
        check!(!fs::exists(path.with_added_extension("2"))?);

        let _ = ParquetLogger::create_unique(path.clone())?;

        check!(fs::exists(path.with_added_extension("0"))?);
        check!(fs::exists(path.with_added_extension("1"))?);
        check!(fs::exists(path.with_added_extension("2"))?);

        Ok(())
    }
}
