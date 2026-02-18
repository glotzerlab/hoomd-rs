// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Common data logging methods.

use parquet::{
    file::{properties::WriterProperties, writer::SerializedFileWriter},
    record::RecordWriter,
};
use std::{fs::File, io, path::Path, sync::Arc};
use thiserror::Error;

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


/// TODO: Document
pub struct ParquetLogger<T> where
for<'a> &'a [T]: RecordWriter<T>
{
    /// Parquet writer.
    writer: SerializedFileWriter<File>,

    /// Logged records that have not been written to the file.
    buffer: Vec<T>,

    /// Buffer at most this many records.
    maximum_buffer_size: usize,
}

impl<T> ParquetLogger<T> where
for<'a> &'a [T]: RecordWriter<T>,
{

    #[inline]
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let buffer = Vec::<T>::new();
        let schema = buffer.as_slice().schema()?;
        let props = Arc::new(WriterProperties::builder().build());
        let log_file = File::create(path)?;
        let writer = SerializedFileWriter::new(log_file, schema, props)?;

        Ok(Self { writer, buffer, maximum_buffer_size: 2_usize.pow(17) })
    }

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

    #[inline]
    pub fn log(&mut self, record: T) -> Result<(), Error> {
        self.buffer.push(record);

        if self.buffer.len() >= self.maximum_buffer_size {
            self.sync()?;
        }

        Ok(())
    }
}

impl<T> Drop for ParquetLogger<T> where
for<'a> &'a [T]: RecordWriter<T> 
{
    #[inline]
    fn drop(&mut self) {
        let _ = self.sync();
        let _ = self.writer.finish();
    }
}
