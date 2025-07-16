// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Read and write data chunks in GSD files.

Use [`GsdFile`] to interact with GSD files on the filesystem.
 */

use itertools::Itertools;
use memmap2::Mmap;
use std::cmp::{Ord, Ordering, PartialOrd};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, SeekFrom, prelude::*};
use std::num::TryFromIntError;
use std::path::{Path, PathBuf};
use std::string::FromUtf8Error;
use thiserror::Error;

/// The name buffer is a multiple of `NAME_SIZE` bytes.
const NAME_SIZE: u64 = 64;

/// Number of bytes in an index entry.
const INDEX_ENTRY_SIZE: u64 = 32;

/// Index entry as a usize.
const INDEX_ENTRY_USIZE: usize = 32;

/// Number of bytes in the header.
const HEADER_SIZE: u64 = 256;

/// Header size as a usize.
const HEADER_USIZE: usize = 256;

/// Magic value identifying a GSD file
const MAGIC_ID: u64 = 0x65DF_65DF_65DF_65DF;

/// Current GSD file version
const CURRENT_FILE_VERSION: (u16, u16) = (2, 1);

/// The size of the file index in new GSD files.
const INITIAL_INDEX_SIZE: u64 = 128;

/// Initial name list size
const INITIAL_NAME_LIST_SIZE: u64 = 1024;

/// Initial maximum write buffer size.
const INITIAL_MAXIMUM_WRITE_BUFFER_SIZE: usize = 1024 * 1024;

/// Errors that can occur during while decoding file content.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum DecodeError {
    /// Encountered an IO error.
    #[error("I/O error")]
    IO(#[from] io::Error),

    /// Invalid application string.
    #[error("invalid `application`")]
    InvalidApplication(#[source] FromUtf8Error),

    /// Invalid schema string.
    #[error("invalid `schema`")]
    InvalidSchema(#[source] FromUtf8Error),

    /// Invalid file identifier.
    #[error("invalid file identifier `{0}`")]
    InvalidFileIdentifier(u64),

    /// Index outside the file.
    #[error("index out of bounds (location={0}, length={1})")]
    IndexOutOfBounds(u64, u64),

    /// Name list outside the file.
    #[error("name list out of bounds (location={0}, length={1})")]
    NameListOutOfBounds(u64, u64),

    /// Name list outside the file.
    #[error("name list not terminated")]
    NameListNotTerminated,

    /// Cannot add any more chunk names.
    #[error("too many chunk names")]
    NameListOverflow,

    /// Unsupported version.
    #[error("unsupported GSD file version ({0}, {1})")]
    UnsupportedVersion(u16, u16),

    /// An index is not addressable.
    #[error("file index not addressable")]
    UnaddressableIndex(#[source] TryFromIntError),

    /// File content is not addressable.
    #[error("file content not addressable")]
    UnaddressableContent(#[source] TryFromIntError),

    /// Invalid chunk name string.
    #[error("invalid chunk name")]
    InvalidChunkName(#[source] FromUtf8Error),

    /// Duplicate chunk name.
    #[error("duplicate chunk name")]
    DuplicateChunkName,

    /// Corrupt index entry.
    #[error("corrupt index entry: `{0:?}`")]
    CorruptIndexEntry(IndexEntry),

    /// Invalid type.
    #[error("expected type {0}, found {1}")]
    InvalidType(u8, u8),

    /// Invalid number of columns.
    #[error("expected to read read {0} columns, found {1}")]
    InvalidColumns(usize, u32),

    /// Invalid string in chunk.
    #[error("invalid string")]
    InvalidString(#[source] FromUtf8Error),
}

/// Errors that can occur while creating or opening a file.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum OpenError {
    /// Encountered an IO error.
    #[error("I/O error while creating or opening `{0}`")]
    IO(PathBuf, #[source] io::Error),

    /// Cannot decode the file contents.
    #[error("cannot decode `{0}`")]
    Decode(PathBuf, #[source] DecodeError),

    /// Name length overflow.
    #[error("the name `{0}` is too long")]
    NameTooLong(String),
}

/// Errors that can occur while reading from a file.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum ReadError {
    /// Chunk not found.
    #[error("chunk `{0}` not found in frame {1}")]
    ChunkNotFound(String, u64),

    /// Cannot decode the file contents.
    #[error("cannot decode chunk `{0}` at frame {1}")]
    Decode(String, u64, #[source] DecodeError),
}

/// Errors that can occur while encoding data to a file.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum EncodeError {
    /// Encountered an I/O error.
    #[error("I/O error")]
    IO(#[from] io::Error),

    /// Cannot add any more chunk names.
    #[error("too many chunk names")]
    NameListOverflow,

    /// File is not writable.
    #[error("file opened in read-only mode")]
    NotWritable,

    /// A chunk name was duplicated in a single frame.
    #[error("chunk `{0}` has already been written in frame {1}")]
    DuplicateChunkName(String, u64),

    /// Invalid number of columns.
    #[error("the number of columns must be greater than zero and fit in a u32, got {0}")]
    InvalidColumns(usize),

    /// Index outside the file.
    #[error("index out of bounds (location={0}, length={1})")]
    IndexOutOfBounds(u64, u64),

    /// Name list outside the file.
    #[error("name list out of bounds (location={0}, length={1})")]
    NameListOutOfBounds(u64, u64),
}

/// Errors that can occur while writing to a file.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum WriteError {
    /// Cannot encode a write to the file.
    #[error("cannot encode chunk `{0}` at frame {1}")]
    Encode(String, u64, #[source] EncodeError),
}

// FUTURE: Replace ArrayChunks with itertools implementation when available
// FUTURE: Replace ArrayChunks with std library implementation when iter_array_chunks is stabilized

/// Iterate over arrays of size M
struct ArrayChunks<I, const M: usize> {
    /// The iterator over scalars
    iter: I,
}

impl<T, I, const M: usize> Iterator for ArrayChunks<I, M>
where
    I: Iterator<Item = T>,
{
    type Item = [T; M];

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next_array::<M>()
    }
}

impl<T, I, const M: usize> ExactSizeIterator for ArrayChunks<I, M>
where
    I: ExactSizeIterator<Item = T>,
{
    fn len(&self) -> usize {
        self.iter.len() / M
    }
}

/** Implement a sealed trait for each data type supported by GSD.

This enables generic implementations that operate on these types.
*/
mod private {
    /// Seal the data type traits so that users cannot add new types.
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/** Data types that can be stored in chunk arrays.

GSD files store arrays of data of one of the following types:
* [`u8`]
* [`u16`]
* [`u32`]
* [`u64`]
* [`i8`]
* [`i16`]
* [`i32`]
* [`i64`]
* [`f32`]
* [`f64`]

The [`Type`] trait facilitates the generic methods including
[`GsdFile::iter_scalars`], [`GsdFile::write_scalars`], and others. When needed,
pass the type explicitly to these methods to read or write data chunks of the
given type. In some cases, the Rust compiler may be able to determine the type
from context.

# Example

```
use hoomd_gsd::file_layer::GsdFile;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# use tempfile::tempdir;
# let tmp_dir = tempdir().expect("temp dir should be created");
# let path = tmp_dir.path().join("test.gsd");
let mut gsd_file = GsdFile::create_new(path, "example", "hoomd", (1, 4))?;
gsd_file.write_scalars("configuration/box", &[10.0f32, 20.0, 15.0, 0.0, 0.0, 0.0])?;
gsd_file.end_frame()?;
gsd_file.sync_all()?;

let box_iter = gsd_file.iter_scalars::<f32>(0, "configuration/box")?;
itertools::assert_equal(box_iter, [10.0, 20.0, 15.0, 0.0, 0.0, 0.0]);
# Ok(())
# }
```
*/
pub trait Type: private::Sealed {
    /// Value denoting this type in the file layer.
    #[doc(hidden)]
    fn gsd_data_type() -> u8;

    /** Convert a native endian byte slice to this type.

    This is not the proper idiomatic way to do this, but it gets the job done
    with minimal lines of code.
    */
    #[doc(hidden)]
    fn from_ne_byte_slice(bytes: &[u8]) -> Self;

    /** Append this type to a native endian byte array.

    This is not the proper idiomatic way to do this, but it gets the job done
    with minimal lines of code.
    */
    #[doc(hidden)]
    fn append_ne_bytes(&self, v: &mut Vec<u8>);
}

impl Type for u8 {
    #[inline]
    fn gsd_data_type() -> u8 {
        1
    }
    #[inline]
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        bytes[0]
    }
    #[inline]
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}
impl Type for u16 {
    #[inline]
    fn gsd_data_type() -> u8 {
        2
    }
    #[inline]
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        u16::from_ne_bytes(bytes.try_into().expect("byte slice should contain 2 bytes"))
    }
    #[inline]
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}
impl Type for u32 {
    #[inline]
    fn gsd_data_type() -> u8 {
        3
    }
    #[inline]
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        u32::from_ne_bytes(bytes.try_into().expect("byte slice should contain 4 bytes"))
    }
    #[inline]
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}
impl Type for u64 {
    #[inline]
    fn gsd_data_type() -> u8 {
        4
    }
    #[inline]
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        u64::from_ne_bytes(bytes.try_into().expect("byte slice should contain 8 bytes"))
    }
    #[inline]
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}
impl Type for i8 {
    #[inline]
    fn gsd_data_type() -> u8 {
        5
    }
    #[inline]
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        i8::from_ne_bytes(bytes.try_into().expect("byte slice should contain 1 byte"))
    }
    #[inline]
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}
impl Type for i16 {
    #[inline]
    fn gsd_data_type() -> u8 {
        6
    }
    #[inline]
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        i16::from_ne_bytes(bytes.try_into().expect("byte slice should contain 2 bytes"))
    }
    #[inline]
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}
impl Type for i32 {
    #[inline]
    fn gsd_data_type() -> u8 {
        7
    }
    #[inline]
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        i32::from_ne_bytes(bytes.try_into().expect("byte slice should contain 4 bytes"))
    }
    #[inline]
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}
impl Type for i64 {
    #[inline]
    fn gsd_data_type() -> u8 {
        8
    }
    #[inline]
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        i64::from_ne_bytes(bytes.try_into().expect("byte slice should contain 8 bytes"))
    }
    #[inline]
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}
impl Type for f32 {
    #[inline]
    fn gsd_data_type() -> u8 {
        9
    }
    #[inline]
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        f32::from_ne_bytes(bytes.try_into().expect("byte slice should contain 8 bytes"))
    }
    #[inline]
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}
impl Type for f64 {
    #[inline]
    fn gsd_data_type() -> u8 {
        10
    }
    #[inline]
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        f64::from_ne_bytes(bytes.try_into().expect("byte slice should contain 8 bytes"))
    }
    #[inline]
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}

/// In memory representation of the GSD file header.
#[derive(Debug, PartialEq)]
pub(crate) struct GsdHeader {
    /// Magic number marking that this is a GSD file.
    magic: u64,

    /// Location of the chunk index in the file.
    index_location: u64,

    /// Number of index entries that will fit in the space allocated.
    index_allocated_entries: u64,

    /// Location of the name list in the file.
    namelist_location: u64,

    /// Number of bytes in the namelist divided by `NAME_SIZE`.
    namelist_allocated_entries: u64,

    /// Schema version.
    schema_version: (u16, u16),

    /// GSD file format version.
    gsd_version: (u16, u16),

    /// Name of the application that generated this file.
    application: String,

    /// Name of data schema.
    schema: String,
}

/// Details about the name list
#[derive(Debug)]
struct NameList {
    /// Name/id mapping.
    name_id: HashMap<String, u16>,

    /// Number of names in the map.
    n_names: u16,

    /// Insert position in the name list.
    insert_position: u64,

    /// Name write buffer.
    buffer: Vec<u8>,
}

/** Details about the index.

* `n` counts the number of entries stored in the actual file.
* `buffer` stores index entries in memory that have not yet been written to the
  tile (as bytes).
* `pending` counts the number of entries that are pending in the current frame.

Pending entries are those where `write_*` has been called, but not yet
`end_frame`. These should not be synced to the file to avoid having
partial frames in the file.
*/
#[derive(Debug)]
struct Index {
    /// Number of index entries stored in the file.
    n: u64,

    /// Index entry buffer.
    buffer: Vec<IndexEntry>,

    /// Index entry byte buffer.
    byte_buffer: Vec<u8>,

    /// Pending entries.
    pending: usize,

    /// Chunk ids that have been written in this frame.
    frame_names: HashSet<u16>,
}

/** Interact with GSD files on the filesystem.

# Overview

Open files with:
* [`open`](GsdFile::open)
* [`create`](GsdFile::create)
* [`create_new`](GsdFile::create_new)

Access file metadata with:
* [`n_frames`](GsdFile::n_frames)
* [`schema`](GsdFile::schema)
* [`schema_version`](GsdFile::schema_version)
* [`name_id`](GsdFile::name_id)
* [`find_chunk`](GsdFile::find_chunk)

Write data with:
* [`write_scalars`](GsdFile::write_scalars)
* [`write_arrays`](GsdFile::write_arrays)
* [`write_string`](GsdFile::write_string)
* [`end_frame`](GsdFile::end_frame)
* [`sync_all`](GsdFile::sync_all)

Read data with:
* [`iter_scalars`](GsdFile::iter_scalars)
* [`iter_arrays`](GsdFile::iter_arrays)
* [`read_string`](GsdFile::read_string)
*/
#[derive(Debug)]
pub struct GsdFile {
    /// The underlying file.
    file: File,

    /// The file's mode.
    mode: Mode,

    /// Parsed copy of the file's header.
    header: GsdHeader,

    /// Memory map of the file.
    mmap: Mmap,

    /// The name list.
    name_list: NameList,

    /// The index buffer.
    index: Index,

    /// The array data buffer.
    data_buffer: Vec<u8>,

    /// Record whether the data buffer has been flushed this frame.
    data_buffer_flushed: bool,

    /// Length of the file in bytes.
    file_len: u64,

    /// Index of the current buffered frame.
    buffer_frame: u64,

    /// Index of the current frame committed to the file.
    file_frame: u64,

    /// Write buffered data when more than `maximum_write_buffer_size` bytes are buffered.
    maximum_write_buffer_size: usize,
}

/** Properties that describe a given data chunk.

    GSD files store a set of arrays, uniquely identified by their *name* and
    *frame*. The [`GsdFile::find_chunk`] method search for a matching index
    entry. The returned [`IndexEntry`] (if present) also carries information
    about the dimension and type of the array.
*/
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct IndexEntry {
    /// Frame index of the chunk.
    frame: u64,

    /// Number of rows in the chunk.
    n: u64,

    /// Location of the chunk in the file.
    location: u64,

    /// Number of columns in the chunk.
    m: u32,

    /// Index of the chunk name in the name list.
    id: u16,

    /// Data type of the chunk.
    data_type: u8,

    /// Flags (unused)
    flags: u8,
}

/** Data types that can be stored in chunks.

Provided by [`IndexEntry::data_type`].
*/
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DataType {
    /// [`u8`]
    U8,
    /// [`u16`]
    U16,
    /// [`u32`]
    U32,
    /// [`u64`]
    U64,
    /// [`i8`]
    I8,
    /// [`i16`]
    I16,
    /// [`i32`]
    I32,
    /// [`i64`]
    I64,
    /// [`f32`]
    F32,
    /// [`f64`]
    F64,
    /// [`String`]
    String,
}

/** Choose how opened files can be accessed.

Pass a [`Mode`] value to [`GsdFile::open`].

In the [`Mode::Read`] mode, you can call methods that read the file, such as
[`GsdFile::find_chunk`] and [`GsdFile::iter_scalars`]. Calling methods that
write the file, such as [`GsdFile::write_scalars`] or [`GsdFile::sync_all`] will
result in an error.

In the [`Mode::Write`] mode, you can call both read and write methods.
*/
#[derive(Copy, Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Mode {
    /// Read-only.
    Read,
    /// Allow both read and write operations.
    Write,
}

/** Read the first u64 in a byte slice (native endian).

Returns the [`u64`] and the rest of the slice. Testing in Godbolt shows that
repeated calls to this method can be optimized to a simple series of mov
instructions.
*/
#[inline]
fn extract_ne_u64(bytes: &[u8]) -> (u64, &[u8]) {
    let (bytes, rest) = bytes.split_at(size_of::<u64>());
    (
        u64::from_ne_bytes(
            bytes
                .try_into()
                .expect("bytes slice should contain 8 bytes"),
        ),
        rest,
    )
}

/// Read the first u32 in a byte slice (native endian).
#[inline]
fn extract_ne_u32(bytes: &[u8]) -> (u32, &[u8]) {
    let (bytes, rest) = bytes.split_at(size_of::<u32>());
    (
        u32::from_ne_bytes(
            bytes
                .try_into()
                .expect("bytes slice should contain 4 bytes"),
        ),
        rest,
    )
}

/// Read the first u16 in a byte slice (native endian).
#[inline]
fn extract_ne_u16(bytes: &[u8]) -> (u16, &[u8]) {
    let (bytes, rest) = bytes.split_at(size_of::<u16>());
    (
        u16::from_ne_bytes(
            bytes
                .try_into()
                .expect("bytes slice should contain 2 bytes"),
        ),
        rest,
    )
}

/** Read the first null terminated string in a byte slice.

Returns the [`String`] without the null terminator. Also returns the rest of the
slice after consuming 1 null terminator.
*/
#[inline]
fn extract_null_terminated_utf8(bytes: &[u8]) -> Result<(String, &[u8]), FromUtf8Error> {
    let null_range_end = bytes
        .iter()
        .position(|&c| c == b'\0')
        .unwrap_or(bytes.len());
    let (bytes, mut rest) = bytes.split_at(null_range_end);
    let s = String::from_utf8(bytes.into())?;
    if !rest.is_empty() {
        (_, rest) = rest.split_at(1);
    }
    Ok((s, rest))
}

impl PartialOrd for IndexEntry {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for IndexEntry {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        (self.frame, self.id).cmp(&(other.frame, other.id))
    }
}

impl GsdHeader {
    /// Parse the header.
    fn try_from_ne_bytes(value: [u8; HEADER_USIZE]) -> Result<Self, DecodeError> {
        // Validate the magic number first to ensure that we expect the rest
        // of the header to be formatted appropriately. Otherwise, later
        // error checks in this method will be examining undefined data.
        let (magic, rest) = extract_ne_u64(&value);
        if magic != MAGIC_ID {
            return Err(DecodeError::InvalidFileIdentifier(magic));
        }

        let (index_location, rest) = extract_ne_u64(rest);
        let (index_allocated_entries, rest) = extract_ne_u64(rest);
        let (namelist_location, rest) = extract_ne_u64(rest);
        let (namelist_allocated_entries, rest) = extract_ne_u64(rest);

        // Verify that all locations are addressable in the memory map once on
        // initialization. That way, it is safe to cast from the various byte
        // locations to usize in the read methods.
        usize::try_from(index_location).map_err(DecodeError::UnaddressableIndex)?;
        usize::try_from(index_allocated_entries).map_err(DecodeError::UnaddressableIndex)?;
        usize::try_from(namelist_location).map_err(DecodeError::UnaddressableIndex)?;
        usize::try_from(namelist_allocated_entries).map_err(DecodeError::UnaddressableIndex)?;

        let (schema_version, rest) = extract_ne_u32(rest);
        let (gsd_version, rest) = extract_ne_u32(rest);
        let (application, _) =
            extract_null_terminated_utf8(rest).map_err(DecodeError::InvalidApplication)?;
        let (schema, _) =
            extract_null_terminated_utf8(&value[112..178]).map_err(DecodeError::InvalidSchema)?;

        let schema_version = (
            (schema_version >> 16) as u16,
            (schema_version & 0xffff) as u16,
        );
        let gsd_version = ((gsd_version >> 16) as u16, (gsd_version & 0xffff) as u16);

        // Cannot pretend to have a valid header unless we are also sure that
        // the file version is one that we understand.
        if !((2, 0)..(3, 0)).contains(&gsd_version) {
            return Err(DecodeError::UnsupportedVersion(
                gsd_version.0,
                gsd_version.1,
            ));
        }

        Ok(GsdHeader {
            magic,
            index_location,
            index_allocated_entries,
            namelist_location,
            namelist_allocated_entries,
            schema_version,
            gsd_version,
            application,
            schema,
        })
    }

    /// Encode the header into bytes following the GSD specification.
    #[inline]
    fn to_ne_bytes(&self) -> [u8; HEADER_USIZE] {
        let mut result = [0u8; HEADER_USIZE];
        result[0..8].copy_from_slice(&self.magic.to_ne_bytes());
        result[8..16].copy_from_slice(&self.index_location.to_ne_bytes());
        result[16..24].copy_from_slice(&self.index_allocated_entries.to_ne_bytes());
        result[24..32].copy_from_slice(&self.namelist_location.to_ne_bytes());
        result[32..40].copy_from_slice(&self.namelist_allocated_entries.to_ne_bytes());
        let schema_version =
            u32::from(self.schema_version.0) << 16 | u32::from(self.schema_version.1);
        result[40..44].copy_from_slice(&schema_version.to_ne_bytes());
        let gsd_version: u32 = u32::from(self.gsd_version.0) << 16 | u32::from(self.gsd_version.1);
        result[44..48].copy_from_slice(&gsd_version.to_ne_bytes());
        result[48..48 + self.application.len()].copy_from_slice(self.application.as_bytes());
        result[112..112 + self.schema.len()].copy_from_slice(self.schema.as_bytes());

        result
    }
}

impl IndexEntry {
    /// Trajectory frame number.
    #[must_use]
    #[inline]
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Number of rows in the array.
    #[must_use]
    #[inline]
    pub fn rows(&self) -> u64 {
        self.n
    }

    /// Number of columns in the array.
    #[must_use]
    #[inline]
    pub fn columns(&self) -> u32 {
        self.m
    }

    /** The array's data type.

    Returns [`Some(data_type)`](Option::Some) when the type is known and
    [`None`] when it is not.

    # Example
    ```
    use hoomd_gsd::file_layer::{DataType, IndexEntry};

    # fn do_something() { }
    # fn func(index_entry: &IndexEntry) {
    match index_entry.data_type() {
        Some(DataType::F32) => do_something(),
        _ => (),
    }
    # }
    ```
    */
    #[must_use]
    #[inline]
    pub fn data_type(&self) -> Option<DataType> {
        match self.data_type {
            1 => Some(DataType::U8),
            2 => Some(DataType::U16),
            3 => Some(DataType::U32),
            4 => Some(DataType::U64),
            5 => Some(DataType::I8),
            6 => Some(DataType::I16),
            7 => Some(DataType::I32),
            8 => Some(DataType::I64),
            9 => Some(DataType::F32),
            10 => Some(DataType::F64),
            11 => Some(DataType::String),
            _ => None,
        }
    }

    /// Parse an index entry.
    #[inline]
    fn from_ne_bytes(value: [u8; 32]) -> Self {
        let (frame, rest) = extract_ne_u64(&value);
        let (n, rest) = extract_ne_u64(rest);
        let (location, rest) = extract_ne_u64(rest);
        let (m, rest) = extract_ne_u32(rest);
        let (id, rest) = extract_ne_u16(rest);
        let data_type = rest[0];
        let flags = rest[1];
        Self {
            frame,
            n,
            location,
            m,
            id,
            data_type,
            flags,
        }
    }

    /// Encode an index entry.
    #[inline]
    fn to_ne_bytes(self) -> [u8; INDEX_ENTRY_USIZE] {
        let mut result = [0u8; INDEX_ENTRY_USIZE];
        result[0..8].copy_from_slice(&self.frame.to_ne_bytes());
        result[8..16].copy_from_slice(&self.n.to_ne_bytes());
        result[16..24].copy_from_slice(&self.location.to_ne_bytes());
        result[24..28].copy_from_slice(&self.m.to_ne_bytes());
        result[28..30].copy_from_slice(&self.id.to_ne_bytes());
        result[30] = self.data_type;
        result[31] = self.flags;

        result
    }
}

impl GsdFile {
    /** Open a GSD file with the given mode.

    # Examples

    Open a file for reading:
    ```
    use hoomd_gsd::file_layer::{GsdFile, Mode};
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    # use tempfile::tempdir;
    # let tmp_dir = tempdir().expect("temp dir should be created");
    # let path = tmp_dir.path().join("test.gsd");
    # GsdFile::create_new(path.clone(), "example", "hoomd", (1, 4))?;
    let gsd_file = GsdFile::open(path, Mode::Read);
    # Ok(())
    # }
    ```

    Open a file for both reading and writing:
    ```
    use hoomd_gsd::file_layer::{GsdFile, Mode};
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    # use tempfile::tempdir;
    # let tmp_dir = tempdir().expect("temp dir should be created");
    # let path = tmp_dir.path().join("test.gsd");
    # GsdFile::create_new(path.clone(), "example", "hoomd", (1, 4))?;
    let gsd_file = GsdFile::open(path, Mode::Write);
    # Ok(())
    # }
    ```

    # Errors

    Returns a [`OpenError`] when any of the following occur:
    * The file does not exist.
    * The file is corrupt, unreadable, or there is an I/O error (see
      [`DecodeError`]).
    */
    #[inline]
    pub fn open<P: AsRef<Path>>(path: P, mode: Mode) -> Result<Self, OpenError> {
        let file = File::open(&path).map_err(|e| OpenError::IO(path.as_ref().into(), e))?;
        GsdFile::from_file(file, mode).map_err(|e| OpenError::Decode(path.as_ref().into(), e))
    }

    /** Overwrite an existing GSD file (or create a new file).

    Creates a GSD file at the given path, overwriting any file that may already
    exist. When successful, return a [`GsdFile`] opened in write mode.

    Each GSD file contains metadata describing which application created the
    file, the data chunk schema, and the schema's version. `application` and
    `schema` are strings (and must each be less than 80 bytes). `schema_version`
    is a tuple listing the major and minor version numbers. In your code,
    replace `"example"` with the name of your application.

    # Example

    ```
    use hoomd_gsd::file_layer::{GsdFile, Mode};
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    # use tempfile::tempdir;
    # let tmp_dir = tempdir().expect("temp dir should be created");
    # let path = tmp_dir.path().join("test.gsd");
    let gsd_file = GsdFile::create(path, "example", "hoomd", (1, 4))?;
    # Ok(())
    # }
    ```

    # Errors

    Returns a [`OpenError`] when any of the following occur:
    * The file cannot be created.
    * The file is corrupt, unreadable, or there is an I/O error (see
      [`DecodeError`]).
    */
    #[inline]
    pub fn create<P: AsRef<Path>>(
        path: P,
        application: &str,
        schema: &str,
        schema_version: (u16, u16),
    ) -> Result<Self, OpenError> {
        let mut file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| OpenError::IO(path.as_ref().into(), e))?;
        GsdFile::initialize_file(&mut file, &path, application, schema, schema_version)?;
        GsdFile::from_file(file, Mode::Write)
            .map_err(|e| OpenError::Decode(path.as_ref().into(), e))
    }

    /** Create a new GSD file.

    Creates a new GSD file at the given path, returning an error when the
    path already exists. When successful, return a [`GsdFile`] opened in
    write mode.

    Each GSD file contains metadata describing which application created the
    file, the data chunk schema, and the schema's version. `application` and
    `schema` are strings (and must each be less than 80 bytes). `schema_version`
    is a tuple listing the major and minor version numbers. In your code,
    replace `"example"` with the name of your application.

    # Example

    ```
    use hoomd_gsd::file_layer::{GsdFile, Mode};
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    # use tempfile::tempdir;
    # let tmp_dir = tempdir().expect("temp dir should be created");
    # let path = tmp_dir.path().join("test.gsd");
    let gsd_file = GsdFile::create_new(path, "example", "hoomd", (1, 4))?;
    # Ok(())
    # }
    ```

    # Errors

    Returns a [`OpenError`] when any of the following occur:
    * The file cannot be created.
    * The file already exists.
    * The file is corrupt, unreadable, or there is an I/O error (see
      [`DecodeError`]).
    */
    #[inline]
    pub fn create_new<P: AsRef<Path>>(
        path: P,
        application: &str,
        schema: &str,
        schema_version: (u16, u16),
    ) -> Result<Self, OpenError> {
        let mut file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| OpenError::IO(path.as_ref().into(), e))?;
        GsdFile::initialize_file(&mut file, &path, application, schema, schema_version)?;
        GsdFile::from_file(file, Mode::Write)
            .map_err(|e| OpenError::Decode(path.as_ref().into(), e))
    }

    /// Initialize an empty file.
    fn initialize_file<P: AsRef<Path>>(
        file: &mut File,
        path: &P,
        application: &str,
        schema: &str,
        schema_version: (u16, u16),
    ) -> Result<(), OpenError> {
        let application = String::from(application);
        if application.len() as u64 > NAME_SIZE - 1 {
            return Err(OpenError::NameTooLong(application));
        }
        let schema = String::from(schema);
        if schema.len() as u64 > NAME_SIZE - 1 {
            return Err(OpenError::NameTooLong(schema));
        }

        let header = GsdHeader {
            magic: MAGIC_ID,
            gsd_version: CURRENT_FILE_VERSION,
            application,
            schema,
            schema_version,
            index_location: HEADER_SIZE,
            index_allocated_entries: INITIAL_INDEX_SIZE,
            namelist_location: HEADER_SIZE + INDEX_ENTRY_SIZE * INITIAL_INDEX_SIZE,
            namelist_allocated_entries: INITIAL_NAME_LIST_SIZE / NAME_SIZE,
        };

        file.write_all(&header.to_ne_bytes())
            .map_err(|e| OpenError::IO(path.as_ref().into(), e))?;

        file.set_len(HEADER_SIZE + INDEX_ENTRY_SIZE * INITIAL_INDEX_SIZE + INITIAL_NAME_LIST_SIZE)
            .map_err(|e| OpenError::IO(path.as_ref().into(), e))?;

        file.sync_all()
            .map_err(|e| OpenError::IO(path.as_ref().into(), e))?;

        Ok(())
    }

    /// Populate the fields in `GsdFile` given an open `File`.
    fn from_file(file: File, mode: Mode) -> Result<GsdFile, DecodeError> {
        let mut file = file;
        file.rewind()?;

        let mut header_bytes = [0_u8; HEADER_USIZE];
        file.read_exact(&mut header_bytes)?;
        let header = GsdHeader::try_from_ne_bytes(header_bytes)?;

        let file_len = file.seek(SeekFrom::End(0))?;
        // Verify that the entire file is addressable in the mmap. This makes
        // the usize::try_from checks in get_index will not fail.
        usize::try_from(file_len).map_err(DecodeError::UnaddressableContent)?;

        // Provide the caller with helpful errors when the code would otherwise
        // access the memory map outside the contents of the file.
        if header.index_location > file_len
            || header.index_location + header.index_allocated_entries * INDEX_ENTRY_SIZE > file_len
            || header.index_allocated_entries == 0
        {
            return Err(DecodeError::IndexOutOfBounds(
                header.index_location,
                header.index_allocated_entries * INDEX_ENTRY_SIZE,
            ));
        }
        let namelist_range_end =
            header.namelist_location + header.namelist_allocated_entries * NAME_SIZE;
        if namelist_range_end > file_len || header.namelist_allocated_entries == 0 {
            return Err(DecodeError::NameListOutOfBounds(
                header.namelist_location,
                header.namelist_allocated_entries * NAME_SIZE,
            ));
        }

        let mmap = unsafe { Mmap::map(&file)? };
        let last_namelist_offset =
            usize::try_from(namelist_range_end - 1).map_err(DecodeError::UnaddressableIndex)?;
        if mmap[last_namelist_offset] != 0 {
            return Err(DecodeError::NameListNotTerminated);
        }

        let start =
            usize::try_from(header.namelist_location).map_err(DecodeError::UnaddressableIndex)?;
        let end = usize::try_from(namelist_range_end).map_err(DecodeError::UnaddressableIndex)?;
        let name_list = GsdFile::decode_name_map(&mmap[start..end])?;
        let index = Index {
            n: 0,
            buffer: Vec::new(),
            byte_buffer: Vec::new(),
            pending: 0,
            frame_names: HashSet::new(),
        };

        let mut gsd_file = GsdFile {
            file,
            mode,
            header,
            mmap,
            file_len,
            name_list,
            index,
            data_buffer: Vec::new(),
            data_buffer_flushed: false,
            buffer_frame: 0,
            file_frame: 0,
            maximum_write_buffer_size: INITIAL_MAXIMUM_WRITE_BUFFER_SIZE,
        };

        gsd_file.index.n = gsd_file.count_index_entries()?;
        if gsd_file.index.n > 0 {
            let last_entry = gsd_file.get_index(gsd_file.index.n - 1)?;
            gsd_file.file_frame = last_entry.frame + 1;
            gsd_file.buffer_frame = gsd_file.file_frame;
        }

        // Silently upgrade writable files from a previous matching major version to the latest
        // minor version.
        if gsd_file.mode == Mode::Write
            && gsd_file.header.gsd_version.0 == CURRENT_FILE_VERSION.0
            && gsd_file.header.gsd_version.1 < CURRENT_FILE_VERSION.1
        {
            gsd_file.header.gsd_version.1 = CURRENT_FILE_VERSION.1;

            gsd_file.file.seek(SeekFrom::Start(0))?;
            gsd_file.file.write_all(&gsd_file.header.to_ne_bytes())?;
        }

        Ok(gsd_file)
    }

    /// Read the initial name map from the file.
    fn decode_name_map(bytes: &[u8]) -> Result<NameList, DecodeError> {
        let mut name_id = HashMap::new();
        let mut bytes = bytes;

        let mut current_id: u16 = 0;
        let mut insert_position: u64 = 0;
        while !bytes.is_empty() && bytes[0] != 0 {
            let (name, rest) =
                extract_null_terminated_utf8(bytes).map_err(DecodeError::InvalidChunkName)?;
            bytes = rest;

            // The GSD spec ensures that all names in the map are always terminated.
            insert_position += (name.len() + 1) as u64;

            let previous = name_id.insert(name, current_id);
            if previous.is_some() {
                return Err(DecodeError::DuplicateChunkName);
            }

            if current_id == u16::MAX {
                return Err(DecodeError::NameListOverflow);
            }

            current_id += 1;
        }

        Ok(NameList {
            name_id,
            n_names: current_id,
            insert_position,
            buffer: Vec::new(),
        })
    }

    /// Get the `id` of a name. Add a new `id` if needed.
    #[inline]
    fn get_id(&mut self, name: &str) -> Result<u16, EncodeError> {
        if let Some(id) = self.name_list.name_id.get(name) {
            return Ok(*id);
        }

        let new_id = self.name_list.n_names;
        if new_id == u16::MAX {
            return Err(EncodeError::NameListOverflow);
        }

        self.name_list.n_names += 1;
        self.name_list.buffer.extend(name.as_bytes());
        self.name_list.buffer.push(0);
        self.name_list.name_id.insert(String::from(name), new_id);
        Ok(new_id)
    }

    /// Remap the file
    #[inline]
    #[cfg(target_os = "linux")]
    fn remap(&mut self) -> Result<(), io::Error> {
        unsafe {
            self.mmap.remap(
                self.file_len
                    .try_into()
                    .expect("file length should be validated elsewhere"),
                memmap2::RemapOptions::new().may_move(true),
            )?;
        }
        Ok(())
    }

    /// Remap the file
    #[inline]
    #[cfg(not(target_os = "linux"))]
    fn remap(&mut self) -> Result<(), io::Error> {
        self.mmap = unsafe { Mmap::map(&self.file)? };
        Ok(())
    }

    /// Access a single index entry from the memory map.
    #[inline]
    fn get_index(&self, i: u64) -> Result<IndexEntry, DecodeError> {
        // get_index is an internal method, assume that any caller has already
        // called remap() if needed. Verify this in debug builds.
        debug_assert!(self.mmap.len() as u64 == self.file_len);

        let start = self.header.index_location + i * INDEX_ENTRY_SIZE;
        let end = start + INDEX_ENTRY_SIZE;
        debug_assert!(
            end <= self.header.index_location
                + self.header.index_allocated_entries * INDEX_ENTRY_SIZE
        );

        let start = usize::try_from(start).map_err(DecodeError::UnaddressableIndex)?;
        let end = usize::try_from(end).map_err(DecodeError::UnaddressableIndex)?;
        let bytes: [u8; INDEX_ENTRY_USIZE] = self.mmap[start..end]
            .try_into()
            .expect("slice should always be the correct size");
        Ok(IndexEntry::from_ne_bytes(bytes))
    }

    /// Get the size of a type given by its identifier.
    #[inline]
    fn size_of(data_type: u8) -> Option<usize> {
        match data_type {
            1 => Some(size_of::<u8>()),
            2 => Some(size_of::<u16>()),
            3 => Some(size_of::<u32>()),
            4 => Some(size_of::<u64>()),
            5 => Some(size_of::<i8>()),
            6 => Some(size_of::<i16>()),
            7 => Some(size_of::<i32>()),
            8 => Some(size_of::<i64>()),
            9 => Some(size_of::<f32>()),
            10 => Some(size_of::<f64>()),
            11 => Some(1),
            _ => None,
        }
    }

    /// Test if an index entry is valid in the context of the file.
    fn is_entry_valid(&self, entry: &IndexEntry) -> bool {
        match GsdFile::size_of(entry.data_type) {
            Some(element_size) => {
                let total_size = entry.n * u64::from(entry.m) * element_size as u64;
                assert!(entry.location + total_size <= self.file_len);
                if entry.location + total_size > self.file_len {
                    return false;
                }
            }
            None => return false,
        }

        // is_entry_valid is used before the file is fully loaded and the number
        // of frames is not yet known. Check that the frame is at least within
        // the number of allocated index entries.
        assert!(entry.frame < self.header.index_allocated_entries);
        if entry.frame >= self.header.index_allocated_entries {
            return false;
        }

        // There is no need to include buffered names here because
        // is_entry_valid is only called on file open, not after any write_
        // methods.
        assert!(entry.id < self.name_list.n_names);
        if entry.id >= self.name_list.n_names {
            return false;
        }

        if entry.flags != 0 {
            return false;
        }

        true
    }

    /// Determine the number of frames in the file.
    fn count_index_entries(&self) -> Result<u64, DecodeError> {
        let first_entry = self.get_index(0)?;
        if first_entry.location != 0 && !self.is_entry_valid(&first_entry) {
            return Err(DecodeError::CorruptIndexEntry(first_entry));
        }

        if first_entry.location == 0 {
            return Ok(0);
        }

        // determine the number of index entries (marked by location = 0)
        // binary search for the first index entry with location 0
        let mut l: u64 = 0;
        let mut r = self.header.index_allocated_entries;

        // progressively narrow the search window by halves
        loop {
            let m = l.midpoint(r);

            // file is corrupt if any index entry is invalid or frame does not increase
            // monotonically
            let entry_m = self.get_index(m)?;
            let entry_l = self.get_index(l)?;

            if entry_m.location != 0 {
                if !self.is_entry_valid(&entry_m) || entry_m.frame < entry_l.frame {
                    return Err(DecodeError::CorruptIndexEntry(entry_m));
                }
                l = m;
            } else {
                r = m;
            }

            if r - l == 1 {
                break;
            }
        }

        // this finds R = the first index entry with location = 0
        Ok(r)
    }

    /** Find a chunk in the index.

    Returns [`Some(index_entry)`](Option::Some) when the data chunk is present
    in the file and [`None`] when it is not.

    # Example

    ```
    use hoomd_gsd::file_layer::{DataType, GsdFile};
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    # use tempfile::tempdir;
    # let tmp_dir = tempdir().expect("temp dir should be created");
    # let path = tmp_dir.path().join("test.gsd");
    let mut gsd_file = GsdFile::create_new(path, "example", "hoomd", (1, 4))?;
    gsd_file.write_scalars("configuration/step", &[100_000u64])?;
    gsd_file.end_frame()?;
    gsd_file.sync_all()?;

    let configuration_box = gsd_file.find_chunk(0, "configuration/box");
    assert_eq!(configuration_box, None);

    let step = gsd_file.find_chunk(0, "configuration/step");
    assert!(step.is_some());
    if let Some(index) = step {
        assert_eq!(index.frame(), 0);
        assert_eq!(index.rows(), 1);
        assert_eq!(index.columns(), 1);
        assert_eq!(index.data_type(), Some(DataType::U64));
    }
    # Ok(())
    # }
    ```
    */
    #[must_use]
    pub fn find_chunk(&self, frame: u64, name: &str) -> Option<IndexEntry> {
        if frame >= self.file_frame || self.index.n == 0 {
            return None;
        }

        let id = match self.name_list.name_id.get(name) {
            None => return None,
            Some(id) => *id,
        };

        // binary search for the index entry
        let mut l: u64 = 0;
        let mut r = self.index.n - 1;

        while l <= r {
            let m = l.midpoint(r);

            // We can map an error to None here because the unaddressable index error
            // would have previously been caught on open or sync.
            if let Ok(index_entry_m) = self.get_index(m) {
                match (index_entry_m.frame, index_entry_m.id).cmp(&(frame, id)) {
                    Ordering::Less => l = m + 1,
                    Ordering::Greater => r = m - 1,
                    Ordering::Equal => return Some(index_entry_m),
                }
            } else {
                return None;
            }
        }
        None
    }

    /** Iterate over an array of scalars in the given frame.

    Returns [`Ok(iterator)`](Result::Ok) when the data chunk is present in the
    file and `Err(`[`ReadError::ChunkNotFound`]`)` when it is not. Collect the
    iterator into a [`Vec`] to make a copy of the data, or process the data in
    place while iterating.

    Data written to a file is not available for reading until the file
    is closed or after a call to [`sync_all`](GsdFile::sync_all).

    # Example

    ```
    use hoomd_gsd::file_layer::{DataType, GsdFile};
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    # use tempfile::tempdir;
    # let tmp_dir = tempdir().expect("temp dir should be created");
    # let path = tmp_dir.path().join("test.gsd");
    let mut gsd_file = GsdFile::create_new(path, "example", "hoomd", (1, 4))?;
    gsd_file.write_scalars("configuration/box", &[10.0f32, 20.0, 15.0, 0.0, 0.0, 0.0])?;
    gsd_file.end_frame()?;
    gsd_file.sync_all()?;

    let box_iter = gsd_file.iter_scalars::<f32>(0, "configuration/box")?;
    let box_vec = box_iter.collect::<Vec<_>>();
    assert_eq!(box_vec, vec![10.0f32, 20.0, 15.0, 0.0, 0.0, 0.0]);
    # Ok(())
    # }
    ```

    # Errors

    Returns a [`ReadError`] when any of the following occur:
    * A chunk by the given `name` is not present in the given `frame`.
    * The data type stored in the file does not match `T`.
    * The array stored in the file does not have dimensions `N x 1`.
    * The file is corrupt, unreadable, or there is an I/O error (see
      [`DecodeError`]).
    */
    pub fn iter_scalars<T: Type>(
        &self,
        frame: u64,
        name: &str,
    ) -> Result<impl ExactSizeIterator<Item = T> + use<'_, T>, ReadError> {
        let index_entry = match self.find_chunk(frame, name) {
            None => return Err(ReadError::ChunkNotFound(name.into(), frame)),
            Some(e) => e,
        };

        if index_entry.m as usize != 1 {
            return Err(ReadError::Decode(
                name.into(),
                frame,
                DecodeError::InvalidColumns(1, index_entry.m),
            ));
        }

        if index_entry.data_type != T::gsd_data_type() {
            return Err(ReadError::Decode(
                name.into(),
                frame,
                DecodeError::InvalidType(T::gsd_data_type(), index_entry.data_type),
            ));
        }

        self.read_details(&index_entry)
            .map_err(|e| ReadError::Decode(name.into(), frame, e))
    }

    /** Iterate over an array of arrays in the given frame.

    Returns [`Ok(iterator)`](Result::Ok) when the data chunk is present in the
    file and `Err(`[`ReadError::ChunkNotFound`]`)` when it is not. Collect the
    iterator into a [`Vec`] to make a copy of the data, or process the data in
    place while iterating.

    Data written to a file is not available for reading until the file
    is closed or after a call to [`sync_all`](GsdFile::sync_all).

    # Example

    ```
    use hoomd_gsd::file_layer::{DataType, GsdFile};
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    # use tempfile::tempdir;
    # let tmp_dir = tempdir().expect("temp dir should be created");
    # let path = tmp_dir.path().join("test.gsd");
    let position = vec![[5.0f32, 3.0, -4.0], [-2.0, 3.0, -6.0]];

    let mut gsd_file = GsdFile::create_new(path, "example", "hoomd", (1, 4))?;
    gsd_file.write_arrays("particles/position", &position)?;
    gsd_file.end_frame()?;
    gsd_file.sync_all()?;

    let position_iter = gsd_file.iter_arrays::<f32, 3>(0, "particles/position")?;
    let position_vec = position_iter.collect::<Vec<_>>();
    assert_eq!(position_vec, position);
    # Ok(())
    # }
    ```

    # Errors

    Returns a [`ReadError`] when any of the following occur:
    * A chunk by the given `name` is not present in the given `frame`.
    * The data type stored in the file does not match `T`.
    * The array stored in the file does not have dimensions `N x M`.
    * The file is corrupt, unreadable, or there is an I/O error (see
      [`DecodeError`]).
    */
    pub fn iter_arrays<T: Type, const M: usize>(
        &self,
        frame: u64,
        name: &str,
    ) -> Result<impl ExactSizeIterator<Item = [T; M]> + use<'_, T, M>, ReadError> {
        let index_entry = match self.find_chunk(frame, name) {
            None => return Err(ReadError::ChunkNotFound(name.into(), frame)),
            Some(e) => e,
        };

        if index_entry.m as usize != M {
            return Err(ReadError::Decode(
                name.into(),
                frame,
                DecodeError::InvalidColumns(M, index_entry.m),
            ));
        }

        if index_entry.data_type != T::gsd_data_type() {
            return Err(ReadError::Decode(
                name.into(),
                frame,
                DecodeError::InvalidType(T::gsd_data_type(), index_entry.data_type),
            ));
        }

        Ok(ArrayChunks {
            iter: self
                .read_details::<T>(&index_entry)
                .map_err(|e| ReadError::Decode(name.into(), frame, e))?,
        })
    }

    /** Read a string in the given frame.

    Returns [`Ok(String)`](Result::Ok) when the data chunk is present
    in the file and `Err(`[`ReadError::ChunkNotFound`]`)` when it is not.

    Data written to a file is not available for reading until the file
    is closed or after a call to [`sync_all`](GsdFile::sync_all).

    # Example

    ```
    use hoomd_gsd::file_layer::{DataType, GsdFile};
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    # use tempfile::tempdir;
    # let tmp_dir = tempdir().expect("temp dir should be created");
    # let path = tmp_dir.path().join("test.gsd");
    let mut gsd_file = GsdFile::create_new(path, "example", "hoomd", (1, 4))?;
    gsd_file.write_string("log/my_string", "Hello, GSD.")?;
    gsd_file.end_frame()?;
    gsd_file.sync_all()?;

    let string = gsd_file.read_string(0, "log/my_string")?;
    assert_eq!(string, "Hello, GSD.");
    # Ok(())
    # }
    ```

    # Errors

    Returns a [`ReadError`] when any of the following occur:
    * A chunk by the given `name` is not present in the given `frame`.
    * The data type stored in the file is not a UTF-8 string.
    * The array stored in the file does not have dimensions `N x 1`.
    * The file is corrupt, unreadable, or there is an I/O error (see
      [`DecodeError`]).
    */
    pub fn read_string(&self, frame: u64, name: &str) -> Result<String, ReadError> {
        let index_entry = match self.find_chunk(frame, name) {
            None => return Err(ReadError::ChunkNotFound(name.into(), frame)),
            Some(e) => e,
        };

        if index_entry.m as usize != 1 {
            return Err(ReadError::Decode(
                name.into(),
                frame,
                DecodeError::InvalidColumns(1, index_entry.m),
            ));
        }

        if index_entry.data_type != 11 {
            return Err(ReadError::Decode(
                name.into(),
                frame,
                DecodeError::InvalidType(11, index_entry.data_type),
            ));
        }

        let array = self
            .read_details::<u8>(&index_entry)
            .map_err(|e| ReadError::Decode(name.into(), frame, e))?
            .collect::<Vec<_>>();

        String::from_utf8(array)
            .map_err(|e| ReadError::Decode(name.into(), frame, DecodeError::InvalidString(e)))
    }

    /// Implement the details of `iter_scalars` and `iter_arrays`.
    fn read_details<T: Type>(
        &self,
        index_entry: &IndexEntry,
    ) -> Result<impl ExactSizeIterator<Item = T> + use<'_, T>, DecodeError> {
        let n_elements = index_entry.n * u64::from(index_entry.m);
        let n_bytes = usize::try_from(n_elements * size_of::<T>() as u64)
            .map_err(DecodeError::UnaddressableContent)?;

        let location =
            usize::try_from(index_entry.location).map_err(DecodeError::UnaddressableContent)?;

        if index_entry.location == 0 {
            return Err(DecodeError::CorruptIndexEntry(*index_entry));
        }

        debug_assert!(location + n_bytes <= self.mmap.len());

        Ok(self.mmap[location..location + n_bytes]
            .chunks(size_of::<T>())
            .map(T::from_ne_byte_slice))
    }

    /** Append an array of scalar values to the current frame.

    `write_scalars` writes one-dimensional array data to a named chunk in the
    current frame of the GSD file. Call [`end_frame`](GsdFile::end_frame) to
    complete the frame and start the next.

    <div class="warning">

    Dropping a [`GsdFile`] will also drop any pending data chunks in incomplete
    frames.

    </div>

    # Example

    ```
    use hoomd_gsd::file_layer::{DataType, GsdFile};
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    # use tempfile::tempdir;
    # let tmp_dir = tempdir().expect("temp dir should be created");
    # let path = tmp_dir.path().join("test.gsd");
    let mut gsd_file = GsdFile::create_new(path, "example", "hoomd", (1, 4))?;
    gsd_file.write_scalars("configuration/box", &[10.0f32, 20.0, 15.0, 0.0, 0.0, 0.0])?;
    gsd_file.end_frame()?;
    # Ok(())
    # }
    ```

    # Errors

    Returns a [`WriteError`] when any of the following occur:
    * The file is not opened in a write mode.
    * There are no available chunk identifiers.
    * A chunk with the same name has already been written in this frame.
    * There is an I/O error while writing to the file.
    */
    pub fn write_scalars<'a, T, I>(&mut self, name: &str, data: I) -> Result<(), WriteError>
    where
        T: Type + 'a,
        I: IntoIterator<Item = &'a T>,
        I::IntoIter: ExactSizeIterator,
    {
        // This is required for the function to accept arguments types such as
        // &Vec<T>: https://github.com/rust-lang/rust/issues/77214
        let data = data.into_iter();

        self.write_details(
            name,
            data.len() as u64,
            1,
            T::gsd_data_type(),
            |buffer: &mut Vec<u8>| {
                for value in data {
                    value.append_ne_bytes(buffer);
                }
            },
        )
        .map_err(|e| WriteError::Encode(name.into(), self.buffer_frame, e))
    }

    /** Append an array of array values to the current frame.

    `write_arrays` writes two-dimensional array data to a named chunk in the
    current frame of the GSD file. Call [`end_frame`](GsdFile::end_frame) to
    complete the frame and start the next.

    <div class="warning">

    Dropping a [`GsdFile`] will also drop any pending data chunks in incomplete
    frames.

    </div>

    # Example

    ```
    use hoomd_gsd::file_layer::{DataType, GsdFile};
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    # use tempfile::tempdir;
    # let tmp_dir = tempdir().expect("temp dir should be created");
    # let path = tmp_dir.path().join("test.gsd");
    let position = vec![[5.0f32, 3.0, -4.0], [-2.0, 3.0, -6.0]];

    let mut gsd_file = GsdFile::create_new(path, "example", "hoomd", (1, 4))?;
    gsd_file.write_arrays("particles/position", &position)?;
    gsd_file.end_frame()?;
    # Ok(())
    # }
    ```

    # Errors

    Returns a [`WriteError`] when any of the following occur:
    * The file is not opened in a write mode.
    * There are no available chunk identifiers.
    * A chunk with the same name has already been written in this frame.
    * `M` is 0.
    * `M` cannot be represented by a `u32`.
    */
    pub fn write_arrays<'a, T, I, const M: usize>(
        &mut self,
        name: &str,
        data: I,
    ) -> Result<(), WriteError>
    where
        T: Type + 'a,
        I: IntoIterator<Item = &'a [T; M]>,
        I::IntoIter: ExactSizeIterator,
    {
        if M == 0 {
            return Err(WriteError::Encode(
                name.into(),
                self.buffer_frame,
                EncodeError::InvalidColumns(M),
            ));
        }

        let columns = u32::try_from(M).or(Err(WriteError::Encode(
            name.into(),
            self.buffer_frame,
            EncodeError::InvalidColumns(M),
        )))?;

        let data = data.into_iter();

        self.write_details(
            name,
            data.len() as u64,
            columns,
            T::gsd_data_type(),
            |buffer: &mut Vec<u8>| {
                for element in data {
                    for value in element {
                        value.append_ne_bytes(buffer);
                    }
                }
            },
        )
        .map_err(|e| WriteError::Encode(name.into(), self.buffer_frame, e))
    }

    /** Append a string to the current frame.

    `write_string` writes a UTF-8 string to a named chunk in the current frame
    of the GSD file. Call [`end_frame`](GsdFile::end_frame) to complete the
    frame and start the next.

    <div class="warning">

    Dropping a [`GsdFile`] will also drop any pending data chunks in incomplete
    frames.

    </div>

    # Example

    ```
    use hoomd_gsd::file_layer::{DataType, GsdFile};
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    # use tempfile::tempdir;
    # let tmp_dir = tempdir().expect("temp dir should be created");
    # let path = tmp_dir.path().join("test.gsd");
    let mut gsd_file = GsdFile::create_new(path, "example", "hoomd", (1, 4))?;
    gsd_file.write_string("log/my_string", "Hello, GSD.")?;
    gsd_file.end_frame()?;
    # Ok(())
    # }
    ```

    # Errors

    Returns a [`WriteError`] when any of the following occur:
    * The file is not opened in a write mode.
    * There are no available chunk identifiers.
    * A chunk with the same name has already been written in this frame.
    */
    pub fn write_string(&mut self, name: &str, data: &str) -> Result<(), WriteError> {
        let data = data.as_bytes();

        self.write_details(name, data.len() as u64, 1, 11, |buffer: &mut Vec<u8>| {
            buffer.extend(data);
        })
        .map_err(|e| WriteError::Encode(name.into(), self.buffer_frame, e))
    }

    /// Common code used in all write_ methods.
    fn write_details<F>(
        &mut self,
        name: &str,
        rows: u64,
        columns: u32,
        data_type: u8,
        append: F,
    ) -> Result<(), EncodeError>
    where
        F: FnOnce(&mut Vec<u8>),
    {
        if self.mode != Mode::Write {
            return Err(EncodeError::NotWritable);
        }

        let id = self.get_id(name)?;

        // write_scalars doesn't actually write any data to the file itself. For
        // performance, it buffers all writes. Above, `get_id` appended any
        // new names to `self.name_list.buffer`. Now, `write_scalars` needs to
        // construct the index entry and put the bytes of the array in the data
        // buffer. `sync_all` will write the data buffer first, so all index
        // entries can be constructed with the known location:
        // file_len + currently buffered bytes.
        let index_entry = IndexEntry {
            frame: self.buffer_frame,
            n: rows,
            m: columns,
            location: self.file_len + self.data_buffer.len() as u64,
            id,
            data_type,
            flags: 0,
        };

        if !self.index.frame_names.insert(index_entry.id) {
            return Err(EncodeError::DuplicateChunkName(
                name.into(),
                self.buffer_frame,
            ));
        }

        self.index.buffer.push(index_entry);
        self.index.pending += 1;

        // This implementation is a departure from the GSD C implementation
        // which would eagerly write large arrays directly to the file before
        // flushing the previous entries. That complicated the code and
        // required two index buffers that needed to be patched up.
        // This implementation always appends data to the write buffer
        // (via the append call).
        //
        // The Rust implementation always writes full data chunks into the
        // buffer, but flushes the buffer first in, first out. That way, no
        // index entries need to be patched up. When the buffer is flushed here,
        // we do need to flag to `end_frame` that `sync_all` needs to be called.
        append(&mut self.data_buffer);

        if self.data_buffer.len() >= self.maximum_write_buffer_size {
            self.flush_data()?;
            self.data_buffer_flushed = true;
        }

        Ok(())
    }

    /** Complete the current frame.

    Commits previous calls to `write_*` methods to the current frame. Calls to
    `write_*` methods following `end_frame` will write to the next frame.

    Calling `end_frame` does **not** ensure that all buffered data is synced to
    the filesystem. Call [`sync_all`](GsdFile::sync_all) to do so.

    # Example

    ```
    use hoomd_gsd::file_layer::{DataType, GsdFile};
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    # use tempfile::tempdir;
    # let tmp_dir = tempdir().expect("temp dir should be created");
    # let path = tmp_dir.path().join("test.gsd");
    let mut gsd_file = GsdFile::create_new(path, "example", "hoomd", (1, 4))?;
    gsd_file.write_scalars("configuration/step", &[100_000u64])?;
    gsd_file.end_frame()?;

    gsd_file.write_scalars("configuration/step", &[200_000u64])?;
    gsd_file.end_frame()?;

    gsd_file.write_scalars("configuration/step", &[300_000u64])?;
    gsd_file.end_frame()?;
    # Ok(())
    # }
    ```

    # Errors

    Returns a [`EncodeError`] when any of the following occur:
    * The file is not opened in a write mode.
    */
    pub fn end_frame(&mut self) -> Result<(), EncodeError> {
        if self.mode != Mode::Write {
            return Err(EncodeError::NotWritable);
        }

        self.buffer_frame += 1;
        self.index.pending = 0;
        self.index.frame_names.clear();

        Ok(())
    }

    #[inline]
    #[must_use]
    /** The number of frames *written to the file*.

    `n_frames` returns the number of frames *available to read* from the file.
    Each call to [`end_frame`](GsdFile::end_frame) increments the number of
    frames, but they are not written to the file until it is closed or by a call
    to [`sync_all`](GsdFile::sync_all).

    ```
    use hoomd_gsd::file_layer::{DataType, GsdFile};
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    # use tempfile::tempdir;
    # let tmp_dir = tempdir().expect("temp dir should be created");
    # let path = tmp_dir.path().join("test.gsd");
    let mut gsd_file = GsdFile::create_new(path, "example", "hoomd", (1, 4))?;
    gsd_file.write_scalars("configuration/step", &[100_000u64])?;
    gsd_file.end_frame()?;

    gsd_file.write_scalars("configuration/step", &[200_000u64])?;
    gsd_file.end_frame()?;

    gsd_file.write_scalars("configuration/step", &[300_000u64])?;
    gsd_file.end_frame()?;
    gsd_file.sync_all()?;

    let n_frames = gsd_file.n_frames();
    assert_eq!(n_frames, 3);
    # Ok(())
    # }
    ```
    */
    pub fn n_frames(&self) -> u64 {
        self.file_frame
    }

    #[inline]
    #[must_use]
    /** Provide the mapping from data chunk names to ids.

    The mapping includes keys for all data chunk names in the file.

    # Example

    ```
    use hoomd_gsd::file_layer::GsdFile;
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    # use tempfile::tempdir;
    # let tmp_dir = tempdir().expect("temp dir should be created");
    # let path = tmp_dir.path().join("test.gsd");
    let position = vec![[5.0f32, 3.0, -4.0], [-2.0, 3.0, -6.0]];

    let mut gsd_file = GsdFile::create_new(path, "example", "hoomd", (1, 4))?;
    gsd_file.write_scalars("configuration/step", &[100_000u64])?;
    gsd_file.write_scalars("configuration/box", &[10.0f32, 20.0, 15.0, 0.0, 0.0, 0.0])?;
    gsd_file.write_arrays("particles/position", &position)?;
    gsd_file.end_frame()?;

    let name_id = gsd_file.name_id();
    assert!(name_id.contains_key("configuration/step"));
    assert!(name_id.contains_key("configuration/box"));
    assert!(name_id.contains_key("particles/position"));
    assert!(!name_id.contains_key("particles/orientation"));
    # Ok(())
    # }
    ```
    */
    pub fn name_id(&self) -> &HashMap<String, u16> {
        &self.name_list.name_id
    }

    /** The name of the application used to write the file.

    # Example

    ```
    use hoomd_gsd::file_layer::GsdFile;
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    # use tempfile::tempdir;
    # let tmp_dir = tempdir().expect("temp dir should be created");
    # let path = tmp_dir.path().join("test.gsd");
    let mut gsd_file = GsdFile::create_new(path, "example", "hoomd", (1, 4))?;

    let application = gsd_file.application();
    assert_eq!(application, "example");
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn application(&self) -> &str {
        &self.header.application
    }

    /** The schema that describes the expected data chunks.

    # Example

    ```
    use hoomd_gsd::file_layer::GsdFile;
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    # use tempfile::tempdir;
    # let tmp_dir = tempdir().expect("temp dir should be created");
    # let path = tmp_dir.path().join("test.gsd");
    let mut gsd_file = GsdFile::create_new(path, "example", "hoomd", (1, 4))?;

    let schema = gsd_file.schema();
    assert_eq!(schema, "hoomd");
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn schema(&self) -> &str {
        &self.header.schema
    }

    /** The schema version (major, minor).

    # Example

    ```
    use hoomd_gsd::file_layer::GsdFile;
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    # use tempfile::tempdir;
    # let tmp_dir = tempdir().expect("temp dir should be created");
    # let path = tmp_dir.path().join("test.gsd");
    let mut gsd_file = GsdFile::create_new(path, "example", "hoomd", (1, 4))?;

    let schema_version = gsd_file.schema_version();
    assert_eq!(schema_version, (1,4));
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn schema_version(&self) -> (u16, u16) {
        self.header.schema_version
    }

    #[inline]
    #[must_use]
    /// The maximum number of bytes to store in the write buffer.
    pub fn maximum_write_buffer_size(&self) -> usize {
        self.maximum_write_buffer_size
    }

    #[inline]
    #[must_use]
    /// Mutable reference to the maximum number of bytes to store in the write buffer.
    pub fn maximum_write_buffer_size_mut(&mut self) -> &mut usize {
        &mut self.maximum_write_buffer_size
    }

    /** Flush data buffer to the filesystem.

    Returns true when any data was written to the file.
    */
    fn flush_data(&mut self) -> Result<bool, EncodeError> {
        if self.data_buffer.is_empty() {
            Ok(false)
        } else {
            let current_len = self.file.seek(SeekFrom::End(0))?;
            debug_assert_eq!(current_len, self.file_len);
            self.file.write_all(&self.data_buffer)?;
            self.file_len += self.data_buffer.len() as u64;
            self.data_buffer.clear();
            Ok(true)
        }
    }

    /** Flush the name buffer to the filesystem.

    Returns true when any data was written to the file.
    */
    fn flush_names(&mut self) -> Result<bool, EncodeError> {
        if self.name_list.buffer.is_empty() {
            Ok(false)
        } else {
            if self.name_list.insert_position + self.name_list.buffer.len() as u64
                > self.header.namelist_allocated_entries * NAME_SIZE
            {
                self.expand_name_list_to(
                    self.name_list.insert_position + self.name_list.buffer.len() as u64,
                )?;
            }
            debug_assert!(
                self.name_list.insert_position + self.name_list.buffer.len() as u64
                    <= self.header.namelist_allocated_entries * NAME_SIZE
            );
            self.file.seek(SeekFrom::Start(
                self.header.namelist_location + self.name_list.insert_position,
            ))?;
            self.file.write_all(&self.name_list.buffer)?;
            self.name_list.insert_position += self.name_list.buffer.len() as u64;
            self.name_list.buffer.clear();
            Ok(true)
        }
    }

    /** Write buffered data to the filesystem.

    `sync_all` ensures that the data and indices for all complete frames is
    written to the filesystem.

    In most cases, callers should not call `sync_all` manually. It will be
    called automatically when a [`GsdFile`] is dropped. Call `sync_all` only
    when you need to read data arrays written in previous frames or when you
    want to ensure that all data up to a specific frame are present in the file.

    # Errors

    Returns a [`WriteError`] when any of the following occur:
    * The file is not opened in a write mode.
    * An I/O error writing to the file.
    */
    pub fn sync_all(&mut self) -> Result<(), EncodeError> {
        if self.mode != Mode::Write {
            return Err(EncodeError::NotWritable);
        }

        let mut need_remap = false;

        // Write the data buffer to the file first. Should any error occur here,
        // the file might have some extra bytes at the end, but the index of
        // written data so far will be correct.
        if self.flush_data()? || self.data_buffer_flushed {
            need_remap = true;
            self.data_buffer_flushed = false;
            self.file.sync_all()?;
        }

        // Write the new name next to ensure that the references in the index
        // will be consistent with the names.
        self.flush_names()?;

        // Now write all the non-pending index entries.
        // Index entries must be sorted by (frame, id) to be valid. Given that
        // pending index entries are guaranteed to have `frame+1`, we do not
        // need to sort the pending entries here.
        let index_entries_to_write = self.index.buffer.len() - self.index.pending;
        if index_entries_to_write > 0 {
            if self.index.n + index_entries_to_write as u64 > self.header.index_allocated_entries {
                need_remap = true;
                self.expand_index_to(
                    (self.index.n + index_entries_to_write as u64) * INDEX_ENTRY_SIZE,
                )?;
            }
            debug_assert!(
                self.index.n + index_entries_to_write as u64 <= self.header.index_allocated_entries
            );
            self.index.buffer[0..index_entries_to_write].sort_unstable();

            // format the index entries to write in the file byte order and
            // remove them from the index buffer.
            self.index.byte_buffer.clear();
            for entry in self.index.buffer.drain(0..index_entries_to_write) {
                self.index.byte_buffer.extend(&entry.to_ne_bytes());
            }
            self.file.seek(SeekFrom::Start(
                self.header.index_location + self.index.n * INDEX_ENTRY_SIZE,
            ))?;
            self.file.write_all(&self.index.byte_buffer)?;
            self.index.n += index_entries_to_write as u64;

            self.file.sync_all()?;
        }

        if need_remap {
            self.remap()?;
        }

        self.file_frame = self.buffer_frame;

        Ok(())
    }

    /// Expand the name list.
    fn expand_name_list_to(&mut self, capacity: u64) -> Result<(), EncodeError> {
        let old_size = self.header.namelist_allocated_entries * NAME_SIZE;
        let mut new_size = old_size;
        while new_size <= capacity {
            new_size *= 2;
        }

        // Ensure that the new buffer size is a multiple of NAME_SIZE because
        // GSD files always allocate name lists in those multiples.
        let new_allocated_entries = new_size.div_ceil(NAME_SIZE);
        let new_size = new_allocated_entries * NAME_SIZE;
        let new_location = self.file.seek(SeekFrom::End(0))?;

        usize::try_from(new_location)
            .map_err(|_| EncodeError::NameListOutOfBounds(new_location, new_size))?;
        usize::try_from(new_location + new_size)
            .map_err(|_| EncodeError::NameListOutOfBounds(new_location, new_size))?;

        let old_start = usize::try_from(self.header.namelist_location)
            .expect("namelist should be validated addressable previously");
        let old_end =
            usize::try_from(self.header.namelist_location + self.name_list.insert_position)
                .expect("namelist should be validated addressable previously");
        self.file.write_all(&self.mmap[old_start..old_end])?;
        self.file.set_len(new_location + new_size)?;
        self.file_len = new_location + new_size;

        // Ensure that the new name list is in place before updating the
        // header. If one of the writes fails, the file could otherwise
        // be left in a state where the header points to a non-existent
        // name list.
        self.file.sync_all()?;

        self.header.namelist_location = new_location;
        self.header.namelist_allocated_entries = new_allocated_entries;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&self.header.to_ne_bytes())?;

        self.file.sync_all()?;

        Ok(())
    }

    /// Expand the index.
    fn expand_index_to(&mut self, capacity: u64) -> Result<(), EncodeError> {
        let old_size = self.header.index_allocated_entries * INDEX_ENTRY_SIZE;
        let mut new_size = old_size;
        while new_size <= capacity {
            new_size *= 2;
        }

        // Ensure that the new buffer size is a multiple of INDEX_ENTRY_SIZE
        // because GSD files always allocate indices in those multiples.
        let new_allocated_entries = new_size.div_ceil(INDEX_ENTRY_SIZE);
        let new_size = new_allocated_entries * INDEX_ENTRY_SIZE;
        let new_location = self.file.seek(SeekFrom::End(0))?;

        usize::try_from(new_location)
            .map_err(|_| EncodeError::IndexOutOfBounds(new_location, new_size))?;
        usize::try_from(new_location + new_size)
            .map_err(|_| EncodeError::IndexOutOfBounds(new_location, new_size))?;

        let old_start = usize::try_from(self.header.index_location)
            .expect("index should be validated addressable previously");
        let old_end = usize::try_from(self.header.index_location + self.index.n * INDEX_ENTRY_SIZE)
            .expect("index should be validated addressable previously");
        if old_end > self.mmap.len() {
            return Err(EncodeError::IndexOutOfBounds(
                old_start as u64,
                old_end as u64,
            ));
        }
        self.file.write_all(&self.mmap[old_start..old_end])?;
        self.file.set_len(new_location + new_size)?;
        self.file_len = new_location + new_size;

        // Ensure that the new index is in place before updating the
        // header. If one of the writes fails, the file could otherwise
        // be left in a state where the header points to a non-existent
        // index.
        self.file.sync_all()?;

        self.header.index_location = new_location;
        self.header.index_allocated_entries = new_allocated_entries;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&self.header.to_ne_bytes())?;

        self.file.sync_all()?;

        Ok(())
    }
}

/** Automatically synchronize buffered data before closing the file.

[`GsdFile`] automatically calls [`sync_all`](GsdFile::sync_all) when
dropped and ignores and errors. To check for any potential errors, call
[`sync_all`](GsdFile::sync_all) before dropping a [`GsdFile`].
*/
impl Drop for GsdFile {
    fn drop(&mut self) {
        let _ = self.sync_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn create_new() {
        let tmp_dir = tempdir().expect("temp dir should be created");
        let path = tmp_dir.path().join("test.gsd");
        GsdFile::create_new(path.clone(), "application", "schema", (12, 42))
            .expect("gsd file should be created");

        let gsd_file =
            GsdFile::open(path.clone(), Mode::Read).expect("test.gsd should be created above");
        assert_eq!(gsd_file.application(), "application");
        assert_eq!(gsd_file.schema(), "schema");
        assert_eq!(gsd_file.schema_version(), (12, 42));
        assert_eq!(gsd_file.n_frames(), 0);
        assert!(gsd_file.name_id().is_empty());

        assert!(matches!(
            GsdFile::create_new(path.clone(), "application", "schema", (12, 42)),
            Err(OpenError::IO(_, _))
        ));
    }

    #[test]
    fn create_errors() {
        let tmp_dir = tempdir().expect("temp dir should be created");
        let path = tmp_dir.path().join("test.gsd");
        GsdFile::create(path.clone(), "application", "schema", (12, 42))
            .expect("gsd file should be created");

        let long_application = "a".repeat(64);
        let result = GsdFile::create(path.clone(), &long_application, "schema", (1, 0));
        assert!(matches!(result, Err(OpenError::NameTooLong(_))));

        let long_schema = "s".repeat(64);
        let result = GsdFile::create(path.clone(), "a", &long_schema, (1, 0));
        assert!(matches!(result, Err(OpenError::NameTooLong(_))));

        let just_right_application = "a".repeat(63);
        let just_right_schema = "s".repeat(63);
        let result = GsdFile::create(
            path.clone(),
            &just_right_application,
            &just_right_schema,
            (1, 0),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn maximum_write_buffer_size() {
        let tmp_dir = tempdir().expect("temp dir should be created");
        let path = tmp_dir.path().join("test.gsd");
        let mut gsd_file =
            GsdFile::create(path.clone(), "a", "s", (1, 0)).expect("gsd file should be created");

        *gsd_file.maximum_write_buffer_size_mut() = 8;
        assert_eq!(gsd_file.maximum_write_buffer_size(), 8);

        let initial_size = gsd_file
            .file
            .metadata()
            .expect("metadata should be valid")
            .len();
        assert_eq!(initial_size, gsd_file.file_len);

        gsd_file
            .write_scalars::<u64, _>("a", &[1])
            .expect("write should succeed");
        gsd_file.end_frame().expect("write should succeed");

        let final_size = gsd_file
            .file
            .metadata()
            .expect("metadata should be valid")
            .len();
        assert_eq!(final_size, gsd_file.file_len);
        assert_eq!(final_size, initial_size + 8);
    }

    #[test]
    fn sync_all() {
        let tmp_dir = tempdir().expect("temp dir should be created");
        let path = tmp_dir.path().join("test.gsd");
        let mut gsd_file =
            GsdFile::create(path.clone(), "a", "s", (1, 0)).expect("gsd file should be created");

        let initial_size = gsd_file
            .file
            .metadata()
            .expect("metadata should be valid")
            .len();

        gsd_file
            .write_scalars::<u64, _>("a", &[1])
            .expect("write should succeed");
        gsd_file.end_frame().expect("write should succeed");

        let final_size = gsd_file
            .file
            .metadata()
            .expect("metadata should be valid")
            .len();
        assert_eq!(final_size, gsd_file.file_len);
        assert_eq!(final_size, initial_size);

        gsd_file.sync_all().expect("write should succeed");
        let final_size = gsd_file
            .file
            .metadata()
            .expect("metadata should be valid")
            .len();
        assert_eq!(final_size, gsd_file.file_len);
        assert_eq!(final_size, initial_size + 8);
    }

    #[test]
    fn pending_index() {
        let tmp_dir = tempdir().expect("temp dir should be created");
        let path = tmp_dir.path().join("test.gsd");
        let mut gsd_file =
            GsdFile::create(path.clone(), "a", "s", (1, 0)).expect("gsd file should be created");

        gsd_file
            .write_scalars("a", &[1])
            .expect("write should succeed");
        gsd_file.end_frame().expect("write should succeed");

        gsd_file
            .write_scalars("a", &[1])
            .expect("write should succeed");
        gsd_file
            .write_scalars("b", &[2])
            .expect("write should succeed");
        gsd_file
            .write_scalars("c", &[3])
            .expect("write should succeed");
        gsd_file
            .write_scalars("d", &[4])
            .expect("write should succeed");
        gsd_file
            .write_scalars("e", &[5])
            .expect("write should succeed");
        gsd_file
            .write_scalars("f", &[6])
            .expect("write should succeed");
        gsd_file
            .write_scalars("g", &[7])
            .expect("write should succeed");
        gsd_file
            .write_scalars("h", &[8])
            .expect("write should succeed");

        assert_eq!(gsd_file.n_frames(), 0);

        gsd_file.sync_all().expect("write should succeed");

        assert!(gsd_file.find_chunk(0, "a").is_some());
        assert_eq!(gsd_file.n_frames(), 1);

        // frame 1 should not be in the file yet.
        assert!(gsd_file.find_chunk(1, "a").is_none());
        assert!(gsd_file.find_chunk(1, "b").is_none());
        assert!(gsd_file.find_chunk(1, "c").is_none());
        assert!(gsd_file.find_chunk(1, "d").is_none());
        assert!(gsd_file.find_chunk(1, "e").is_none());
        assert!(gsd_file.find_chunk(1, "f").is_none());
        assert!(gsd_file.find_chunk(1, "g").is_none());
        assert!(gsd_file.find_chunk(1, "h").is_none());

        gsd_file.end_frame().expect("write should succeed");
        assert_eq!(gsd_file.n_frames(), 1);
        gsd_file.sync_all().expect("write should succeed");
        assert_eq!(gsd_file.n_frames(), 2);

        // frame 1 should now contain all test chunks
        assert!(gsd_file.find_chunk(1, "a").is_some());
        assert!(gsd_file.find_chunk(1, "b").is_some());
        assert!(gsd_file.find_chunk(1, "c").is_some());
        assert!(gsd_file.find_chunk(1, "d").is_some());
        assert!(gsd_file.find_chunk(1, "e").is_some());
        assert!(gsd_file.find_chunk(1, "f").is_some());
        assert!(gsd_file.find_chunk(1, "g").is_some());
        assert!(gsd_file.find_chunk(1, "h").is_some());
    }

    #[expect(clippy::too_many_lines, reason = "There are many data types to test")]
    #[test]
    fn all_types() {
        let tmp_dir = tempdir().expect("temp dir should be created");
        let path = tmp_dir.path().join("test.gsd");
        let mut gsd_file =
            GsdFile::create(path.clone(), "a", "s", (1, 0)).expect("gsd file should be created");

        let u8_data = [1, 2, 3];
        let u16_data = [4, 5, 6];
        let u32_data = [7, 8, 9];
        let u64_data = [10, 11, 12];
        let i8_data = [-1, -2, -3];
        let i16_data = [-4, -5, -6];
        let i32_data = [-7, -8, -9];
        let i64_data = [-10, -11, -12];
        let f32_data = [13.0, 14.0, 15.0];
        let f64_data = [16.0, 17.0, 18.0];
        let string_data = "Test string.";

        gsd_file
            .write_scalars("u8", &u8_data)
            .expect("write should succeed");
        gsd_file
            .write_scalars("u16", &u16_data)
            .expect("write should succeed");
        gsd_file
            .write_scalars("u32", &u32_data)
            .expect("write should succeed");
        gsd_file
            .write_scalars("u64", &u64_data)
            .expect("write should succeed");
        gsd_file
            .write_scalars("i8", &i8_data)
            .expect("write should succeed");
        gsd_file
            .write_scalars("i16", &i16_data)
            .expect("write should succeed");
        gsd_file
            .write_scalars("i32", &i32_data)
            .expect("write should succeed");
        gsd_file
            .write_scalars("i64", &i64_data)
            .expect("write should succeed");
        gsd_file
            .write_scalars("f32", &f32_data)
            .expect("write should succeed");
        gsd_file
            .write_scalars("f64", &f64_data)
            .expect("write should succeed");
        gsd_file
            .write_string("string", string_data)
            .expect("write should succeed");
        gsd_file.end_frame().expect("write should succeed");
        drop(gsd_file);

        let gsd_file =
            GsdFile::open(path.clone(), Mode::Read).expect("test.gsd should be created above");
        assert_eq!(gsd_file.n_frames(), 1);

        let u8_array = gsd_file
            .iter_scalars::<u8>(0, "u8")
            .expect("u8 should be written above");
        let u16_array = gsd_file
            .iter_scalars::<u16>(0, "u16")
            .expect("u16 should be written above");
        let u32_array = gsd_file
            .iter_scalars::<u32>(0, "u32")
            .expect("u32 should be written above");
        let u64_array = gsd_file
            .iter_scalars::<u64>(0, "u64")
            .expect("u64 should be written above");
        let i8_array = gsd_file
            .iter_scalars::<i8>(0, "i8")
            .expect("i8 should be written above");
        let i16_array = gsd_file
            .iter_scalars::<i16>(0, "i16")
            .expect("i16 should be written above");
        let i32_array = gsd_file
            .iter_scalars::<i32>(0, "i32")
            .expect("i32 should be written above");
        let i64_array = gsd_file
            .iter_scalars::<i64>(0, "i64")
            .expect("i64 should be written above");
        let f32_array = gsd_file
            .iter_scalars::<f32>(0, "f32")
            .expect("f32 should be written above");
        let f64_array = gsd_file
            .iter_scalars::<f64>(0, "f64")
            .expect("f64 should be written above");
        let string_array = gsd_file
            .read_string(0, "string")
            .expect("string should be written above");

        itertools::assert_equal(u8_array, u8_data);
        itertools::assert_equal(u16_array, u16_data);
        itertools::assert_equal(u32_array, u32_data);
        itertools::assert_equal(u64_array, u64_data);
        itertools::assert_equal(i8_array, i8_data);
        itertools::assert_equal(i16_array, i16_data);
        itertools::assert_equal(i32_array, i32_data);
        itertools::assert_equal(i64_array, i64_data);
        itertools::assert_equal(f32_array, f32_data);
        itertools::assert_equal(f64_array, f64_data);
        assert_eq!(string_array, string_data);

        assert_eq!(
            GsdFile::size_of(u8::gsd_data_type()).expect("type should be valid"),
            size_of::<u8>()
        );
        assert_eq!(
            GsdFile::size_of(u16::gsd_data_type()).expect("type should be valid"),
            size_of::<u16>()
        );
        assert_eq!(
            GsdFile::size_of(u32::gsd_data_type()).expect("type should be valid"),
            size_of::<u32>()
        );
        assert_eq!(
            GsdFile::size_of(u64::gsd_data_type()).expect("type should be valid"),
            size_of::<u64>()
        );
        assert_eq!(
            GsdFile::size_of(i8::gsd_data_type()).expect("type should be valid"),
            size_of::<i8>()
        );
        assert_eq!(
            GsdFile::size_of(i16::gsd_data_type()).expect("type should be valid"),
            size_of::<i16>()
        );
        assert_eq!(
            GsdFile::size_of(i32::gsd_data_type()).expect("type should be valid"),
            size_of::<i32>()
        );
        assert_eq!(
            GsdFile::size_of(i64::gsd_data_type()).expect("type should be valid"),
            size_of::<i64>()
        );
        assert_eq!(
            GsdFile::size_of(f32::gsd_data_type()).expect("type should be valid"),
            size_of::<f32>()
        );
        assert_eq!(
            GsdFile::size_of(f64::gsd_data_type()).expect("type should be valid"),
            size_of::<f64>()
        );

        assert_eq!(
            gsd_file
                .find_chunk(0, "u8")
                .expect("u8 should be written above")
                .data_type(),
            Some(DataType::U8)
        );
        assert_eq!(
            gsd_file
                .find_chunk(0, "u16")
                .expect("u16 should be written above")
                .data_type(),
            Some(DataType::U16)
        );
        assert_eq!(
            gsd_file
                .find_chunk(0, "u32")
                .expect("u32 should be written above")
                .data_type(),
            Some(DataType::U32)
        );
        assert_eq!(
            gsd_file
                .find_chunk(0, "u64")
                .expect("c should be written above")
                .data_type(),
            Some(DataType::U64)
        );
        assert_eq!(
            gsd_file
                .find_chunk(0, "i8")
                .expect("i8 should be written above")
                .data_type(),
            Some(DataType::I8)
        );
        assert_eq!(
            gsd_file
                .find_chunk(0, "i16")
                .expect("i16 should be written above")
                .data_type(),
            Some(DataType::I16)
        );
        assert_eq!(
            gsd_file
                .find_chunk(0, "i32")
                .expect("i32 should be written above")
                .data_type(),
            Some(DataType::I32)
        );
        assert_eq!(
            gsd_file
                .find_chunk(0, "i64")
                .expect("i64 should be written above")
                .data_type(),
            Some(DataType::I64)
        );
        assert_eq!(
            gsd_file
                .find_chunk(0, "f32")
                .expect("f32 should be written above")
                .data_type(),
            Some(DataType::F32)
        );
        assert_eq!(
            gsd_file
                .find_chunk(0, "f64")
                .expect("f64 should be written above")
                .data_type(),
            Some(DataType::F64)
        );
        assert_eq!(
            gsd_file
                .find_chunk(0, "string")
                .expect("string should be written above")
                .data_type(),
            Some(DataType::String)
        );
    }

    #[test]
    fn dimensions() {
        let tmp_dir = tempdir().expect("temp dir should be created");
        let path = tmp_dir.path().join("test.gsd");
        let mut gsd_file =
            GsdFile::create(path.clone(), "a", "s", (1, 0)).expect("gsd file should be created");

        let initial_size = gsd_file
            .file
            .metadata()
            .expect("metadata should be valid")
            .len();

        gsd_file
            .write_scalars::<u64, _>("a", [])
            .expect("write should succeed");
        gsd_file.end_frame().expect("write should succeed");
        gsd_file
            .write_scalars::<u64, _>("b", &[1, 2, 3, 4, 5, 6])
            .expect("write should succeed");

        gsd_file
            .write_arrays("c", &[[1u64, 2, 3], [4, 5, 6]])
            .expect("write should succeed");
        gsd_file.end_frame().expect("write should succeed");

        gsd_file.sync_all().expect("write should succeed");
        let final_size = gsd_file
            .file
            .metadata()
            .expect("metadata should be valid")
            .len();
        assert_eq!(final_size, gsd_file.file_len);
        assert_eq!(final_size, initial_size + (12 * size_of::<u64>()) as u64);

        drop(gsd_file);

        let gsd_file =
            GsdFile::open(path.clone(), Mode::Read).expect("test.gsd should be created above");
        assert_eq!(gsd_file.n_frames(), 2);

        let array_a = gsd_file
            .iter_scalars::<u64>(0, "a")
            .expect("a should be written above");
        assert_eq!(array_a.len(), 0);

        let array_b = gsd_file
            .iter_scalars::<u64>(1, "b")
            .expect("b should be written above");
        assert_eq!(array_b.len(), 6);
        itertools::assert_equal(array_b, [1, 2, 3, 4, 5, 6]);

        // Scalar data can be read as an array with M=1.
        let array_b = gsd_file
            .iter_arrays::<u64, 1>(1, "b")
            .expect("b should be written above");
        assert_eq!(array_b.len(), 6);
        itertools::assert_equal(array_b, [[1], [2], [3], [4], [5], [6]]);

        let array_c = gsd_file
            .iter_arrays::<u64, 3>(1, "c")
            .expect("c should be written above");
        itertools::assert_equal(array_c, [[1, 2, 3], [4, 5, 6]]);

        let entry_a = gsd_file
            .find_chunk(0, "a")
            .expect("a should be written above");
        assert_eq!(entry_a.frame(), 0);
        assert_eq!(entry_a.rows(), 0);
        assert_eq!(entry_a.columns(), 1);
        assert_eq!(entry_a.data_type(), Some(DataType::U64));

        let entry_b = gsd_file
            .find_chunk(1, "b")
            .expect("a should be written above");
        assert_eq!(entry_b.frame(), 1);
        assert_eq!(entry_b.rows(), 6);
        assert_eq!(entry_b.columns(), 1);
        assert_eq!(entry_b.data_type(), Some(DataType::U64));

        let entry_c = gsd_file
            .find_chunk(1, "c")
            .expect("c should be written above");
        assert_eq!(entry_c.frame(), 1);
        assert_eq!(entry_c.rows(), 2);
        assert_eq!(entry_c.columns(), 3);
        assert_eq!(entry_c.data_type(), Some(DataType::U64));
    }

    #[test]
    fn invalid_writes() {
        let tmp_dir = tempdir().expect("temp dir should be created");
        let path = tmp_dir.path().join("test.gsd");
        let mut gsd_file =
            GsdFile::create(path.clone(), "a", "s", (1, 0)).expect("gsd file should be created");

        let result = gsd_file.write_arrays::<u64, _, 0>("a", []);
        assert!(matches!(
            result,
            Err(WriteError::Encode(_, _, EncodeError::InvalidColumns(_)))
        ));

        let mut gsd_file =
            GsdFile::open(path.clone(), Mode::Read).expect("test.gsd should be created above");

        let result = gsd_file.write_scalars::<u64, _>("a", []);
        assert!(matches!(
            result,
            Err(WriteError::Encode(_, _, EncodeError::NotWritable))
        ));

        let result = gsd_file.end_frame();
        assert!(matches!(result, Err(EncodeError::NotWritable)));

        let result = gsd_file.sync_all();
        assert!(matches!(result, Err(EncodeError::NotWritable)));
    }

    #[test]
    fn duplicate_chunk_name() {
        let tmp_dir = tempdir().expect("temp dir should be created");
        let path = tmp_dir.path().join("test.gsd");
        let mut gsd_file =
            GsdFile::create(path.clone(), "a", "s", (1, 0)).expect("gsd file should be created");

        gsd_file
            .write_scalars("a", &[1])
            .expect("write should succeed");
        let result = gsd_file.write_scalars("a", &[1, 2]);
        assert!(matches!(
            result,
            Err(WriteError::Encode(
                _,
                _,
                EncodeError::DuplicateChunkName(_, _)
            ))
        ));
    }

    #[test]
    fn read_invalid_reads() {
        let tmp_dir = tempdir().expect("temp dir should be created");
        let path = tmp_dir.path().join("test.gsd");
        let mut gsd_file =
            GsdFile::create(path.clone(), "a", "s", (1, 0)).expect("gsd file should be created");

        gsd_file
            .write_scalars("a", &[1u64])
            .expect("write should succeed");
        gsd_file
            .write_arrays("b", &[[1u64, 2], [3, 4]])
            .expect("write should succeed");
        gsd_file.end_frame().expect("write should succeed");
        gsd_file.sync_all().expect("write should succeed");

        let result = gsd_file.iter_scalars::<u32>(0, "a");
        assert!(matches!(
            result,
            Err(ReadError::Decode(_, _, DecodeError::InvalidType(_, _)))
        ));

        let result = gsd_file.iter_scalars::<u64>(0, "b");
        assert!(matches!(
            result,
            Err(ReadError::Decode(_, _, DecodeError::InvalidColumns(1, 2)))
        ));

        let result = gsd_file.iter_arrays::<u64, 2>(0, "a");
        assert!(matches!(
            result,
            Err(ReadError::Decode(_, _, DecodeError::InvalidColumns(2, 1)))
        ));

        let result = gsd_file.iter_arrays::<u64, 8>(0, "b");
        assert!(matches!(
            result,
            Err(ReadError::Decode(_, _, DecodeError::InvalidColumns(8, 2)))
        ));

        let result = gsd_file.iter_scalars::<u32>(1, "a");
        assert!(matches!(result, Err(ReadError::ChunkNotFound(_, _))));

        let result = gsd_file.iter_scalars::<u32>(0, "q");
        assert!(matches!(result, Err(ReadError::ChunkNotFound(_, _))));
    }

    #[test]
    fn chunk_name_limit() {
        let tmp_dir = tempdir().expect("temp dir should be created");
        let path = tmp_dir.path().join("test.gsd");
        let mut gsd_file =
            GsdFile::create(path.clone(), "a", "s", (1, 0)).expect("gsd file should be created");

        for i in 0..u16::MAX {
            gsd_file
                .write_scalars::<u64, _>(&format!("{i:x}"), [])
                .expect("write should succeed");
        }

        let i = u16::MAX;
        let result = gsd_file.write_scalars::<u64, _>(&format!("{i:x}"), []);
        assert!(matches!(
            result,
            Err(WriteError::Encode(_, _, EncodeError::NameListOverflow))
        ));

        drop(gsd_file);

        let gsd_file =
            GsdFile::open(path.clone(), Mode::Read).expect("test.gsd should be created above");

        assert_eq!(gsd_file.name_id().len(), u16::MAX as usize);
        for i in 0..u16::MAX {
            assert!(gsd_file.name_id().contains_key(&format!("{i:x}")));
        }

        let size = gsd_file
            .file
            .metadata()
            .expect("metadata should be valid")
            .len();
        assert_eq!(size, gsd_file.file_len);
    }

    #[test]
    fn expand_index_multi() {
        const N_ENTRIES: u16 = 1024;

        let tmp_dir = tempdir().expect("temp dir should be created");
        let path = tmp_dir.path().join("test.gsd");
        let mut gsd_file =
            GsdFile::create(path.clone(), "a", "s", (1, 0)).expect("gsd file should be created");

        for i in 0..N_ENTRIES {
            gsd_file
                .write_scalars::<u16, _>(&format!("{i:x}"), &[i])
                .expect("write should succeed");
        }
        gsd_file.end_frame().expect("write should succeed");
        gsd_file.sync_all().expect("write should succeed");

        drop(gsd_file);

        let gsd_file =
            GsdFile::open(path.clone(), Mode::Read).expect("test.gsd should be created above");

        assert_eq!(gsd_file.index.n, u64::from(N_ENTRIES));
        for i in 0..N_ENTRIES {
            let array = gsd_file
                .iter_scalars::<u16>(0, &format!("{i:x}"))
                .expect("read should succeed");
            itertools::assert_equal(array, [i]);
        }
    }

    #[test]
    fn string() {
        let tmp_dir = tempdir().expect("temp dir should be created");
        let path = tmp_dir.path().join("test.gsd");
        let mut gsd_file =
            GsdFile::create(path.clone(), "a", "s", (1, 0)).expect("gsd file should be created");

        gsd_file
            .write_string("a", "this is a string")
            .expect("write should succeed");
        gsd_file.end_frame().expect("write should succeed");
        gsd_file
            .write_scalars::<u8, _>("b", &[0, 159, 146, 150])
            .expect("write should succeed");
        gsd_file.end_frame().expect("write should succeed");
        gsd_file.sync_all().expect("write should succeed");

        let a = gsd_file.read_string(0, "a").expect("read should succeed");
        assert_eq!(a, "this is a string");

        let b = gsd_file.read_string(1, "b");
        assert!(matches!(
            b,
            Err(ReadError::Decode(_, _, DecodeError::InvalidType(11, 1)))
        ));
    }
}
