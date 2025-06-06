// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Directly access GSD data chunks.

TODO: Expand documentation.
 */

use memmap2::Mmap;
use std::cmp::Ordering;
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

/// Initial buffer flush threshold.
const INITIAL_FLUSH_THRESHOLD: usize = 1024 * 1024;

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
    NameToLong(String),
}

/// Errors that can occur while reading from a file.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum ReadError {
    /// Encountered an I/O error.
    #[error("I/O error while reading `{0}` at frame {1}")]
    IO(String, u64, #[source] io::Error),

    /// Chunk not found.
    #[error("chunk not found")]
    ChunkNotFound,

    /// Invalid type.
    #[error("invalid type for chunk `{0}` at frame {1}")]
    InvalidType(String, u64),

    /// Cannot decode the file contents.
    #[error("cannot decode `{0}` at frame {1}")]
    Decode(String, u64, #[source] DecodeError),
}

/// Errors that can occur while writing to a file.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum WriteError {
    /// Attempted to write 0 columns.
    #[error("columns must be non-zero")]
    InvalidColumns,

    /// Attempted to write an array with an invalid length.
    #[error("data length {0} is not a multiple of {1}")]
    InvalidDataLength(usize, u32),

    /// Encountered an I/O error.
    #[error("I/O error while writing `{0}` at frame {1}")]
    IO(String, u64, #[source] io::Error),

    /// Cannot add any more chunk names.
    #[error("too many chunk names")]
    NameListOverflow,

    /// File is not writable.
    #[error("file opened in read-only mode")]
    NotWritable,

    /// A chunk name was duplicated in a single frame.
    #[error("chunk `{0}` has already been written in frame {1}")]
    DuplicateChunkName(String, u64),
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

The [`Type`] trait facilitates the generic methods [`GsdFile::read_array`] and
[`GsdFile::write_array`]. When needed, pass the type explicitly to these methods
to read or write data chunks of the given type. In some cases, the Rust compiler
may be able to determine the type from context.

# Examples

Read a [`u64`] data chunk:
```
use hoomd_gsd::file_layer::GsdFile;

# fn func(gsd_file: &mut GsdFile) -> Result<(), Box<dyn std::error::Error>> {
let array = gsd_file.read_array::<u64>(0, "configuration/step")?;
# Ok(())
# }
```

Write a [`f32`] data chunk:
TODO
*/
pub trait Type: private::Sealed {
    /// Value denoting this type in the file layer.
    #[doc(hidden)]
    fn gsd_data_type() -> u8;

    /** Convert a native endian byte slice to this type.

    This is not the proper idiomatic way to do this, but it gets the job done.
    */
    #[doc(hidden)]
    fn from_ne_byte_slice(bytes: &[u8]) -> Self;

    /** Append this type to a native endian byte array.

    This is not the proper idiomatic way to do this, but it gets the job done.
    */
    #[doc(hidden)]
    fn append_ne_bytes(&self, v: &mut Vec<u8>);
}

impl Type for u8 {
    #[inline]
    fn gsd_data_type() -> u8 {
        1
    }
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        bytes[0]
    }
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}
impl Type for u16 {
    #[inline]
    fn gsd_data_type() -> u8 {
        2
    }
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        u16::from_ne_bytes(bytes.try_into().expect("byte slice should contain 2 bytes")) 
    }
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}
impl Type for u32 {
    #[inline]
    fn gsd_data_type() -> u8 {
        3
    }
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        u32::from_ne_bytes(bytes.try_into().expect("byte slice should contain 4 bytes")) 
    }
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}
impl Type for u64 {
    #[inline]
    fn gsd_data_type() -> u8 {
        4
    }
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        u64::from_ne_bytes(bytes.try_into().expect("byte slice should contain 8 bytes")) 
    }
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}
impl Type for i8 {
    #[inline]
    fn gsd_data_type() -> u8 {
        5
    }
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        i8::from_ne_bytes(bytes.try_into().expect("byte slice should contain 1 byte")) 
    }
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}
impl Type for i16 {
    #[inline]
    fn gsd_data_type() -> u8 {
        6
    }
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        i16::from_ne_bytes(bytes.try_into().expect("byte slice should contain 2 bytes")) 
    }
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}
impl Type for i32 {
    #[inline]
    fn gsd_data_type() -> u8 {
        7
    }
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        i32::from_ne_bytes(bytes.try_into().expect("byte slice should contain 4 bytes")) 
    }
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}
impl Type for i64 {
    #[inline]
    fn gsd_data_type() -> u8 {
        8
    }
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        i64::from_ne_bytes(bytes.try_into().expect("byte slice should contain 8 bytes")) 
    }
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}
impl Type for f32 {
    #[inline]
    fn gsd_data_type() -> u8 {
        9
    }
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        f32::from_ne_bytes(bytes.try_into().expect("byte slice should contain 8 bytes")) 
    }
    fn append_ne_bytes(&self, v: &mut Vec<u8>) {
        v.extend(&self.to_ne_bytes());
    }
}
impl Type for f64 {
    #[inline]
    fn gsd_data_type() -> u8 {
        10
    }
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        f64::from_ne_bytes(bytes.try_into().expect("byte slice should contain 8 bytes")) 
    }
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

    /// Index entry write buffer.
    buffer: Vec<u8>,

    /// Pending entries.
    pending: u64,

    /// Chunk ids that have been written in this frame.
    frame_names: HashSet<u16>,
}

/** Interact with GSD files on the filesystem.

# TODO
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

    /// Length of the file in bytes.
    file_len: u64,

    /// Index of the current frame.
    current_frame: u64,

    /// Automatically flush when more than flush_threshold bytes are buffered.
    flush_threshold: usize,
}

/** Properties that describe a given data chunk.

    GSD files store a set of arrays, uniquely identified by their *name* and
    *frame*. The [`GsdFile::find_chunk`] and [`GsdFile::read_array`] methods
    search for a matching index entry. The returned [`IndexEntry`] (if present)
    also carries information about the dimension and type of the array.    
*/
#[derive(Copy, Clone, Debug, PartialEq)]
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

/** Two-dimensional row-major contiguous data structure.

GSD stores all data in named chunks that contain two-dimensional contiguous
arrays in row-major order. [`GsdFile::read_array`] returns an [`Array`]
that includes both the data and the dimensions.
*/
#[derive(Clone, Debug, PartialEq)]
pub struct Array<T> {
    /// Contents.
    data: Vec<T>,

    /// Number of rows in the array.
    rows: u64,

    /// Number of columns in the array.
    columns: u32,
}

/** Data types that can be stored in chunks.

Provided by [`IndexEntry::data_type`].
*/
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

Pass an [`Mode`] value to [`GsdFile::open`].

In the [`Mode::Read`] mode, you can call methods that read the file, such as
[`GsdFile::find_chunk`] and [`GsdFile::read_array`]. Calling methods that write
the file, such as [`GsdFile::write_array`] or [`GsdFile::sync_all`] will result
in an error.

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
    fn to_ne_bytes(&self) -> [u8; HEADER_USIZE] {
        let mut result = [0u8; HEADER_USIZE];
        result[0..8].copy_from_slice(&self.magic.to_ne_bytes());
        result[8..16].copy_from_slice(&self.index_location.to_ne_bytes());
        result[16..24].copy_from_slice(&self.index_allocated_entries.to_ne_bytes());
        result[24..32].copy_from_slice(&self.namelist_location.to_ne_bytes());
        result[32..40].copy_from_slice(&self.namelist_allocated_entries.to_ne_bytes());
        let schema_version = u32::from(self.schema_version.0) << 16 | u32::from(self.schema_version.1);
        result[40..44].copy_from_slice(&schema_version.to_ne_bytes());
        let gsd_version: u32 = u32::from(self.gsd_version.0) << 16 | u32::from(self.gsd_version.1);
        result[44..48].copy_from_slice(&gsd_version.to_ne_bytes());
        result[48..48+self.application.len()].copy_from_slice(self.application.as_bytes());
        result[112..112+self.schema.len()].copy_from_slice(self.schema.as_bytes());

        result
    }
}

impl IndexEntry {
    /** Trajectory frame number.

    # Example
    ```
    use hoomd_gsd::file_layer::IndexEntry;

    # fn func(index_entry: &IndexEntry) {
    let frame = index_entry.frame();
    # }
    ```
    */
    #[must_use]
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /** Number of rows in the array.

    # Example
    ```
    use hoomd_gsd::file_layer::IndexEntry;

    # fn func(index_entry: &IndexEntry) {
    let rows = index_entry.rows();
    # }
    ```
    */
    #[must_use]
    pub fn rows(&self) -> u64 {
        self.n
    }

    /** Number of columns in the array.

    # Example
    ```
    use hoomd_gsd::file_layer::IndexEntry;

    # fn func(index_entry: &IndexEntry) {
    let columns = index_entry.columns();
    # }
    ```
    */
    #[must_use]
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
    /** Open a GSD file for reading.

    TODO: Full docs.
    */
    #[inline]
    pub fn open<P: AsRef<Path>>(path: P, mode: Mode) -> Result<Self, OpenError> {
        let file = File::open(&path).map_err(|e| OpenError::IO(path.as_ref().into(), e))?;
        GsdFile::from_file(file, mode).map_err(|e| OpenError::Decode(path.as_ref().into(), e))
    }

    /** Create a GSD file.

    Creates a GSD file at the given path, overwriting any file that may
    already exist. When successful, return a [`GsdFile`] opened in
    write mode.

    TODO: Description.
    TODO: Examples.
    */
    #[inline]
    pub fn create<P: AsRef<Path>>(path: P, application: &str, schema: &str, schema_version: (u16, u16)) -> Result<Self, OpenError> {
        let mut file = File::options().read(true).write(true).create(true).truncate(true).open(&path).map_err(|e| OpenError::IO(path.as_ref().into(), e))?;
        GsdFile::initialize_file(&mut file, &path, application, schema, schema_version)?;
        GsdFile::from_file(file, Mode::Write).map_err(|e| OpenError::Decode(path.as_ref().into(), e))
    }

    /** Create a new GSD file.

    Creates a new GSD file at the given path, returning an error when the
    path already exists. When successful, return a [`GsdFile`] opened in
    write mode. 

    TODO: Description.
    TODO: Examples.
    */
    #[inline]
    pub fn create_new<P: AsRef<Path>>(path: P, application: &str, schema: &str, schema_version: (u16, u16)) -> Result<Self, OpenError> {
        let mut file = File::options().read(true).write(true).create_new(true).open(&path).map_err(|e| OpenError::IO(path.as_ref().into(), e))?;
        GsdFile::initialize_file(&mut file, &path, application, schema, schema_version)?;
        GsdFile::from_file(file, Mode::Write).map_err(|e| OpenError::Decode(path.as_ref().into(), e))
    }
    
    /// Initialize an empty file.
    fn initialize_file<P: AsRef<Path>>(file: &mut File, path: &P, application: &str, schema: &str, schema_version: (u16, u16)) -> Result<(), OpenError> {

        let application = String::from(application);
        if application.len() as u64 > NAME_SIZE-1 {
            return Err(OpenError::NameToLong(application));
        }
        let schema = String::from(schema);
        if schema.len() as u64 > NAME_SIZE-1 {
            return Err(OpenError::NameToLong(schema));
        }

        let header = GsdHeader {
            magic: MAGIC_ID,
            gsd_version: CURRENT_FILE_VERSION,
            application,
            schema,
            schema_version,
            index_location: HEADER_SIZE,
            index_allocated_entries:  INITIAL_INDEX_SIZE,
            namelist_location: HEADER_SIZE + INDEX_ENTRY_SIZE * INITIAL_INDEX_SIZE,
            namelist_allocated_entries: INITIAL_NAME_LIST_SIZE / NAME_SIZE,
        };

        file.write_all(&header.to_ne_bytes()).map_err(|e| OpenError::IO(path.as_ref().into(), e))?;

        file.set_len(HEADER_SIZE + INDEX_ENTRY_SIZE * INITIAL_INDEX_SIZE + INITIAL_NAME_LIST_SIZE).map_err(|e| OpenError::IO(path.as_ref().into(), e))?;

        file.sync_all().map_err(|e| OpenError::IO(path.as_ref().into(), e))?;
        
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
        if namelist_range_end > file_len
            || header.namelist_allocated_entries == 0
        {
            return Err(DecodeError::NameListOutOfBounds(
                header.namelist_location,
                header.namelist_allocated_entries * NAME_SIZE,
            ));
        }

        let mmap = unsafe { Mmap::map(&file)? };
        let last_namelist_offset = usize::try_from(namelist_range_end - 1).map_err(DecodeError::UnaddressableIndex)?;
        if mmap[last_namelist_offset] != 0 {
            return Err(DecodeError::NameListNotTerminated);
        }

        let start = usize::try_from(header.namelist_location).map_err(DecodeError::UnaddressableIndex)?;
        let end = usize::try_from(namelist_range_end).map_err(DecodeError::UnaddressableIndex)?;
        let name_list = GsdFile::decode_name_map(&mmap[start..end])?;
        let index = Index { n: 0, buffer: Vec::new(), pending: 0, frame_names: HashSet::new() };        

        // TODO: Write buffers.

        let mut gsd_file = GsdFile {
            file,
            mode,
            header,
            mmap,
            file_len,
            name_list,
            index,
            data_buffer: Vec::new(),
            current_frame: 0,
            flush_threshold: INITIAL_FLUSH_THRESHOLD,
        };

        gsd_file.index.n = gsd_file.count_index_entries()?;
        if gsd_file.index.n > 0 {
            let last_entry = gsd_file.get_index(gsd_file.index.n - 1)?;
            gsd_file.current_frame = last_entry.frame + 1;
        }

        // TODO: silently upgrade writable files to the latest minor version.

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
            insert_position += (name.len()+1) as u64;
            
            let previous = name_id.insert(name, current_id);
            if previous.is_some() {
                return Err(DecodeError::DuplicateChunkName);
            }
            current_id += 1;

            // TODO: Detect when there are too many names.
        }

        Ok(NameList { name_id,
            n_names: current_id,
            insert_position,
            buffer: Vec::new(),
            })
    }

    /// Add a new name to the file.
    fn add_name(&mut self, name: &str) -> u16 {
        self.name_list.n_names += 1;
        self.name_list.buffer.extend(name.as_bytes());
        self.name_list.buffer.push(0);
        self.name_list.n_names
    }

    /// Remap the file
    #[cfg(target_os = "linux")]
    fn remap(&mut self) -> Result<(), io::Error> {
        self.mmap
            .remap(self.file_len, memmap2::RemapOptions.new().may_move(true));
    }

    /// Remap the file
    #[cfg(not(target_os = "linux"))]
    fn remap(&mut self) -> Result<(), io::Error> {
        self.mmap = unsafe { Mmap::map(&self.file)? };
        Ok(())
    }

    /// Access a single index entry from the memory map.
    fn get_index(&self, i: u64) -> Result<IndexEntry, DecodeError> {
        // get_index is an internal method, assume that any caller has already
        // called remap() if needed. Verify this in debug builds.
        debug_assert!(self.mmap.len() as u64 == self.file_len);

        let start = self.header.index_location + i * INDEX_ENTRY_SIZE;
        let end = start + INDEX_ENTRY_SIZE;
        debug_assert!(
            end < self.header.index_location
                + self.header.index_allocated_entries * INDEX_ENTRY_SIZE
        );

        let start = usize::try_from(start)
            .map_err(DecodeError::UnaddressableIndex)?;
        let end = usize::try_from(end)
            .map_err(DecodeError::UnaddressableIndex)?;
        let bytes: [u8; INDEX_ENTRY_USIZE] = self.mmap[start..end]
            .try_into()
            .expect("slice should always be the correct size");
        Ok(IndexEntry::from_ne_bytes(bytes))
    }

    /// Get the size of a type given by its identifier.
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
                if entry.location + total_size > self.file_len {
                    return false;
                }
            }
            None => return false,
        }

        // is_entry_valid is used before the file is fully loaded and the number
        // of frames is not yet known. Check that the frame is at least within
        // the number of allocated index entries.
        if entry.frame >= self.header.index_allocated_entries {
            return false;
        }

        // There is no need to include buffered names here because
        // is_entry_valid is only called on file open, not after any write_
        // methods.
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
        let mut l: u64= 0;
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

    Returns [`Some(index_entry)`](Option::Some) when the data chunk is present in the file
    and [`None`] when it is not.

    # Example

    ```
    use hoomd_gsd::file_layer::GsdFile;

    # fn do_something() {};
    # fn func(gsd_file: &mut GsdFile) -> Result<(), Box<dyn std::error::Error>> {
    match gsd_file.find_chunk(0, "configuration/step") {
        Some(index_entry) => do_something(),
        None => (),
    }
    # Ok(())
    # }
    ```
    */
    #[must_use]
    pub fn find_chunk(&self, frame: u64, name: &str) -> Option<IndexEntry> {
        if frame >= self.current_frame || self.index.n == 0 {
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
                return None
            }
        }
        None
    }

    /** Read an array chunk.

    Returns [`Ok(data, index_entry)`](Result::Ok) when the data chunk is present
    in the file and `Err(`[`ReadError::ChunkNotFound`]`)` when it is not.

    # Errors

    `read_array` may experience I/O errors or find corrupt data in the file. The
    returned [`ReadError`] describes the cause of any error encountered.

    # Example

    ```
    use hoomd_gsd::file_layer::GsdFile;

    # fn func(gsd_file: &mut GsdFile) -> Result<(), Box<dyn std::error::Error>> {
    let array = gsd_file.read_array::<u64>(0, "configuration/step")?;
    # Ok(())
    # }
    ```
    */
    pub fn read_array<T: Type>(&self, frame: u64, name: &str) -> Result<Array<T>, ReadError> {
        let index_entry = match self.find_chunk(frame, name) {
            None => return Err(ReadError::ChunkNotFound),
            Some(e) => e,
            };

        if index_entry.data_type != T::gsd_data_type() {
            return Err(ReadError::InvalidType(name.into(), frame));
        }

        if index_entry.location == 0 {        
            return Err(ReadError::Decode(name.into(), frame, DecodeError::CorruptIndexEntry(index_entry)));
        }
        
        if index_entry.n == 0 {
            return Ok(Array{ data: Vec::new(), rows: 0, columns: index_entry.m } );
        }

        self.read_array_details(&index_entry)
            .map_err(|e| ReadError::Decode(name.into(), frame, e))
    } 

    /// Implement the details of `read_array`.
    fn read_array_details<T: Type>(&self, index_entry: &IndexEntry) -> Result<Array<T>, DecodeError> {
        let n_elements = index_entry.n * u64::from(index_entry.m);
        let n_bytes = usize::try_from(n_elements * size_of::<T>() as u64)
            .map_err(DecodeError::UnaddressableContent)?;
        let mut data = Vec::with_capacity(n_bytes);

        let location = usize::try_from(index_entry.location)
            .map_err(DecodeError::UnaddressableContent)?;

        debug_assert!(location + n_bytes <= self.mmap.len());
            
        for offset in (location..location+n_bytes).step_by(size_of::<T>()) {
            data.push(T::from_ne_byte_slice(&self.mmap[offset..offset+size_of::<T>()]));
        }

        Ok(Array { data, rows: index_entry.n, columns: index_entry.m })
    }

    // TODO: Implement read_string. The conversion steps needed for strings
    // cannot be rolled into a generic read_array. But it can leverage
    // `read_array_details<u8>` to reduce code duplication.
    
    // TODO: Consider implementing read_vector_array<Cartesian<N>> and a
    // similar write method. These convenience methods will make it easier
    // to implement the hoomd schema which stores many chunks as vectors.
    // Because Cartesian is always f64, read will need to convert f32 to
    // f64. The hoomd schema is always f32, so we also need a way to
    // opt into casting on write.

    /** Append data to the current frame.

    `write_array` writes two-dimensional array data to a named chunk in
    the current frame of the GSD file.

    # Errors

    Returns a [`WriteError`] when any of the following occur:
    * The file is not opened in a write mode.
    * An I/O error writing to the file.
    * `columns` is 0.
    * The `data` length is not an integer multiple of `columns`.
    * TODO: Error when name has already been written this frame.
    
    */
    pub fn write_array<T: Type>(&mut self, name: &str, columns: u32, data: &[T]) -> Result<(), WriteError> {
        if self.mode != Mode::Write {
            return Err(WriteError::NotWritable);
        }

        if columns == 0 {
            return Err(WriteError::InvalidColumns);
        }

        if data.len() % (columns as usize) != 0 {
            return Err(WriteError::InvalidDataLength(data.len(), columns));
        }

        let id = if let Some(id) = self.name_list.name_id.get(name) { *id } else {
            let id = self.add_name(name);
            self.name_list.name_id.insert(String::from(name), id);
            id
        };

        if id == u16::MAX {
            return Err(WriteError::NameListOverflow);
        }

        if !self.index.frame_names.insert(id) {
            return Err(WriteError::DuplicateChunkName(name.into(), self.current_frame));
        }

        // write_array doesn't actually write any data to the file itself. For
        // performance, it buffers all writes. Above, `add_name` appended any
        // new names to `self.name_list.buffer`. Now, `write_array` needs to
        // construct the index entry and put the bytes of the array in the data
        // buffer. `sync_all` will write the data buffer first, so all index
        // entries can be constructed with the known location:
        // file_len + currently buffered bytes.
        //
        // This implementation is a departure from the GSD C implementation
        // which would eagerly write large arrays directly to the file and
        // buffer the index entries for them. That complicated the code with
        // the need for two levels of index buffering and corrections to some
        // entries' location fields. Due to the need to call `to_ne_bytes`, the
        // Rust code is simpler to write when it always buffers all data.

        let index_entry = IndexEntry {
            frame: self.current_frame,
            n: (data.len() / columns as usize) as u64,
            m: columns,
            location: self.file_len + self.data_buffer.len() as u64,
            id,
            data_type: T::gsd_data_type(),
            flags: 0,
        };

        self.index.buffer.extend(&index_entry.to_ne_bytes());
        self.index.pending += 1;
        
        for value in data {
            value.append_ne_bytes(&mut self.data_buffer);
        }

    Ok(())
    }
}
