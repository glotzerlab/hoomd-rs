// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Directly access GSD data chunks.

TODO: Expand documentation.
 */

use memmap2::Mmap;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, SeekFrom, prelude::*};
use std::num::TryFromIntError;
use std::path::{Path, PathBuf};
use std::string::FromUtf8Error;
use thiserror::Error;

/// The name buffer is a multiple of `NAME_SIZE` bytes.
const NAME_SIZE: usize = 64;

/// Number of bytes in an index entry.
const INDEX_ENTRY_SIZE: usize = 32;

/// Number of bytes in the header.
const HEADER_SIZE: usize = 256;

/// Magic value identifying a GSD file
const MAGIC_ID: u64 = 0x65DF_65DF_65DF_65DF;

/// Current GSD file version
const CURRENT_FILE_VERSION: (u16, u16) = (2, 1);

/// The size of the file index in new GSD files.
const INITIAL_INDEX_SIZE: usize = 128;

/// Initial name list size
const INITIAL_NAME_BUFFER_SIZE: usize = 1024;


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
    IndexOutOfBounds(usize, usize),

    /// Name list outside the file.
    #[error("name list out of bounds (location={0}, length={1})")]
    NameListOutOfBounds(usize, usize),

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
let (data, index_entry) = gsd_file.read_array::<u64>(0, "configuration/step")?;
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

    /// Convert a native endian byte slice to this type.
    #[doc(hidden)]
    fn from_ne_byte_slice(bytes: &[u8]) -> Self;
}

impl Type for u8 {
    #[inline]
    fn gsd_data_type() -> u8 {
        1
    }
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        bytes[0]
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
}
impl Type for u32 {
    #[inline]
    fn gsd_data_type() -> u8 {
        3
    }
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        u32::from_ne_bytes(bytes.try_into().expect("byte slice should contain 4 bytes")) 
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
}
impl Type for i8 {
    #[inline]
    fn gsd_data_type() -> u8 {
        5
    }
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        i8::from_ne_bytes(bytes.try_into().expect("byte slice should contain 1 byte")) 
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
}
impl Type for i32 {
    #[inline]
    fn gsd_data_type() -> u8 {
        7
    }
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        i32::from_ne_bytes(bytes.try_into().expect("byte slice should contain 4 bytes")) 
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
}
impl Type for f32 {
    #[inline]
    fn gsd_data_type() -> u8 {
        9
    }
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        f32::from_ne_bytes(bytes.try_into().expect("byte slice should contain 8 bytes")) 
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
}

/// In memory representation of the GSD file header.
#[derive(Debug, PartialEq)]
pub(crate) struct GsdHeader {
    /// Magic number marking that this is a GSD file.
    magic: u64,

    /// Location of the chunk index in the file.
    index_location: usize,

    /// Number of index entries that will fit in the space allocated.
    index_allocated_entries: usize,

    /// Location of the name list in the file.
    namelist_location: usize,

    /// Number of bytes in the namelist divided by `NAME_SIZE`.
    namelist_allocated_entries: usize,

    /// Schema version.
    schema_version: (u16, u16),

    /// GSD file format version.
    gsd_version: (u16, u16),

    /// Name of the application that generated this file.
    application: String,

    /// Name of data schema.
    schema: String,
}

/** Interact with GSD files on the filesystem.
*/
#[derive(Debug)]
pub struct GsdFile {
    /// The underlying file.
    file: File,

    /// Parsed copy of the file's header.
    header: GsdHeader,

    /// Memory map of the file.
    mmap: Mmap,

    /// Length of the file in bytes.
    file_len: usize,

    /// Name/id mapping.
    name_id: HashMap<String, u16>,

    /// Number of names in the map.
    n_names: usize,

    /// Number of index entries.
    n_index_entries: usize,

    /// Index of the current frame.
    current_frame: u64,
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
    fn try_from_ne_bytes(value: [u8; HEADER_SIZE]) -> Result<Self, DecodeError> {
        // Validate the magic number first to ensure that we expect the rest
        // of the header to be formatted appropriately. Otherwise, later
        // error checks in this method will be examining undefined data.
        let (magic, rest) = extract_ne_u64(&value);
        if magic != MAGIC_ID {
            return Err(DecodeError::InvalidFileIdentifier(magic));
        }

        let (index_location, rest) = extract_ne_u64(rest);
        let index_location =
            usize::try_from(index_location).map_err(DecodeError::UnaddressableIndex)?;

        let (index_allocated_entries, rest) = extract_ne_u64(rest);
        let index_allocated_entries =
            usize::try_from(index_allocated_entries).map_err(DecodeError::UnaddressableIndex)?;

        let (namelist_location, rest) = extract_ne_u64(rest);
        let namelist_location =
            usize::try_from(namelist_location).map_err(DecodeError::UnaddressableIndex)?;

        let (namelist_allocated_entries, rest) = extract_ne_u64(rest);
        let namelist_allocated_entries =
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

    fn to_ne_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut result = [0u8; HEADER_SIZE];
        result[0..8].copy_from_slice(&self.magic.to_ne_bytes());
        result[8..16].copy_from_slice(&(self.index_location as u64).to_ne_bytes());
        result[16..24].copy_from_slice(&(self.index_allocated_entries as u64).to_ne_bytes());
        result[24..32].copy_from_slice(&(self.namelist_location as u64).to_ne_bytes());
        result[32..40].copy_from_slice(&(self.namelist_allocated_entries as u64).to_ne_bytes());
        let schema_version: u32 = (self.schema_version.0 as u32) << 16 | (self.schema_version.1 as u32);
        result[40..44].copy_from_slice(&schema_version.to_ne_bytes());
        let gsd_version: u32 = (self.gsd_version.0 as u32) << 16 | (self.gsd_version.1 as u32);
        result[44..48].copy_from_slice(&gsd_version.to_ne_bytes());
        result[48..48+self.application.len()].copy_from_slice(self.application.as_str().as_bytes());
        result[112..112+self.schema.len()].copy_from_slice(self.schema.as_str().as_bytes());

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
}

impl GsdFile {
    /** Open a GSD file for reading.

    TODO: Full docs.
    TODO: Open in read only vs read/write?
    */
    #[inline]
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, OpenError> {
        let file = File::open(&path).map_err(|e| OpenError::IO(path.as_ref().into(), e))?;
        GsdFile::from_file(file).map_err(|e| OpenError::Decode(path.as_ref().into(), e))
    }

    /** Create a GSD file.

    Creates a GSD file at the given path, overwriting any file that may
    already exist.

    TODO: Description.
    TODO: Examples.
    */
    #[inline]
    pub fn create<P: AsRef<Path>>(path: P, application: &str, schema: &str, schema_version: (u16, u16)) -> Result<Self, OpenError> {
        let mut file = File::options().read(true).write(true).create(true).truncate(true).open(&path).map_err(|e| OpenError::IO(path.as_ref().into(), e))?;
        GsdFile::initialize_file(&mut file, &path, application, schema, schema_version)?;
        GsdFile::from_file(file).map_err(|e| OpenError::Decode(path.as_ref().into(), e))
    }

    /** Create a new GSD file.

    Creates a new GSD file at the given path, returning an error when the
    path already exists.

    TODO: Description.
    TODO: Examples.
    */
    #[inline]
    pub fn create_new<P: AsRef<Path>>(path: P, application: &str, schema: &str, schema_version: (u16, u16)) -> Result<Self, OpenError> {
        let mut file = File::options().read(true).write(true).create_new(true).open(&path).map_err(|e| OpenError::IO(path.as_ref().into(), e))?;
        GsdFile::initialize_file(&mut file, &path, application, schema, schema_version)?;
        GsdFile::from_file(file).map_err(|e| OpenError::Decode(path.as_ref().into(), e))
    }
    
    /// Initialize an empty file.
    fn initialize_file<P: AsRef<Path>>(file: &mut File, path: &P, application: &str, schema: &str, schema_version: (u16, u16)) -> Result<(), OpenError> {

        let application = String::from(application);
        if application.len() > NAME_SIZE-1 {
            return Err(OpenError::NameToLong(application));
        }
        let schema = String::from(schema);
        if schema.len() > NAME_SIZE-1 {
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
            namelist_allocated_entries: INITIAL_NAME_BUFFER_SIZE / NAME_SIZE,
        };

        file.write_all(&header.to_ne_bytes()).map_err(|e| OpenError::IO(path.as_ref().into(), e))?;
        let index = [0u8; INITIAL_INDEX_SIZE * INDEX_ENTRY_SIZE];
        file.write_all(&index).map_err(|e| OpenError::IO(path.as_ref().into(), e))?;
        let namelist = [0u8; INITIAL_NAME_BUFFER_SIZE];
        file.write_all(&namelist).map_err(|e| OpenError::IO(path.as_ref().into(), e))?;

        Ok(())
    }

    /// Populate the fields in `GsdFile` given an open `File`.
    fn from_file(file: File) -> Result<GsdFile, DecodeError> {
        let mut file = file;
        file.rewind()?;

        let mut header_bytes = [0_u8; HEADER_SIZE];
        file.read_exact(&mut header_bytes)?;
        let header = GsdHeader::try_from_ne_bytes(header_bytes)?;

        let file_len = file.seek(SeekFrom::End(0))?;
        let file_len = usize::try_from(file_len).map_err(DecodeError::UnaddressableContent)?;

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
        if header.namelist_location > file_len
            || namelist_range_end > file_len
            || header.namelist_allocated_entries == 0
        {
            return Err(DecodeError::NameListOutOfBounds(
                header.namelist_location,
                header.namelist_allocated_entries * NAME_SIZE,
            ));
        }

        let mmap = unsafe { Mmap::map(&file)? };
        if mmap[namelist_range_end - 1] != 0 {
            return Err(DecodeError::NameListNotTerminated);
        }

        let (name_id, n_names) =
            GsdFile::decode_name_map(&mmap[header.namelist_location..namelist_range_end])?;

        // TODO: Write buffers.
        // TODO: silently upgrade writable files to the latest minor version.

        let mut gsd_file = GsdFile {
            file,
            header,
            mmap,
            file_len,
            name_id,
            n_names,
            n_index_entries: 0,
            current_frame: 0,
        };

        gsd_file.n_index_entries = gsd_file.count_index_entries()?;
        if gsd_file.n_index_entries > 0 {
            gsd_file.current_frame = gsd_file.get_index(gsd_file.n_index_entries - 1).frame + 1;
        }

        Ok(gsd_file)
    }

    /// Read the initial name map from the file.
    fn decode_name_map(bytes: &[u8]) -> Result<(HashMap<String, u16>, usize), DecodeError> {
        let mut name_id = HashMap::new();
        let mut bytes = bytes;

        let mut current_id: u16 = 0;
        loop {
            let (name, rest) =
                extract_null_terminated_utf8(bytes).map_err(DecodeError::InvalidChunkName)?;
            bytes = rest;
            let previous = name_id.insert(name, current_id);
            if previous.is_some() {
                return Err(DecodeError::DuplicateChunkName);
            }
            current_id += 1;

            if bytes.is_empty() || bytes[0] == 0 {
                break;
            }
        }
        Ok((name_id, usize::from(current_id)))
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
    fn get_index(&self, i: usize) -> IndexEntry {
        // get_index is an internal method, assume that any caller has already
        // called remap() if needed. Verify this in debug builds.
        debug_assert!(self.mmap.len() == self.file_len);

        let start = self.header.index_location + i * INDEX_ENTRY_SIZE;
        let end = start + INDEX_ENTRY_SIZE;
        debug_assert!(
            end < self.header.index_location
                + self.header.index_allocated_entries * INDEX_ENTRY_SIZE
        );
        let bytes: [u8; INDEX_ENTRY_SIZE] = self.mmap[start..end]
            .try_into()
            .expect("slice should always be the correct size");
        IndexEntry::from_ne_bytes(bytes)
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
                if entry.location + total_size > self.file_len as u64 {
                    return false;
                }
            }
            None => return false,
        }

        // is_entry_valid is used before the file is fully loaded and the number
        // of frames is not yet known. Check that the frame is at least within
        // the number of allocated index entries.
        if entry.frame >= self.header.index_allocated_entries as u64 {
            return false;
        }

        // TODO: include buffered names
        if usize::from(entry.id) >= self.n_names {
            return false;
        }

        if entry.flags != 0 {
            return false;
        }

        true
    }

    /// Determine the number of frames in the file.
    fn count_index_entries(&self) -> Result<usize, DecodeError> {
        let first_entry = self.get_index(0);
        if first_entry.location != 0 && !self.is_entry_valid(&first_entry) {
            return Err(DecodeError::CorruptIndexEntry(first_entry));
        }

        if self.get_index(0).location == 0 {
            return Ok(0);
        }

        // determine the number of index entries (marked by location = 0)
        // binary search for the first index entry with location 0
        let mut l: usize = 0;
        let mut r = self.header.index_allocated_entries;

        // progressively narrow the search window by halves
        loop {
            let m = l.midpoint(r);

            // file is corrupt if any index entry is invalid or frame does not increase
            // monotonically
            let entry_m = self.get_index(m);
            let entry_l = self.get_index(l);

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
        if frame >= self.current_frame {
            return None;
        }

        let id = match self.name_id.get(name) {
            None => return None,
            Some(id) => *id,
        };

        // binary search for the index entry
        let mut l: usize = 0;
        let mut r = self.n_index_entries - 1;

        while l <= r {
            let m = l.midpoint(r);
            let index_entry_m = self.get_index(m);
            match (index_entry_m.frame, index_entry_m.id).cmp(&(frame, id)) {
                Ordering::Less => l = m + 1,
                Ordering::Greater => r = m - 1,
                Ordering::Equal => return Some(index_entry_m),
            }
        }
        None
    }

    /** Read an array chunk.

    Returns [`Ok(data, index_entry)`](Result::Ok) when the data chunk is present
    in the file and `Err(`[`ReadError::ChunkNotFound`]`)` when it is not.

    # Errors

    `read_arrayk` may experience I/O errors or find corrupt data in the file. The
    returned [`ReadError`] describes the cause of any error encountered.

    # Example

    ```
    use hoomd_gsd::file_layer::GsdFile;

    # fn func(gsd_file: &mut GsdFile) -> Result<(), Box<dyn std::error::Error>> {
    let (data, index_entry) = gsd_file.read_array::<u64>(0, "configuration/step")?;
    # Ok(())
    # }
    ```
    */
    pub fn read_array<T: Type>(&mut self, frame: u64, name: &str) -> Result<(Vec<T>, IndexEntry), ReadError> {
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
            return Ok((Vec::new(), index_entry));
        }

        self.read_array_details(&index_entry)
            .map(|v| (v, index_entry))
            .map_err(|e| ReadError::Decode(name.into(), frame, e))
    } 

    /// Implement the details of `read_array`.
    fn read_array_details<T: Type>(&mut self, index_entry: &IndexEntry) -> Result<Vec<T>, DecodeError> {
        let n_elements = index_entry.n * u64::from(index_entry.m);
        let n_bytes = usize::try_from(n_elements * size_of::<T>() as u64)
            .map_err(DecodeError::UnaddressableContent)?;
        let mut data = Vec::with_capacity(n_bytes);

        let location = usize::try_from(index_entry.location)
            .map_err(DecodeError::UnaddressableContent)?;

        if location + n_bytes > self.mmap.len() {
            self.remap()?;
        }
        debug_assert!(location + n_bytes <= self.mmap.len());
            
        for offset in (location..location+n_bytes).step_by(size_of::<T>()) {
            data.push(T::from_ne_byte_slice(&self.mmap[offset..offset+size_of::<T>()]));
        }

        Ok(data)
    }

    // TODO: Implement read_string. The conversion steps needed for strings
    // cannot be rolled into a generic read_array.
}
