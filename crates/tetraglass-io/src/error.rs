//! Error types for `tetraglass-io`.

use tetraglass_core::CoreError;

/// Errors returned when parsing or writing mesh files.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// Underlying grid or array validation error from `tetraglass-core`.
    #[error(transparent)]
    Core(#[from] CoreError),

    /// Standard I/O error (when writing to a `Write` sink or reading a native file path).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Malformed XML syntax at a byte offset.
    #[error("XML syntax error at byte {offset}: {message}")]
    XmlSyntax {
        /// Approximate byte offset in the input buffer.
        offset: u64,
        /// Error message from the XML tokenizer.
        message: String,
    },

    /// Unexpected or missing XML element or attribute in a `.vtu` file.
    #[error("invalid VTU structure at byte {offset}: {reason}")]
    InvalidVtuStructure {
        /// Byte offset in the input buffer.
        offset: u64,
        /// Description of the structural issue.
        reason: String,
    },

    /// Unexpected or malformed section, header, or tag in a Gmsh `.msh` file.
    #[error("invalid MSH structure at byte {offset}: {reason}")]
    InvalidMshStructure {
        /// Byte offset in the input buffer.
        offset: u64,
        /// Description of the structural issue.
        reason: String,
    },

    /// Unsupported VTK encoding, compressor, header type, or scalar type.
    #[error("unsupported VTU feature at byte {offset}: {feature}")]
    UnsupportedFeature {
        /// Byte offset in the input buffer.
        offset: u64,
        /// Name or value of the unsupported feature.
        feature: String,
    },

    /// Failed to parse an ASCII number or attribute value.
    #[error("invalid numeric value '{text}' in array '{array}' at byte {offset}")]
    InvalidNumber {
        /// Name of the `DataArray` or attribute being parsed.
        array: String,
        /// Offending token text.
        text: String,
        /// Byte offset in the input buffer.
        offset: u64,
    },

    /// Base64 decoding error in a `binary` or `appended` `DataArray`.
    #[error("base64 decode error in array '{array}' at byte {offset}: {message}")]
    Base64Decode {
        /// Name of the `DataArray`.
        array: String,
        /// Byte offset in the input buffer.
        offset: u64,
        /// Error message.
        message: String,
    },

    /// Zlib decompression or binary block header error.
    #[error("binary/compressed data error in array '{array}' at byte {offset}: {reason}")]
    BinaryData {
        /// Name of the `DataArray`.
        array: String,
        /// Byte offset in the input buffer.
        offset: u64,
        /// Description of the failure.
        reason: String,
    },
}
