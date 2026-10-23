//! Error types for scene parsing, serialization and array reference syntax.

use thiserror::Error;

/// Errors returned when parsing or serializing a [`crate::Scene`] or its references.
#[derive(Debug, Error)]
pub enum SceneError {
    /// JSON syntax or schema-level deserialization/serialization failure.
    #[error("scene JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Filesystem I/O failure when reading or writing a scene file on native targets.
    #[error("scene I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Malformed array reference string (for example empty array name or unknown component).
    #[error("invalid array reference `{reference}`: {reason}")]
    InvalidArrayRef {
        /// The raw array reference string that failed to parse.
        reference: String,
        /// Explanation of why the reference is invalid.
        reason: String,
    },
}
