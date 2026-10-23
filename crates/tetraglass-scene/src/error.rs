//! Error types for scene parsing, serialization, validation and pipeline evaluation.

use thiserror::Error;

/// Errors returned when parsing, validating or evaluating a [`crate::Scene`].
#[derive(Debug, Error)]
pub enum SceneError {
    /// JSON syntax or schema-level deserialization/serialization failure.
    #[error("scene JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Filesystem I/O failure when reading or writing a scene file on native targets.
    #[error("scene I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Semantic validation failure on a parsed scene.
    #[error("{0}")]
    Validation(#[from] crate::ValidationReport),

    /// Error returned by a visualization filter during pipeline evaluation.
    #[error("filter evaluation error: {0}")]
    Filter(#[from] tetraglass_filters::FilterError),

    /// Referenced source or filter dataset was not found during evaluation.
    #[error("unknown or unloaded dataset `{name}`")]
    MissingDataset {
        /// Name of the missing source or filter dataset.
        name: String,
    },

    /// Malformed array reference string (for example empty array name or unknown component).
    #[error("invalid array reference `{reference}`: {reason}")]
    InvalidArrayRef {
        /// The raw array reference string that failed to parse.
        reference: String,
        /// Explanation of why the reference is invalid.
        reason: String,
    },
}
