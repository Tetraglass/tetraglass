//! Typed attribute arrays (`DataArray`) with `Arc` backing storage, component counts,
//! cached value ranges, and named attribute tables (`Attributes`).

use std::sync::Arc;

use crate::error::CoreError;

/// Allowed component counts per tuple in a [`DataArray`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ComponentCount {
    /// 1 component per tuple (scalar field).
    Scalar = 1,
    /// 3 components per tuple (3D vector field).
    Vector3 = 3,
    /// 6 components per tuple (3D symmetric second-order tensor).
    SymmetricTensor6 = 6,
    /// 9 components per tuple (full 3x3 tensor).
    Tensor9 = 9,
}

impl ComponentCount {
    /// Returns the component count as a `u8`.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Returns the component count as a `usize`.
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self as usize
    }
}

impl TryFrom<u8> for ComponentCount {
    type Error = CoreError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Self::try_from(usize::from(value))
    }
}

impl TryFrom<usize> for ComponentCount {
    type Error = CoreError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Scalar),
            3 => Ok(Self::Vector3),
            6 => Ok(Self::SymmetricTensor6),
            9 => Ok(Self::Tensor9),
            count => Err(CoreError::InvalidComponentCount { count }),
        }
    }
}

/// Scalar element type stored inside an [`ArrayBuffer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    /// 32-bit IEEE-754 floating-point (`f32`).
    F32,
    /// 64-bit IEEE-754 floating-point (`f64`).
    F64,
    /// 32-bit signed integer (`i32`).
    I32,
    /// 64-bit signed integer (`i64`).
    I64,
    /// 8-bit unsigned integer (`u8`).
    U8,
}

/// Reference-counted typed scalar buffer backing a [`DataArray`].
///
/// Cloning an `ArrayBuffer` (or `DataArray`) only increments the `Arc` strong count.
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayBuffer {
    /// Shared `f32` storage.
    F32(Arc<[f32]>),
    /// Shared `f64` storage.
    F64(Arc<[f64]>),
    /// Shared `i32` storage.
    I32(Arc<[i32]>),
    /// Shared `i64` storage.
    I64(Arc<[i64]>),
    /// Shared `u8` storage.
    U8(Arc<[u8]>),
}

impl ArrayBuffer {
    /// Returns the [`ScalarType`] of this buffer.
    #[must_use]
    pub const fn scalar_type(&self) -> ScalarType {
        match self {
            Self::F32(_) => ScalarType::F32,
            Self::F64(_) => ScalarType::F64,
            Self::I32(_) => ScalarType::I32,
            Self::I64(_) => ScalarType::I64,
            Self::U8(_) => ScalarType::U8,
        }
    }

    /// Returns the total number of scalar elements (`num_tuples * components`) in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::F32(b) => b.len(),
            Self::F64(b) => b.len(),
            Self::I32(b) => b.len(),
            Self::I64(b) => b.len(),
            Self::U8(b) => b.len(),
        }
    }

    /// Returns `true` if the buffer contains zero scalar elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reads the scalar element at flat `index` converted to `f64`.
    #[must_use]
    pub fn get_as_f64(&self, index: usize) -> Option<f64> {
        match self {
            Self::F32(b) => b.get(index).map(|&v| f64::from(v)),
            Self::F64(b) => b.get(index).copied(),
            Self::I32(b) => b.get(index).map(|&v| f64::from(v)),
            Self::I64(b) => b.get(index).map(|&v| v as f64),
            Self::U8(b) => b.get(index).map(|&v| f64::from(v)),
        }
    }
}

/// Cached minimum and maximum finite values for an array component or magnitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValueRange {
    /// Minimum finite value (`0.0` if the array has no finite tuples).
    pub min: f64,
    /// Maximum finite value (`0.0` if the array has no finite tuples).
    pub max: f64,
}

impl ValueRange {
    /// Returns the span `self.max - self.min`.
    #[must_use]
    pub fn span(&self) -> f64 {
        self.max - self.min
    }
}

/// A typed point or cell attribute array with component metadata and cached value ranges.
///
/// ```
/// use tetraglass_core::{ComponentCount, DataArray};
///
/// let arr = DataArray::from_f64(ComponentCount::Vector3, vec![3.0, 4.0, 0.0, 0.0, -5.0, 12.0])
///     .expect("valid vector array");
/// assert_eq!(arr.num_tuples(), 2);
/// assert_eq!(arr.component_range(0).unwrap().min, 0.0);
/// assert_eq!(arr.component_range(0).unwrap().max, 3.0);
/// assert_eq!(arr.magnitude_range().unwrap().min, 5.0);
/// assert_eq!(arr.magnitude_range().unwrap().max, 13.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DataArray {
    components: ComponentCount,
    buffer: ArrayBuffer,
    component_ranges: Vec<ValueRange>,
    magnitude_range: Option<ValueRange>,
}

impl DataArray {
    /// Creates a new [`DataArray`] from a [`ComponentCount`] and an [`ArrayBuffer`], validating
    /// that the buffer length is a multiple of `components` and precomputing per-component and
    /// magnitude ranges.
    pub fn new(components: ComponentCount, buffer: ArrayBuffer) -> Result<Self, CoreError> {
        let ncomp = components.as_usize();
        if !buffer.len().is_multiple_of(ncomp) {
            return Err(CoreError::ArrayLengthMismatch {
                len: buffer.len(),
                components: components.as_u8(),
            });
        }
        let num_tuples = buffer.len() / ncomp;
        let (component_ranges, magnitude_range) = compute_ranges(&buffer, num_tuples, components);

        Ok(Self {
            components,
            buffer,
            component_ranges,
            magnitude_range,
        })
    }

    /// Convenience constructor for `f32` data.
    pub fn from_f32(
        components: ComponentCount,
        values: impl Into<Arc<[f32]>>,
    ) -> Result<Self, CoreError> {
        Self::new(components, ArrayBuffer::F32(values.into()))
    }

    /// Convenience constructor for `f64` data.
    pub fn from_f64(
        components: ComponentCount,
        values: impl Into<Arc<[f64]>>,
    ) -> Result<Self, CoreError> {
        Self::new(components, ArrayBuffer::F64(values.into()))
    }

    /// Convenience constructor for `i32` data.
    pub fn from_i32(
        components: ComponentCount,
        values: impl Into<Arc<[i32]>>,
    ) -> Result<Self, CoreError> {
        Self::new(components, ArrayBuffer::I32(values.into()))
    }

    /// Convenience constructor for `i64` data.
    pub fn from_i64(
        components: ComponentCount,
        values: impl Into<Arc<[i64]>>,
    ) -> Result<Self, CoreError> {
        Self::new(components, ArrayBuffer::I64(values.into()))
    }

    /// Convenience constructor for `u8` data.
    pub fn from_u8(
        components: ComponentCount,
        values: impl Into<Arc<[u8]>>,
    ) -> Result<Self, CoreError> {
        Self::new(components, ArrayBuffer::U8(values.into()))
    }

    /// Returns the number of tuples (`buffer.len() / components`).
    #[must_use]
    pub fn num_tuples(&self) -> usize {
        self.buffer.len() / self.components.as_usize()
    }

    /// Returns the [`ComponentCount`] per tuple.
    #[must_use]
    pub const fn components(&self) -> ComponentCount {
        self.components
    }

    /// Returns the underlying [`ScalarType`].
    #[must_use]
    pub const fn scalar_type(&self) -> ScalarType {
        self.buffer.scalar_type()
    }

    /// Returns a reference to the shared [`ArrayBuffer`].
    #[must_use]
    pub const fn buffer(&self) -> &ArrayBuffer {
        &self.buffer
    }

    /// Returns the cached [`ValueRange`] for every component (`0..components`).
    #[must_use]
    pub fn component_ranges(&self) -> &[ValueRange] {
        &self.component_ranges
    }

    /// Returns the cached [`ValueRange`] for component `component_index`.
    #[must_use]
    pub fn component_range(&self, component_index: usize) -> Option<ValueRange> {
        self.component_ranges.get(component_index).copied()
    }

    /// Returns the cached Euclidean magnitude range for multi-component arrays, or `None` for scalars.
    #[must_use]
    pub const fn magnitude_range(&self) -> Option<ValueRange> {
        self.magnitude_range
    }

    /// Returns the value of `(tuple_index, component_index)` converted to `f64`.
    #[must_use]
    pub fn value_as_f64(&self, tuple_index: usize, component_index: usize) -> Option<f64> {
        let ncomp = self.components.as_usize();
        if component_index >= ncomp {
            return None;
        }
        let flat_idx = tuple_index
            .checked_mul(ncomp)?
            .checked_add(component_index)?;
        self.buffer.get_as_f64(flat_idx)
    }

    /// Returns the Euclidean norm $\|\mathbf{v}\|_2$ of tuple `tuple_index`.
    #[must_use]
    pub fn tuple_magnitude(&self, tuple_index: usize) -> Option<f64> {
        let ncomp = self.components.as_usize();
        let base = tuple_index.checked_mul(ncomp)?;
        if base.checked_add(ncomp)? > self.buffer.len() {
            return None;
        }
        let mut sum_sq = 0.0;
        for c in 0..ncomp {
            let v = self.buffer.get_as_f64(base + c)?;
            sum_sq += v * v;
        }
        Some(sum_sq.sqrt())
    }
}

fn compute_ranges(
    buffer: &ArrayBuffer,
    num_tuples: usize,
    components: ComponentCount,
) -> (Vec<ValueRange>, Option<ValueRange>) {
    let ncomp = components.as_usize();
    let mut comp_min = vec![f64::INFINITY; ncomp];
    let mut comp_max = vec![f64::NEG_INFINITY; ncomp];

    let mut mag_min = f64::INFINITY;
    let mut mag_max = f64::NEG_INFINITY;

    for t in 0..num_tuples {
        let base = t * ncomp;
        let mut sum_sq = 0.0;
        let mut all_finite = true;
        for c in 0..ncomp {
            if let Some(val) = buffer.get_as_f64(base + c) {
                if val.is_finite() {
                    if val < comp_min[c] {
                        comp_min[c] = val;
                    }
                    if val > comp_max[c] {
                        comp_max[c] = val;
                    }
                    sum_sq += val * val;
                } else {
                    all_finite = false;
                }
            }
        }
        if ncomp > 1 && all_finite {
            let mag = sum_sq.sqrt();
            if mag < mag_min {
                mag_min = mag;
            }
            if mag > mag_max {
                mag_max = mag;
            }
        }
    }

    let component_ranges = comp_min
        .into_iter()
        .zip(comp_max)
        .map(|(min, max)| {
            if min <= max {
                ValueRange { min, max }
            } else {
                ValueRange { min: 0.0, max: 0.0 }
            }
        })
        .collect();

    let magnitude_range = if ncomp > 1 {
        if mag_min <= mag_max {
            Some(ValueRange {
                min: mag_min,
                max: mag_max,
            })
        } else {
            Some(ValueRange { min: 0.0, max: 0.0 })
        }
    } else {
        None
    };

    (component_ranges, magnitude_range)
}

/// Ordered collection of named [`DataArray`] attributes attached to points or cells.
///
/// Preserves insertion order so CLI inspection and UI pickers are deterministic.
#[derive(Debug, Clone, PartialEq)]
pub struct Attributes {
    expected_tuples: usize,
    entries: Vec<(String, DataArray)>,
}

impl Attributes {
    /// Creates an empty attribute table expecting `expected_tuples` tuples per array.
    #[must_use]
    pub fn new(expected_tuples: usize) -> Self {
        Self {
            expected_tuples,
            entries: Vec::new(),
        }
    }

    /// Returns the number of tuples required for every array in this table.
    #[must_use]
    pub const fn expected_tuples(&self) -> usize {
        self.expected_tuples
    }

    /// Returns the number of named arrays in the table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no arrays are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Inserts or replaces a named [`DataArray`], checking that `array.num_tuples() == self.expected_tuples()`.
    pub fn insert(&mut self, name: impl Into<String>, array: DataArray) -> Result<(), CoreError> {
        let name = name.into();
        let actual = array.num_tuples();
        if actual != self.expected_tuples {
            return Err(CoreError::AttributeTupleCountMismatch {
                name,
                expected: self.expected_tuples,
                actual,
            });
        }
        if let Some((_, existing)) = self.entries.iter_mut().find(|(k, _)| k == &name) {
            *existing = array;
        } else {
            self.entries.push((name, array));
        }
        Ok(())
    }

    /// Looks up a [`DataArray`] by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&DataArray> {
        self.entries
            .iter()
            .find_map(|(k, v)| if k == name { Some(v) } else { None })
    }

    /// Returns an iterator over `(name, array)` pairs in insertion order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&str, &DataArray)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Returns an iterator over attribute names in insertion order.
    pub fn names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.entries.iter().map(|(k, _)| k.as_str())
    }
}
