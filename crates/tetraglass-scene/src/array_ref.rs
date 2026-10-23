//! Array reference parsing and component resolution (`docs/scene-format.md`).
//!
//! Scene fields such as `color_by` reference dataset arrays using strings of the form:
//! - `"<array>"` (for example `"pressure"`)
//! - `"<array>:<component>"` (for example `"velocity:magnitude"`, `"velocity:x"`, `"velocity:0"`, `"stress:xy"`)
//! - `"point:<array>"` or `"cell:<array>"` (and `"point:<array>:<component>"` / `"cell:<array>:<component>"`)
//!   to disambiguate when both point and cell data contain an array with the same name.

use std::fmt;
use std::str::FromStr;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tetraglass_core::ComponentCount;

use crate::SceneError;

/// Optional association qualifier (`point:` or `cell:`) in an array reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ArrayAssociation {
    /// Point-associated attribute array (`"point:<array>"`).
    Point,
    /// Cell-associated attribute array (`"cell:<array>"`).
    Cell,
}

impl fmt::Display for ArrayAssociation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Point => f.write_str("point"),
            Self::Cell => f.write_str("cell"),
        }
    }
}

/// Named vector axis component selector (`x`, `y`, `z`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum VectorAxis {
    /// First vector component (`x`, index `0`).
    X,
    /// Second vector component (`y`, index `1`).
    Y,
    /// Third vector component (`z`, index `2`).
    Z,
}

impl VectorAxis {
    /// Returns the zero-based component index (`0`, `1`, or `2`).
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

impl fmt::Display for VectorAxis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::X => f.write_str("x"),
            Self::Y => f.write_str("y"),
            Self::Z => f.write_str("z"),
        }
    }
}

/// Named 3x3 tensor component selector (`xx`, `xy`, `xz`, `yx`, `yy`, `yz`, `zx`, `zy`, `zz`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TensorComponent {
    /// Normal component `(0, 0)`.
    Xx,
    /// Shear component `(0, 1)`.
    Xy,
    /// Shear component `(0, 2)`.
    Xz,
    /// Shear component `(1, 0)`.
    Yx,
    /// Normal component `(1, 1)`.
    Yy,
    /// Shear component `(1, 2)`.
    Yz,
    /// Shear component `(2, 0)`.
    Zx,
    /// Shear component `(2, 1)`.
    Zy,
    /// Normal component `(2, 2)`.
    Zz,
}

impl fmt::Display for TensorComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Xx => "xx",
            Self::Xy => "xy",
            Self::Xz => "xz",
            Self::Yx => "yx",
            Self::Yy => "yy",
            Self::Yz => "yz",
            Self::Zx => "zx",
            Self::Zy => "zy",
            Self::Zz => "zz",
        };
        f.write_str(s)
    }
}

/// Component selector parsed from the `:<component>` suffix of an array reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComponentSelector {
    /// Euclidean norm (`"magnitude"`).
    Magnitude,
    /// Named vector axis (`"x"`, `"y"`, `"z"`).
    Vector(VectorAxis),
    /// Named tensor component (`"xx"`, `"xy"`, ...).
    Tensor(TensorComponent),
    /// Explicit zero-based numeric component index (`"0"`, `"1"`, ...).
    Index(usize),
}

/// Concrete scalar extraction mode resolved against a specific [`ComponentCount`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResolvedComponent {
    /// Extract the Euclidean magnitude across all components.
    Magnitude,
    /// Extract a specific zero-based component index.
    Component(usize),
}

impl ComponentSelector {
    /// Parses a component selector token (`"x"`, `"y"`, `"z"`, `"magnitude"`, `"xx"`..`"zz"`, or a zero-based index).
    pub fn parse(token: &str) -> Result<Self, String> {
        match token {
            "magnitude" => Ok(Self::Magnitude),
            "x" => Ok(Self::Vector(VectorAxis::X)),
            "y" => Ok(Self::Vector(VectorAxis::Y)),
            "z" => Ok(Self::Vector(VectorAxis::Z)),
            "xx" => Ok(Self::Tensor(TensorComponent::Xx)),
            "xy" => Ok(Self::Tensor(TensorComponent::Xy)),
            "xz" => Ok(Self::Tensor(TensorComponent::Xz)),
            "yx" => Ok(Self::Tensor(TensorComponent::Yx)),
            "yy" => Ok(Self::Tensor(TensorComponent::Yy)),
            "yz" => Ok(Self::Tensor(TensorComponent::Yz)),
            "zx" => Ok(Self::Tensor(TensorComponent::Zx)),
            "zy" => Ok(Self::Tensor(TensorComponent::Zy)),
            "zz" => Ok(Self::Tensor(TensorComponent::Zz)),
            other => other.parse::<usize>().map(Self::Index).map_err(|_| {
                format!(
                    "unknown component `{other}`; expected `x`, `y`, `z`, `magnitude`, \
                     a tensor component (`xx`..`zz`), or a zero-based index"
                )
            }),
        }
    }

    /// Resolves this component selector for an array with `component_count` components.
    ///
    /// Returns `None` if the selector is incompatible with `component_count` (for example
    /// requesting `"x"` or `"magnitude"` on a 1-component scalar array, or an out-of-bounds index).
    #[must_use]
    pub const fn resolve_for(self, component_count: ComponentCount) -> Option<ResolvedComponent> {
        match (self, component_count) {
            (Self::Index(0), ComponentCount::Scalar) => Some(ResolvedComponent::Component(0)),
            (
                Self::Magnitude | Self::Vector(_) | Self::Tensor(_) | Self::Index(_),
                ComponentCount::Scalar,
            ) => None,

            (
                Self::Magnitude,
                ComponentCount::Vector3
                | ComponentCount::SymmetricTensor6
                | ComponentCount::Tensor9,
            ) => Some(ResolvedComponent::Magnitude),
            (Self::Vector(axis), ComponentCount::Vector3) => {
                Some(ResolvedComponent::Component(axis.index()))
            }
            (Self::Vector(_), ComponentCount::SymmetricTensor6 | ComponentCount::Tensor9) => None,

            (Self::Tensor(tc), ComponentCount::SymmetricTensor6) => {
                // VTK 6-component symmetric tensor ordering: xx, yy, zz, xy, yz, xz.
                let idx = match tc {
                    TensorComponent::Xx => 0,
                    TensorComponent::Yy => 1,
                    TensorComponent::Zz => 2,
                    TensorComponent::Xy | TensorComponent::Yx => 3,
                    TensorComponent::Yz | TensorComponent::Zy => 4,
                    TensorComponent::Xz | TensorComponent::Zx => 5,
                };
                Some(ResolvedComponent::Component(idx))
            }
            (Self::Tensor(tc), ComponentCount::Tensor9) => {
                // Row-major 9-component tensor ordering: xx, xy, xz, yx, yy, yz, zx, zy, zz.
                let idx = match tc {
                    TensorComponent::Xx => 0,
                    TensorComponent::Xy => 1,
                    TensorComponent::Xz => 2,
                    TensorComponent::Yx => 3,
                    TensorComponent::Yy => 4,
                    TensorComponent::Yz => 5,
                    TensorComponent::Zx => 6,
                    TensorComponent::Zy => 7,
                    TensorComponent::Zz => 8,
                };
                Some(ResolvedComponent::Component(idx))
            }
            (Self::Tensor(_), ComponentCount::Vector3) => None,

            (Self::Index(idx), count) => {
                if idx < count.as_usize() {
                    Some(ResolvedComponent::Component(idx))
                } else {
                    None
                }
            }
        }
    }
}

impl fmt::Display for ComponentSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Magnitude => f.write_str("magnitude"),
            Self::Vector(axis) => write!(f, "{axis}"),
            Self::Tensor(tc) => write!(f, "{tc}"),
            Self::Index(idx) => write!(f, "{idx}"),
        }
    }
}

/// Parsed representation of an array reference string (`[point:|cell:]<array>[:<component>]`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParsedArrayRef {
    /// Optional explicit association (`point` or `cell`). When `None`, point data wins on ties.
    pub association: Option<ArrayAssociation>,
    /// Name of the attribute array on the dataset.
    pub name: String,
    /// Optional component selector. When `None`, scalars use component `0` and multi-component arrays use magnitude.
    pub component: Option<ComponentSelector>,
}

impl ParsedArrayRef {
    /// Parses an array reference string according to `docs/scene-format.md`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tetraglass_scene::{
    ///     ArrayAssociation, ComponentSelector, ParsedArrayRef, VectorAxis,
    /// };
    ///
    /// let r = ParsedArrayRef::parse("velocity:magnitude").unwrap();
    /// assert_eq!(r.association, None);
    /// assert_eq!(r.name, "velocity");
    /// assert_eq!(r.component, Some(ComponentSelector::Magnitude));
    ///
    /// let r2 = ParsedArrayRef::parse("cell:velocity:x").unwrap();
    /// assert_eq!(r2.association, Some(ArrayAssociation::Cell));
    /// assert_eq!(r2.name, "velocity");
    /// assert_eq!(r2.component, Some(ComponentSelector::Vector(VectorAxis::X)));
    /// ```
    pub fn parse(raw: &str) -> Result<Self, SceneError> {
        if raw.is_empty() {
            return Err(SceneError::InvalidArrayRef {
                reference: raw.to_string(),
                reason: "array reference must not be empty".to_string(),
            });
        }

        let (association, remainder) = if let Some(rest) = raw.strip_prefix("point:") {
            (Some(ArrayAssociation::Point), rest)
        } else if let Some(rest) = raw.strip_prefix("cell:") {
            (Some(ArrayAssociation::Cell), rest)
        } else {
            (None, raw)
        };

        if remainder.is_empty() {
            return Err(SceneError::InvalidArrayRef {
                reference: raw.to_string(),
                reason: "missing array name".to_string(),
            });
        }

        let mut parts = remainder.split(':');
        let name = parts.next().unwrap_or_default();
        if name.is_empty() {
            return Err(SceneError::InvalidArrayRef {
                reference: raw.to_string(),
                reason: "missing array name".to_string(),
            });
        }

        let component = match parts.next() {
            None => None,
            Some(comp_str) => {
                if comp_str.is_empty() {
                    return Err(SceneError::InvalidArrayRef {
                        reference: raw.to_string(),
                        reason: "empty component suffix after `:`".to_string(),
                    });
                }
                if parts.next().is_some() {
                    return Err(SceneError::InvalidArrayRef {
                        reference: raw.to_string(),
                        reason:
                            "too many `:` segments; expected `[point:|cell:]<array>[:<component>]`"
                                .to_string(),
                    });
                }
                let selector = ComponentSelector::parse(comp_str).map_err(|reason| {
                    SceneError::InvalidArrayRef {
                        reference: raw.to_string(),
                        reason,
                    }
                })?;
                Some(selector)
            }
        };

        Ok(Self {
            association,
            name: name.to_string(),
            component,
        })
    }

    /// Resolves the effective scalar extraction mode for an array with `component_count` components.
    ///
    /// When [`Self::component`] is `None`, scalar arrays resolve to `Component(0)` and
    /// multi-component arrays (`Vector3`, `SymmetricTensor6`, `Tensor9`) resolve to `Magnitude`.
    #[must_use]
    pub const fn resolve_component(
        &self,
        component_count: ComponentCount,
    ) -> Option<ResolvedComponent> {
        match self.component {
            Some(selector) => selector.resolve_for(component_count),
            None => match component_count {
                ComponentCount::Scalar => Some(ResolvedComponent::Component(0)),
                ComponentCount::Vector3
                | ComponentCount::SymmetricTensor6
                | ComponentCount::Tensor9 => Some(ResolvedComponent::Magnitude),
            },
        }
    }
}

impl FromStr for ParsedArrayRef {
    type Err = SceneError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl fmt::Display for ParsedArrayRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(assoc) = self.association {
            write!(f, "{assoc}:")?;
        }
        f.write_str(&self.name)?;
        if let Some(comp) = self.component {
            write!(f, ":{comp}")?;
        }
        Ok(())
    }
}
