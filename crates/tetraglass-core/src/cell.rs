//! Cell types (linear, quadratic, arbitrary-order Lagrange, and polyhedra) and polyhedron face streams.

use crate::error::CoreError;

/// Canonical VTK-ordered cell types supported by `UnstructuredGrid`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CellType {
    /// Single vertex (VTK type 1, 1 node, 0D).
    Vertex,
    /// Linear 2-node line segment (VTK type 3, 2 nodes, 1D).
    Line,
    /// Linear 3-node triangle (VTK type 5, 3 nodes, 2D).
    Triangle,
    /// Linear 4-node quadrilateral (VTK type 9, 4 nodes, 2D).
    Quad,
    /// Linear 4-node tetrahedron (VTK type 10, 4 nodes, 3D).
    Tetra,
    /// Linear 8-node hexahedron (VTK type 12, 8 nodes, 3D).
    Hexahedron,
    /// Linear 6-node triangular prism / wedge (VTK type 13, 6 nodes, 3D).
    Wedge,
    /// Linear 5-node pyramid with quad base (VTK type 14, 5 nodes, 3D).
    Pyramid,

    /// Quadratic 3-node edge (VTK type 21, 3 nodes, 1D).
    QuadraticEdge,
    /// Quadratic 6-node triangle (VTK type 22, 6 nodes, 2D).
    QuadraticTriangle,
    /// Quadratic 8-node serendipity quadrilateral (VTK type 23, 8 nodes, 2D).
    QuadraticQuad,
    /// Quadratic 10-node tetrahedron (VTK type 24, 10 nodes, 3D).
    QuadraticTetra,
    /// Quadratic 20-node serendipity hexahedron (VTK type 25, 20 nodes, 3D).
    QuadraticHexahedron,
    /// Quadratic 15-node serendipity wedge (VTK type 26, 15 nodes, 3D).
    QuadraticWedge,
    /// Quadratic 13-node serendipity pyramid (VTK type 27, 13 nodes, 3D).
    QuadraticPyramid,

    /// General polyhedron with explicit face stream (VTK type 42, variable node count >= 4, 3D).
    Polyhedron,

    /// Arbitrary-order Lagrange curve (VTK type 68, `order + 1` nodes, 1D).
    LagrangeCurve {
        /// Polynomial order (`>= 1`).
        order: u8,
    },
    /// Arbitrary-order Lagrange triangle (VTK type 69, `(order + 1)(order + 2)/2` nodes, 2D).
    LagrangeTriangle {
        /// Polynomial order (`>= 1`).
        order: u8,
    },
    /// Arbitrary-order Lagrange quadrilateral (VTK type 70, `(order + 1)^2` nodes, 2D).
    LagrangeQuadrilateral {
        /// Polynomial order (`>= 1`).
        order: u8,
    },
    /// Arbitrary-order Lagrange tetrahedron (VTK type 71, `(order + 1)(order + 2)(order + 3)/6` nodes, 3D).
    LagrangeTetrahedron {
        /// Polynomial order (`>= 1`).
        order: u8,
    },
    /// Arbitrary-order Lagrange hexahedron (VTK type 72, `(order + 1)^3` nodes, 3D).
    LagrangeHexahedron {
        /// Polynomial order (`>= 1`).
        order: u8,
    },
    /// Arbitrary-order Lagrange wedge (VTK type 73, `(order + 1)^2 (order + 2)/2` nodes, 3D).
    LagrangeWedge {
        /// Polynomial order (`>= 1`).
        order: u8,
    },
}

impl CellType {
    /// Returns the VTK numeric cell type identifier.
    #[must_use]
    pub const fn vtk_type_id(self) -> u8 {
        match self {
            Self::Vertex => 1,
            Self::Line => 3,
            Self::Triangle => 5,
            Self::Quad => 9,
            Self::Tetra => 10,
            Self::Hexahedron => 12,
            Self::Wedge => 13,
            Self::Pyramid => 14,
            Self::QuadraticEdge => 21,
            Self::QuadraticTriangle => 22,
            Self::QuadraticQuad => 23,
            Self::QuadraticTetra => 24,
            Self::QuadraticHexahedron => 25,
            Self::QuadraticWedge => 26,
            Self::QuadraticPyramid => 27,
            Self::Polyhedron => 42,
            Self::LagrangeCurve { .. } => 68,
            Self::LagrangeTriangle { .. } => 69,
            Self::LagrangeQuadrilateral { .. } => 70,
            Self::LagrangeTetrahedron { .. } => 71,
            Self::LagrangeHexahedron { .. } => 72,
            Self::LagrangeWedge { .. } => 73,
        }
    }

    /// Returns the topological dimension (`0..=3`) of the cell.
    #[must_use]
    pub const fn dimension(self) -> u8 {
        match self {
            Self::Vertex => 0,
            Self::Line | Self::QuadraticEdge | Self::LagrangeCurve { .. } => 1,
            Self::Triangle
            | Self::Quad
            | Self::QuadraticTriangle
            | Self::QuadraticQuad
            | Self::LagrangeTriangle { .. }
            | Self::LagrangeQuadrilateral { .. } => 2,
            Self::Tetra
            | Self::Hexahedron
            | Self::Wedge
            | Self::Pyramid
            | Self::QuadraticTetra
            | Self::QuadraticHexahedron
            | Self::QuadraticWedge
            | Self::QuadraticPyramid
            | Self::Polyhedron
            | Self::LagrangeTetrahedron { .. }
            | Self::LagrangeHexahedron { .. }
            | Self::LagrangeWedge { .. } => 3,
        }
    }

    /// Returns `true` if the cell is a first-order linear element.
    #[must_use]
    pub const fn is_linear(self) -> bool {
        matches!(
            self,
            Self::Vertex
                | Self::Line
                | Self::Triangle
                | Self::Quad
                | Self::Tetra
                | Self::Hexahedron
                | Self::Wedge
                | Self::Pyramid
        )
    }

    /// Returns the required number of connectivity nodes for this cell type, or `None` for
    /// [`CellType::Polyhedron`] (whose node count varies per cell).
    #[must_use]
    pub const fn expected_node_count(self) -> Option<usize> {
        match self {
            Self::Vertex => Some(1),
            Self::Line => Some(2),
            Self::Triangle => Some(3),
            Self::Quad => Some(4),
            Self::Tetra => Some(4),
            Self::Hexahedron => Some(8),
            Self::Wedge => Some(6),
            Self::Pyramid => Some(5),
            Self::QuadraticEdge => Some(3),
            Self::QuadraticTriangle => Some(6),
            Self::QuadraticQuad => Some(8),
            Self::QuadraticTetra => Some(10),
            Self::QuadraticHexahedron => Some(20),
            Self::QuadraticWedge => Some(15),
            Self::QuadraticPyramid => Some(13),
            Self::Polyhedron => None,
            Self::LagrangeCurve { order } => {
                if order == 0 {
                    Some(0)
                } else {
                    Some((order as usize) + 1)
                }
            }
            Self::LagrangeTriangle { order } => {
                if order == 0 {
                    Some(0)
                } else {
                    let n = (order as usize) + 1;
                    Some(n * (n + 1) / 2)
                }
            }
            Self::LagrangeQuadrilateral { order } => {
                if order == 0 {
                    Some(0)
                } else {
                    let n = (order as usize) + 1;
                    Some(n * n)
                }
            }
            Self::LagrangeTetrahedron { order } => {
                if order == 0 {
                    Some(0)
                } else {
                    let n = (order as usize) + 1;
                    Some(n * (n + 1) * (n + 2) / 6)
                }
            }
            Self::LagrangeHexahedron { order } => {
                if order == 0 {
                    Some(0)
                } else {
                    let n = (order as usize) + 1;
                    Some(n * n * n)
                }
            }
            Self::LagrangeWedge { order } => {
                if order == 0 {
                    Some(0)
                } else {
                    let n = (order as usize) + 1;
                    Some(n * (n + 1) / 2 * n)
                }
            }
        }
    }

    /// Constructs a [`CellType`] from a raw VTK type ID and the cell's node count, inferring
    /// the polynomial order for Lagrange cell types (`68..=73`).
    ///
    /// ```
    /// use tetraglass_core::CellType;
    ///
    /// assert_eq!(
    ///     CellType::from_vtk_type_and_node_count(72, 64),
    ///     Ok(CellType::LagrangeHexahedron { order: 3 })
    /// );
    /// ```
    pub fn from_vtk_type_and_node_count(
        vtk_type: u8,
        node_count: usize,
    ) -> Result<Self, CoreError> {
        let find_order = |f: fn(u8) -> usize| -> Result<u8, CoreError> {
            for order in 1..=16u8 {
                let count = f(order);
                if count == node_count {
                    return Ok(order);
                }
                if count > node_count {
                    break;
                }
            }
            Err(CoreError::InvalidLagrangeNodeCount {
                vtk_type,
                node_count,
            })
        };

        match vtk_type {
            1 => Ok(Self::Vertex),
            3 => Ok(Self::Line),
            5 => Ok(Self::Triangle),
            9 => Ok(Self::Quad),
            10 => Ok(Self::Tetra),
            12 => Ok(Self::Hexahedron),
            13 => Ok(Self::Wedge),
            14 => Ok(Self::Pyramid),
            21 => Ok(Self::QuadraticEdge),
            22 => Ok(Self::QuadraticTriangle),
            23 => Ok(Self::QuadraticQuad),
            24 => Ok(Self::QuadraticTetra),
            25 => Ok(Self::QuadraticHexahedron),
            26 => Ok(Self::QuadraticWedge),
            27 => Ok(Self::QuadraticPyramid),
            42 => Ok(Self::Polyhedron),
            68 => Ok(Self::LagrangeCurve {
                order: find_order(|p| usize::from(p) + 1)?,
            }),
            69 => Ok(Self::LagrangeTriangle {
                order: find_order(|p| {
                    let n = usize::from(p) + 1;
                    n * (n + 1) / 2
                })?,
            }),
            70 => Ok(Self::LagrangeQuadrilateral {
                order: find_order(|p| {
                    let n = usize::from(p) + 1;
                    n * n
                })?,
            }),
            71 => Ok(Self::LagrangeTetrahedron {
                order: find_order(|p| {
                    let n = usize::from(p) + 1;
                    n * (n + 1) * (n + 2) / 6
                })?,
            }),
            72 => Ok(Self::LagrangeHexahedron {
                order: find_order(|p| {
                    let n = usize::from(p) + 1;
                    n * n * n
                })?,
            }),
            73 => Ok(Self::LagrangeWedge {
                order: find_order(|p| {
                    let n = usize::from(p) + 1;
                    n * (n + 1) / 2 * n
                })?,
            }),
            _ => Err(CoreError::UnsupportedVtkCellType { vtk_type }),
        }
    }
}

/// VTK-style polyhedron face stream storage for an [`UnstructuredGrid`](crate::UnstructuredGrid).
///
/// For each cell `i` in the grid, its slice of `faces` is `start..end` where
/// `start = if i == 0 { 0 } else { face_offsets[i - 1] }` and `end = face_offsets[i]`.
/// Non-polyhedral cells have an empty slice (`start == end`).
/// Each polyhedral cell's slice encodes `[num_faces, npts_0, v0, v1, ..., npts_1, v0, v1, ...]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolyhedronFaces {
    face_offsets: Vec<u32>,
    faces: Vec<u32>,
}

impl PolyhedronFaces {
    /// Creates a new polyhedron face-stream container without validating against a grid.
    ///
    /// Full structural validation is performed when constructing an
    /// [`UnstructuredGrid`](crate::UnstructuredGrid).
    #[must_use]
    pub fn new(face_offsets: Vec<u32>, faces: Vec<u32>) -> Self {
        Self {
            face_offsets,
            faces,
        }
    }

    /// Per-cell end offsets into [`PolyhedronFaces::faces`].
    #[must_use]
    pub fn face_offsets(&self) -> &[u32] {
        &self.face_offsets
    }

    /// Flat face stream `[num_faces, npts_0, v0, ..., npts_1, v0, ...]`.
    #[must_use]
    pub fn faces(&self) -> &[u32] {
        &self.faces
    }

    /// Returns the raw face-stream slice for `cell_index`, if `cell_index < self.face_offsets.len()`.
    #[must_use]
    pub fn cell_stream(&self, cell_index: usize) -> Option<&[u32]> {
        let end = *self.face_offsets.get(cell_index)? as usize;
        let start = if cell_index == 0 {
            0
        } else {
            self.face_offsets[cell_index - 1] as usize
        };
        self.faces.get(start..end)
    }

    /// Returns an iterator over the polygon face vertex slices (`&[u32]`) for `cell_index`.
    #[must_use]
    pub fn cell_faces(&self, cell_index: usize) -> Option<PolyhedronFaceIter<'_>> {
        let stream = self.cell_stream(cell_index)?;
        let (&num_faces, rest) = stream.split_first()?;
        Some(PolyhedronFaceIter {
            remaining_faces: num_faces as usize,
            rest,
        })
    }

    /// Validates the face offsets and streams against `cell_types` and `num_points`.
    pub(crate) fn validate(
        &self,
        cell_types: &[CellType],
        num_points: usize,
    ) -> Result<(), CoreError> {
        if self.face_offsets.len() != cell_types.len() {
            return Err(CoreError::PolyhedronFaceOffsetsMismatch {
                face_offsets: self.face_offsets.len(),
                num_cells: cell_types.len(),
            });
        }

        let mut prev_offset = 0u32;
        for (cell_index, (&end_offset, &cell_type)) in
            self.face_offsets.iter().zip(cell_types.iter()).enumerate()
        {
            if end_offset < prev_offset {
                return Err(CoreError::InvalidPolyhedronFaceStream {
                    cell_index,
                    reason: "face_offsets are not monotonic",
                });
            }
            let start = prev_offset as usize;
            let end = end_offset as usize;
            if end > self.faces.len() {
                return Err(CoreError::InvalidPolyhedronFaceStream {
                    cell_index,
                    reason: "face_offset exceeds faces buffer length",
                });
            }
            let slice = &self.faces[start..end];
            if cell_type == CellType::Polyhedron {
                let Some((&num_faces, mut rest)) = slice.split_first() else {
                    return Err(CoreError::InvalidPolyhedronFaceStream {
                        cell_index,
                        reason: "polyhedron cell has empty face stream",
                    });
                };
                if num_faces < 4 {
                    return Err(CoreError::InvalidPolyhedronFaceStream {
                        cell_index,
                        reason: "polyhedron cell must have at least 4 faces",
                    });
                }
                for _ in 0..num_faces {
                    let Some((&npts, after_len)) = rest.split_first() else {
                        return Err(CoreError::InvalidPolyhedronFaceStream {
                            cell_index,
                            reason: "truncated face vertex count in stream",
                        });
                    };
                    if npts < 3 {
                        return Err(CoreError::InvalidPolyhedronFaceStream {
                            cell_index,
                            reason: "polygon face must have at least 3 vertices",
                        });
                    }
                    let npts_usize = npts as usize;
                    if after_len.len() < npts_usize {
                        return Err(CoreError::InvalidPolyhedronFaceStream {
                            cell_index,
                            reason: "truncated face vertex indices in stream",
                        });
                    }
                    let (face_verts, tail) = after_len.split_at(npts_usize);
                    for &pt in face_verts {
                        if (pt as usize) >= num_points {
                            return Err(CoreError::PointIndexOutOfBounds {
                                cell_index,
                                point_index: pt,
                                num_points,
                            });
                        }
                    }
                    rest = tail;
                }
                if !rest.is_empty() {
                    return Err(CoreError::InvalidPolyhedronFaceStream {
                        cell_index,
                        reason: "trailing words after declared faces in stream",
                    });
                }
            } else if !slice.is_empty() {
                return Err(CoreError::InvalidPolyhedronFaceStream {
                    cell_index,
                    reason: "non-polyhedron cell has non-empty face stream",
                });
            }
            prev_offset = end_offset;
        }

        if prev_offset as usize != self.faces.len() {
            return Err(CoreError::InvalidPolyhedronFaceStream {
                cell_index: cell_types.len().saturating_sub(1),
                reason: "final face_offset does not equal faces buffer length",
            });
        }

        Ok(())
    }
}

/// Iterator over the polygon faces (`&[u32]`) of a single polyhedral cell.
#[derive(Debug, Clone)]
pub struct PolyhedronFaceIter<'a> {
    remaining_faces: usize,
    rest: &'a [u32],
}

impl<'a> Iterator for PolyhedronFaceIter<'a> {
    type Item = &'a [u32];

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_faces == 0 {
            return None;
        }
        let (&npts, after_len) = self.rest.split_first()?;
        let (face, tail) = after_len.split_at_checked(npts as usize)?;
        self.remaining_faces -= 1;
        self.rest = tail;
        Some(face)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining_faces, Some(self.remaining_faces))
    }
}

impl ExactSizeIterator for PolyhedronFaceIter<'_> {}
