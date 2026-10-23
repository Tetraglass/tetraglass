//! Data model: unstructured grids, cell types, typed arrays, time series. No GPU, no I/O.
//!
//! `tetraglass-core` sits at the bottom of the dependency graph and must never depend on
//! GPU crates (`wgpu`), GUI crates (`egui`), filesystem I/O, or native C libraries.
//!
//! See `docs/design-plan.md` in the repository for how this crate fits the architecture.

mod array;
mod cell;
mod error;
mod grid;
mod surface;

pub use array::{ArrayBuffer, Attributes, ComponentCount, DataArray, ScalarType, ValueRange};
pub use cell::{CellType, PolyhedronFaceIter, PolyhedronFaces};
pub use error::{CoreError, try_u32_from_i64, try_u32_from_u64, try_u32_from_usize};
pub use grid::{BoundingBox, UnstructuredGrid};
pub use surface::SurfaceMesh;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn sample_tet_points() -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 4.0],
        ]
    }

    #[test]
    fn u32_index_conversions_confirm_d013() {
        assert_eq!(try_u32_from_u64(0), Ok(0));
        assert_eq!(try_u32_from_u64(u64::from(u32::MAX)), Ok(u32::MAX));
        assert_eq!(
            try_u32_from_u64(u64::from(u32::MAX) + 1),
            Err(CoreError::IndexOverflow {
                value: u64::from(u32::MAX) + 1
            })
        );

        assert_eq!(try_u32_from_i64(123), Ok(123));
        assert_eq!(
            try_u32_from_i64(-5),
            Err(CoreError::NegativeIndex { value: -5 })
        );
        assert_eq!(
            try_u32_from_i64(i64::from(u32::MAX) + 10),
            Err(CoreError::IndexOverflow {
                value: u64::from(u32::MAX) + 10
            })
        );

        assert_eq!(try_u32_from_usize(99), Ok(99));
    }

    #[test]
    fn cell_type_metadata_and_lagrange_orders() {
        assert_eq!(CellType::Tetra.vtk_type_id(), 10);
        assert_eq!(CellType::Tetra.dimension(), 3);
        assert!(CellType::Tetra.is_linear());
        assert_eq!(CellType::Tetra.expected_node_count(), Some(4));

        assert_eq!(CellType::QuadraticTetra.vtk_type_id(), 24);
        assert!(!CellType::QuadraticTetra.is_linear());
        assert_eq!(CellType::QuadraticTetra.expected_node_count(), Some(10));

        let lagrange_hex_p3 = CellType::from_vtk_type_and_node_count(72, 64).unwrap();
        assert_eq!(lagrange_hex_p3, CellType::LagrangeHexahedron { order: 3 });
        assert_eq!(lagrange_hex_p3.expected_node_count(), Some(64));
        assert_eq!(lagrange_hex_p3.dimension(), 3);

        let lagrange_tet_p2 = CellType::from_vtk_type_and_node_count(71, 10).unwrap();
        assert_eq!(lagrange_tet_p2, CellType::LagrangeTetrahedron { order: 2 });

        let lagrange_wedge_p2 = CellType::from_vtk_type_and_node_count(73, 18).unwrap();
        assert_eq!(lagrange_wedge_p2, CellType::LagrangeWedge { order: 2 });

        assert_eq!(
            CellType::from_vtk_type_and_node_count(72, 50),
            Err(CoreError::InvalidLagrangeNodeCount {
                vtk_type: 72,
                node_count: 50,
            })
        );
        assert_eq!(
            CellType::from_vtk_type_and_node_count(255, 4),
            Err(CoreError::UnsupportedVtkCellType { vtk_type: 255 })
        );
    }

    #[test]
    fn construct_valid_linear_and_higher_order_grid() {
        let points = sample_tet_points();
        let grid = UnstructuredGrid::new(
            points,
            vec![CellType::Tetra, CellType::Triangle],
            vec![4, 7],
            vec![0, 1, 2, 3, 0, 1, 2],
            None,
        )
        .expect("valid mixed 3D/2D grid");

        assert_eq!(grid.num_points(), 4);
        assert_eq!(grid.num_cells(), 2);
        assert_eq!(grid.cell_connectivity(0), Some([0, 1, 2, 3].as_slice()));
        assert_eq!(grid.cell_connectivity(1), Some([0, 1, 2].as_slice()));
        assert_eq!(grid.cell_connectivity(2), None);

        let bounds = grid.bounds().expect("non-empty grid has bounds");
        assert_eq!(bounds.min, [0.0, 0.0, 0.0]);
        assert_eq!(bounds.max, [1.0, 2.0, 4.0]);
        assert_eq!(bounds.center(), [0.5, 1.0, 2.0]);
        assert_eq!(bounds.extents(), [1.0, 2.0, 4.0]);
        assert!((bounds.diagonal() - 21.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn construct_and_iterate_polyhedron_grid() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        // A tetrahedron represented as a 4-face polyhedron:
        // [4 faces, (3, 0,2,1), (3, 0,1,3), (3, 1,2,3), (3, 0,3,2)]
        let stream = vec![
            4, // num_faces
            3, 0, 2, 1, // face 0
            3, 0, 1, 3, // face 1
            3, 1, 2, 3, // face 2
            3, 0, 3, 2, // face 3
        ];
        let pf = PolyhedronFaces::new(vec![stream.len() as u32], stream);
        let grid = UnstructuredGrid::new(
            points,
            vec![CellType::Polyhedron],
            vec![4],
            vec![0, 1, 2, 3],
            Some(pf),
        )
        .expect("valid polyhedron");

        let faces: Vec<&[u32]> = grid
            .polyhedron_cell_faces(0)
            .expect("cell 0 is polyhedron")
            .collect();
        assert_eq!(
            faces,
            vec![
                &[0, 2, 1][..],
                &[0, 1, 3][..],
                &[1, 2, 3][..],
                &[0, 3, 2][..]
            ]
        );
    }

    #[test]
    fn rejects_inconsistent_grid_inputs() {
        let pts = sample_tet_points();

        // 1. cell_types.len() != offsets.len()
        assert_eq!(
            UnstructuredGrid::new(
                pts.clone(),
                vec![CellType::Tetra],
                vec![4, 8],
                vec![0, 1, 2, 3],
                None,
            ),
            Err(CoreError::CellCountMismatch {
                cell_types: 1,
                offsets: 2,
            })
        );

        // 2. Non-monotonic offsets
        assert_eq!(
            UnstructuredGrid::new(
                pts.clone(),
                vec![CellType::Line, CellType::Line],
                vec![2, 1],
                vec![0, 1, 2, 3],
                None,
            ),
            Err(CoreError::NonMonotonicOffsets {
                cell_index: 1,
                prev: 2,
                current: 1,
            })
        );

        // 3. Wrong node count for cell type
        assert_eq!(
            UnstructuredGrid::new(
                pts.clone(),
                vec![CellType::Tetra],
                vec![3],
                vec![0, 1, 2],
                None,
            ),
            Err(CoreError::InvalidCellNodeCount {
                cell_index: 0,
                cell_type: CellType::Tetra,
                expected: 4,
                actual: 3,
            })
        );

        // 4. Out-of-bounds point index
        assert_eq!(
            UnstructuredGrid::new(
                pts.clone(),
                vec![CellType::Tetra],
                vec![4],
                vec![0, 1, 2, 9],
                None,
            ),
            Err(CoreError::PointIndexOutOfBounds {
                cell_index: 0,
                point_index: 9,
                num_points: 4,
            })
        );

        // 5. Final offset does not match connectivity length
        assert_eq!(
            UnstructuredGrid::new(
                pts.clone(),
                vec![CellType::Line],
                vec![2],
                vec![0, 1, 2],
                None,
            ),
            Err(CoreError::OffsetOutOfBounds {
                cell_index: 0,
                offset: 2,
                connectivity_len: 3,
            })
        );

        // 6. Polyhedron without face stream
        assert_eq!(
            UnstructuredGrid::new(
                pts.clone(),
                vec![CellType::Polyhedron],
                vec![4],
                vec![0, 1, 2, 3],
                None,
            ),
            Err(CoreError::MissingPolyhedronFaces { cell_index: 0 })
        );

        // 7. Polyhedron with truncated face stream
        let bad_pf = PolyhedronFaces::new(vec![5], vec![4, 3, 0, 1, 2]);
        assert_eq!(
            UnstructuredGrid::new(
                pts,
                vec![CellType::Polyhedron],
                vec![4],
                vec![0, 1, 2, 3],
                Some(bad_pf),
            ),
            Err(CoreError::InvalidPolyhedronFaceStream {
                cell_index: 0,
                reason: "truncated face vertex count in stream",
            })
        );
    }

    #[test]
    fn data_array_ranges_and_arc_sharing() {
        // 3D vector array with 3 tuples: (3, 4, 0) -> mag 5; (0, 0, 0) -> mag 0; (-6, 8, 0) -> mag 10
        let vec_arr = DataArray::from_f64(
            ComponentCount::Vector3,
            vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, -6.0, 8.0, 0.0],
        )
        .expect("valid vector array");

        assert_eq!(vec_arr.num_tuples(), 3);
        assert_eq!(vec_arr.scalar_type(), ScalarType::F64);
        assert_eq!(
            vec_arr.component_range(0),
            Some(ValueRange {
                min: -6.0,
                max: 3.0
            })
        );
        assert_eq!(
            vec_arr.component_range(1),
            Some(ValueRange { min: 0.0, max: 8.0 })
        );
        assert_eq!(
            vec_arr.component_range(2),
            Some(ValueRange { min: 0.0, max: 0.0 })
        );
        assert_eq!(
            vec_arr.magnitude_range(),
            Some(ValueRange {
                min: 0.0,
                max: 10.0
            })
        );
        assert_eq!(vec_arr.tuple_magnitude(0), Some(5.0));
        assert_eq!(vec_arr.tuple_magnitude(2), Some(10.0));

        // Verify Arc sharing does not copy the underlying buffer on clone
        let cloned = vec_arr.clone();
        match (vec_arr.buffer(), cloned.buffer()) {
            (ArrayBuffer::F64(a), ArrayBuffer::F64(b)) => assert!(Arc::ptr_eq(a, b)),
            _ => panic!("expected F64 buffer"),
        }

        // Reject invalid component counts and mismatched buffer lengths
        assert_eq!(
            ComponentCount::try_from(2usize),
            Err(CoreError::InvalidComponentCount { count: 2 })
        );
        assert_eq!(
            DataArray::from_f32(ComponentCount::Vector3, vec![1.0, 2.0, 3.0, 4.0]),
            Err(CoreError::ArrayLengthMismatch {
                len: 4,
                components: 3,
            })
        );
    }

    #[test]
    fn attributes_enforce_tuple_counts() {
        let mut grid = UnstructuredGrid::new(
            sample_tet_points(),
            vec![CellType::Tetra],
            vec![4],
            vec![0, 1, 2, 3],
            None,
        )
        .unwrap();

        let point_scalars =
            DataArray::from_f64(ComponentCount::Scalar, vec![0.0, 1.0, 4.0, 12.0]).unwrap();
        grid.insert_point_data("linear", point_scalars).unwrap();

        let cell_ids = DataArray::from_i32(ComponentCount::Scalar, vec![0]).unwrap();
        grid.insert_cell_data("cell_id", cell_ids).unwrap();

        assert_eq!(grid.point_data().len(), 1);
        assert_eq!(grid.cell_data().len(), 1);
        assert_eq!(
            grid.point_data()
                .get("linear")
                .unwrap()
                .component_range(0)
                .unwrap(),
            ValueRange {
                min: 0.0,
                max: 12.0
            }
        );

        let wrong_len = DataArray::from_f64(ComponentCount::Scalar, vec![1.0, 2.0]).unwrap();
        assert_eq!(
            grid.insert_point_data("bad", wrong_len),
            Err(CoreError::AttributeTupleCountMismatch {
                name: "bad".to_string(),
                expected: 4,
                actual: 2,
            })
        );
    }
}
