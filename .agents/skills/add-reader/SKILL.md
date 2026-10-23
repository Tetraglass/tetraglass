---
name: add-reader
description: Procedure for adding or changing a mesh file reader or writer in the io crate (VTK .vtu/.vtk/.pvd, Gmsh .msh and future formats), covering byte-buffer input, cell type mapping, higher-order node ordering, arrays, time series, error handling and tests. Use whenever a task touches file formats, parsing, import or export of mesh data.
---

# Add or change a reader

## 1. Input and output

- Read from `&[u8]` or `Read + Seek`, never from a path. A `from_path` helper may exist behind `#[cfg(not(target_arch = "wasm32"))]`.
- Produce the core crate's types (`UnstructuredGrid`, arrays, time series). Format-specific structs stay private to the io crate.
- If a third-party crate is used (vtkio or mshio, depending on the M0.3 decision), wrap it behind the project's own reader function so it can be replaced later.

## 2. Cells

- Map every supported element type to the core cell type, including order for Lagrange cells and face streams for polyhedra.
- Node ordering in core follows VTK. For other formats, write one permutation table per element type and order, with a comment linking the format's documentation for that element.
- Unsupported element types return a clear error naming the type; never skip them silently.

## 3. Data

- Point and cell arrays keep their names and component counts. Multi-component arrays stay together (a 3-component velocity is one array, not three).
- Time series: `.pvd` collections and Gmsh files with several steps become a lazily loaded time series; only the requested step is parsed.
- Compute the cached value ranges that `inspect` and colormap defaults rely on.

## 4. Robustness

User files are untrusted. Truncated files, wrong counts, out-of-range indices and unknown encodings must return errors with a byte offset or line number, never panic. Use checked arithmetic for every size computed from file contents.

## 5. Test

- One test file per cell type and order from `testdata/` (create missing ones with `/make-test-data`).
- A node-ordering test per higher-order element: evaluate a known field at the nodes and check it lands where the reference element says.
- Round trip through the `.vtu` writer for every test file.
- Corruption tests: truncate each test file at several offsets and assert an error, not a panic.

## 6. Integrate

- Register the format for detection by extension and, where the format allows, by magic bytes.
- Make sure `inspect` (M1 onward) reports everything the reader exposes.
- Document supported versions, encodings and element types in the io crate's rustdoc.
