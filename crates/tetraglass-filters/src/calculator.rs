//! `calculator` mathematical expression filter (`M6.4`, `D-005`, `D-008`, `D-014`).
//!
//! Parses and evaluates scalar and 3-vector mathematical expressions over point or cell
//! [`Attributes`] and 3D spatial coordinates (`x`, `y`, `z`, `coords`), producing a derived
//! `f64` [`DataArray`] (`ComponentCount::Scalar` or `ComponentCount::Vector3`) and inserting it
//! into a [`TetView`] or [`UnstructuredGrid`].

use tetraglass_core::{Attributes, ComponentCount, DataArray, TetView, UnstructuredGrid};

use crate::linearize::{LinearizeParams, linearize};
use crate::surface::FilterError;

/// Default number of tuples processed per parallel worker chunk in [`calculator`].
pub const DEFAULT_CALCULATOR_CHUNK_SIZE: usize = 4096;

/// Target attribute table association for [`CalculatorParams`] (`M6.4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CalculatorAssociation {
    /// Evaluate per vertex over [`TetView::point_data`] (or [`UnstructuredGrid::point_data`])
    /// and insert the resulting [`DataArray`] into `point_data`.
    #[default]
    Point,
    /// Evaluate per cell over [`TetView::cell_data`] (or [`UnstructuredGrid::cell_data`])
    /// using cell centroid coordinates for `x`, `y`, `z`, `coords`, and insert into `cell_data`.
    Cell,
}

/// Parameters for the [`calculator`] filter (`M6.4`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalculatorParams {
    /// Name of the output [`DataArray`] to insert into `point_data` or `cell_data`.
    pub result_name: String,
    /// Mathematical expression to evaluate (for example `"0.5 * mag(velocity)^2"` or `"[-y, x, z]"`).
    pub expression: String,
    /// Attribute association (`Point` default, or `Cell`).
    pub association: CalculatorAssociation,
}

impl CalculatorParams {
    /// Creates [`CalculatorParams`] evaluating `expression` over **point** data and storing the
    /// result under `result_name`.
    #[must_use]
    pub fn new(result_name: impl Into<String>, expression: impl Into<String>) -> Self {
        Self {
            result_name: result_name.into(),
            expression: expression.into(),
            association: CalculatorAssociation::Point,
        }
    }

    /// Creates [`CalculatorParams`] evaluating `expression` over **cell** data and storing the
    /// result under `result_name`.
    #[must_use]
    pub fn cell(result_name: impl Into<String>, expression: impl Into<String>) -> Self {
        Self {
            result_name: result_name.into(),
            expression: expression.into(),
            association: CalculatorAssociation::Cell,
        }
    }

    /// Sets the [`CalculatorAssociation`] (`Point` or `Cell`).
    #[must_use]
    pub const fn with_association(mut self, association: CalculatorAssociation) -> Self {
        self.association = association;
        self
    }
}

/// Evaluates `params.expression` over `view` and returns a new [`TetView`] with the derived
/// [`DataArray`] inserted under `params.result_name` (`M6.4`, `D-005`).
///
/// # Examples
///
/// ```
/// use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
/// use tetraglass_filters::{CalculatorParams, calculator, linearize_default};
///
/// let grid = SyntheticGridSpec::new(2, 2, 2, SyntheticCellLayout::Hexahedra).generate()?;
/// let view = linearize_default(&grid)?;
///
/// // Compute dynamic pressure q = 0.5 * mag(velocity)^2 and a swirl vector field
/// let with_q = calculator(&view, &CalculatorParams::new("q", "0.5 * mag(velocity)^2"))?;
/// assert!(with_q.point_data().get("q").is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn calculator(view: &TetView, params: &CalculatorParams) -> Result<TetView, FilterError> {
    calculator_with_chunk_size(view, params, DEFAULT_CALCULATOR_CHUNK_SIZE)
}

/// Evaluates `params.expression` over `view` using an explicit parallel `chunk_size`.
pub fn calculator_with_chunk_size(
    view: &TetView,
    params: &CalculatorParams,
    chunk_size: usize,
) -> Result<TetView, FilterError> {
    let array = evaluate_calculator_with_chunk_size(view, params, chunk_size)?;
    let mut out = view.clone();
    match params.association {
        CalculatorAssociation::Point => {
            out.insert_point_data(params.result_name.trim(), array)?;
        }
        CalculatorAssociation::Cell => {
            out.insert_cell_data(params.result_name.trim(), array)?;
        }
    }
    Ok(out)
}

/// Evaluates `params.expression` over `view` and returns the resulting [`DataArray`] without
/// cloning the [`TetView`].
pub fn evaluate_calculator(
    view: &TetView,
    params: &CalculatorParams,
) -> Result<DataArray, FilterError> {
    evaluate_calculator_with_chunk_size(view, params, DEFAULT_CALCULATOR_CHUNK_SIZE)
}

/// Evaluates `params.expression` over `view` with an explicit parallel `chunk_size` and returns
/// the resulting [`DataArray`].
pub fn evaluate_calculator_with_chunk_size(
    view: &TetView,
    params: &CalculatorParams,
    chunk_size: usize,
) -> Result<DataArray, FilterError> {
    if params.result_name.trim().is_empty() {
        return Err(FilterError::InvalidCalculatorResultName);
    }
    match params.association {
        CalculatorAssociation::Point => {
            evaluate_on_table(view.points(), view.point_data(), params, chunk_size)
        }
        CalculatorAssociation::Cell => {
            let centroids = compute_tet_centroids(view);
            evaluate_on_table(&centroids, view.cell_data(), params, chunk_size)
        }
    }
}

/// Linearizes `grid` with `linearize_params` and evaluates [`calculator`] (`M6.4`, `D-005`).
pub fn calculator_grid(
    grid: &UnstructuredGrid,
    params: &CalculatorParams,
    linearize_params: &LinearizeParams,
) -> Result<TetView, FilterError> {
    let view = linearize(grid, linearize_params)?;
    calculator(&view, params)
}

/// Evaluates `params.expression` directly on an [`UnstructuredGrid`] and inserts the derived
/// [`DataArray`] into `point_data` or `cell_data` (`M6.4`).
pub fn calculator_unstructured_grid(
    grid: &UnstructuredGrid,
    params: &CalculatorParams,
) -> Result<UnstructuredGrid, FilterError> {
    if params.result_name.trim().is_empty() {
        return Err(FilterError::InvalidCalculatorResultName);
    }
    let array = match params.association {
        CalculatorAssociation::Point => evaluate_on_table(
            grid.points(),
            grid.point_data(),
            params,
            DEFAULT_CALCULATOR_CHUNK_SIZE,
        )?,
        CalculatorAssociation::Cell => {
            let centroids = compute_grid_cell_centroids(grid);
            evaluate_on_table(
                &centroids,
                grid.cell_data(),
                params,
                DEFAULT_CALCULATOR_CHUNK_SIZE,
            )?
        }
    };
    let mut out = grid.clone();
    match params.association {
        CalculatorAssociation::Point => {
            out.insert_point_data(params.result_name.trim(), array)?;
        }
        CalculatorAssociation::Cell => {
            out.insert_cell_data(params.result_name.trim(), array)?;
        }
    }
    Ok(out)
}

fn compute_tet_centroids(view: &TetView) -> Vec<[f64; 3]> {
    let pts = view.points();
    view.tets()
        .iter()
        .map(|&[i0, i1, i2, i3]| {
            let p0 = pts[i0 as usize];
            let p1 = pts[i1 as usize];
            let p2 = pts[i2 as usize];
            let p3 = pts[i3 as usize];
            [
                0.25 * (p0[0] + p1[0] + p2[0] + p3[0]),
                0.25 * (p0[1] + p1[1] + p2[1] + p3[1]),
                0.25 * (p0[2] + p1[2] + p2[2] + p3[2]),
            ]
        })
        .collect()
}

fn compute_grid_cell_centroids(grid: &UnstructuredGrid) -> Vec<[f64; 3]> {
    let pts = grid.points();
    (0..grid.num_cells())
        .map(|cell_idx| {
            let Some(conn) = grid.cell_connectivity(cell_idx) else {
                return [0.0; 3];
            };
            if conn.is_empty() {
                return [0.0; 3];
            }
            let mut sum = [0.0; 3];
            for &vid in conn {
                let p = pts[vid as usize];
                sum[0] += p[0];
                sum[1] += p[1];
                sum[2] += p[2];
            }
            let inv = 1.0 / (conn.len() as f64);
            [sum[0] * inv, sum[1] * inv, sum[2] * inv]
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueKind {
    Scalar,
    Vector3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CalcValue {
    Scalar(f64),
    Vector3([f64; 3]),
}

impl CalcValue {
    #[inline]
    fn as_scalar(self) -> f64 {
        match self {
            Self::Scalar(s) => s,
            Self::Vector3([x, y, z]) => (x * x + y * y + z * z).sqrt(),
        }
    }

    #[inline]
    fn as_vector3(self) -> [f64; 3] {
        match self {
            Self::Scalar(s) => [s, s, s],
            Self::Vector3(v) => v,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnaryOp {
    Plus,
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, PartialEq)]
enum CompAccess {
    Index(usize),
    Named(String),
    Magnitude,
}

#[derive(Debug, Clone, PartialEq)]
enum AstExpr {
    Literal(f64),
    VectorLiteral([Box<AstExpr>; 3]),
    Var {
        name: String,
        comp: Option<CompAccess>,
    },
    Unary {
        op: UnaryOp,
        expr: Box<AstExpr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<AstExpr>,
        rhs: Box<AstExpr>,
    },
    Call {
        func: String,
        args: Vec<AstExpr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnaryFunc {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Sqrt,
    Cbrt,
    Abs,
    Exp,
    Ln,
    Log10,
    Log2,
    Floor,
    Ceil,
    Round,
    Sign,
}

impl UnaryFunc {
    #[inline]
    fn eval_f64(self, x: f64) -> f64 {
        match self {
            Self::Sin => x.sin(),
            Self::Cos => x.cos(),
            Self::Tan => x.tan(),
            Self::Asin => x.asin(),
            Self::Acos => x.acos(),
            Self::Atan => x.atan(),
            Self::Sinh => x.sinh(),
            Self::Cosh => x.cosh(),
            Self::Tanh => x.tanh(),
            Self::Sqrt => x.sqrt(),
            Self::Cbrt => x.cbrt(),
            Self::Abs => x.abs(),
            Self::Exp => x.exp(),
            Self::Ln => x.ln(),
            Self::Log10 => x.log10(),
            Self::Log2 => x.log2(),
            Self::Floor => x.floor(),
            Self::Ceil => x.ceil(),
            Self::Round => x.round(),
            Self::Sign => {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinaryScalarFunc {
    Min,
    Max,
    Pow,
    Atan2,
    Hypot,
}

impl BinaryScalarFunc {
    #[inline]
    fn eval_f64(self, a: f64, b: f64) -> f64 {
        match self {
            Self::Min => a.min(b),
            Self::Max => a.max(b),
            Self::Pow => a.powf(b),
            Self::Atan2 => a.atan2(b),
            Self::Hypot => a.hypot(b),
        }
    }
}

enum BoundExpr<'a> {
    ScalarConst(f64),
    VectorConst([f64; 3]),
    CoordComponent(usize),
    CoordMagnitude,
    CoordsVector,
    ArrayScalar {
        array: &'a DataArray,
        component: usize,
    },
    ArrayMagnitude {
        array: &'a DataArray,
    },
    ArrayVector3 {
        array: &'a DataArray,
    },
    VectorConstruct([Box<BoundExpr<'a>>; 3]),
    Unary {
        op: UnaryOp,
        expr: Box<BoundExpr<'a>>,
        kind: ValueKind,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<BoundExpr<'a>>,
        rhs: Box<BoundExpr<'a>>,
        kind: ValueKind,
    },
    UnaryMath {
        func: UnaryFunc,
        arg: Box<BoundExpr<'a>>,
        kind: ValueKind,
    },
    BinaryMath {
        func: BinaryScalarFunc,
        a: Box<BoundExpr<'a>>,
        b: Box<BoundExpr<'a>>,
    },
    Clamp {
        x: Box<BoundExpr<'a>>,
        lo: Box<BoundExpr<'a>>,
        hi: Box<BoundExpr<'a>>,
    },
    Lerp {
        a: Box<BoundExpr<'a>>,
        b: Box<BoundExpr<'a>>,
        t: Box<BoundExpr<'a>>,
        kind: ValueKind,
    },
    Mag(Box<BoundExpr<'a>>),
    Mag2(Box<BoundExpr<'a>>),
    Normalize(Box<BoundExpr<'a>>),
    Dot(Box<BoundExpr<'a>>, Box<BoundExpr<'a>>),
    Cross(Box<BoundExpr<'a>>, Box<BoundExpr<'a>>),
}

impl<'a> BoundExpr<'a> {
    const fn kind(&self) -> ValueKind {
        match self {
            Self::ScalarConst(_)
            | Self::CoordComponent(_)
            | Self::CoordMagnitude
            | Self::ArrayScalar { .. }
            | Self::ArrayMagnitude { .. }
            | Self::BinaryMath { .. }
            | Self::Clamp { .. }
            | Self::Mag(_)
            | Self::Mag2(_)
            | Self::Dot(_, _) => ValueKind::Scalar,
            Self::VectorConst(_)
            | Self::CoordsVector
            | Self::ArrayVector3 { .. }
            | Self::VectorConstruct(_)
            | Self::Normalize(_)
            | Self::Cross(_, _) => ValueKind::Vector3,
            Self::Unary { kind, .. }
            | Self::Binary { kind, .. }
            | Self::UnaryMath { kind, .. }
            | Self::Lerp { kind, .. } => *kind,
        }
    }

    #[inline]
    fn eval(&self, tuple_idx: usize, coord: [f64; 3]) -> CalcValue {
        match self {
            Self::ScalarConst(c) => CalcValue::Scalar(*c),
            Self::VectorConst(v) => CalcValue::Vector3(*v),
            Self::CoordComponent(axis) => CalcValue::Scalar(coord[*axis]),
            Self::CoordMagnitude => {
                let [x, y, z] = coord;
                CalcValue::Scalar((x * x + y * y + z * z).sqrt())
            }
            Self::CoordsVector => CalcValue::Vector3(coord),
            Self::ArrayScalar { array, component } => {
                CalcValue::Scalar(array.value_as_f64(tuple_idx, *component).unwrap_or(0.0))
            }
            Self::ArrayMagnitude { array } => {
                CalcValue::Scalar(array.tuple_magnitude(tuple_idx).unwrap_or(0.0))
            }
            Self::ArrayVector3 { array } => {
                let buf = array.buffer();
                let base = tuple_idx * 3;
                CalcValue::Vector3([
                    buf.get_as_f64(base).unwrap_or(0.0),
                    buf.get_as_f64(base + 1).unwrap_or(0.0),
                    buf.get_as_f64(base + 2).unwrap_or(0.0),
                ])
            }
            Self::VectorConstruct([ex, ey, ez]) => CalcValue::Vector3([
                ex.eval(tuple_idx, coord).as_scalar(),
                ey.eval(tuple_idx, coord).as_scalar(),
                ez.eval(tuple_idx, coord).as_scalar(),
            ]),
            Self::Unary { op, expr, .. } => {
                let v = expr.eval(tuple_idx, coord);
                match (op, v) {
                    (UnaryOp::Plus, val) => val,
                    (UnaryOp::Neg, CalcValue::Scalar(s)) => CalcValue::Scalar(-s),
                    (UnaryOp::Neg, CalcValue::Vector3([x, y, z])) => {
                        CalcValue::Vector3([-x, -y, -z])
                    }
                }
            }
            Self::Binary { op, lhs, rhs, kind } => {
                let l = lhs.eval(tuple_idx, coord);
                let r = rhs.eval(tuple_idx, coord);
                match kind {
                    ValueKind::Scalar => {
                        let a = l.as_scalar();
                        let b = r.as_scalar();
                        let out = match op {
                            BinaryOp::Add => a + b,
                            BinaryOp::Sub => a - b,
                            BinaryOp::Mul => a * b,
                            BinaryOp::Div => a / b,
                            BinaryOp::Pow => a.powf(b),
                        };
                        CalcValue::Scalar(out)
                    }
                    ValueKind::Vector3 => {
                        let [ax, ay, az] = l.as_vector3();
                        let [bx, by, bz] = r.as_vector3();
                        let out = match op {
                            BinaryOp::Add => [ax + bx, ay + by, az + bz],
                            BinaryOp::Sub => [ax - bx, ay - by, az - bz],
                            BinaryOp::Mul => [ax * bx, ay * by, az * bz],
                            BinaryOp::Div => [ax / bx, ay / by, az / bz],
                            BinaryOp::Pow => [ax.powf(bx), ay.powf(by), az.powf(bz)],
                        };
                        CalcValue::Vector3(out)
                    }
                }
            }
            Self::UnaryMath { func, arg, .. } => match arg.eval(tuple_idx, coord) {
                CalcValue::Scalar(s) => CalcValue::Scalar(func.eval_f64(s)),
                CalcValue::Vector3([x, y, z]) => {
                    CalcValue::Vector3([func.eval_f64(x), func.eval_f64(y), func.eval_f64(z)])
                }
            },
            Self::BinaryMath { func, a, b } => {
                let va = a.eval(tuple_idx, coord).as_scalar();
                let vb = b.eval(tuple_idx, coord).as_scalar();
                CalcValue::Scalar(func.eval_f64(va, vb))
            }
            Self::Clamp { x, lo, hi } => {
                let vx = x.eval(tuple_idx, coord).as_scalar();
                let vlo = lo.eval(tuple_idx, coord).as_scalar();
                let vhi = hi.eval(tuple_idx, coord).as_scalar();
                CalcValue::Scalar(vx.clamp(vlo, vhi))
            }
            Self::Lerp { a, b, t, kind } => {
                let vt = t.eval(tuple_idx, coord).as_scalar();
                match kind {
                    ValueKind::Scalar => {
                        let va = a.eval(tuple_idx, coord).as_scalar();
                        let vb = b.eval(tuple_idx, coord).as_scalar();
                        CalcValue::Scalar(va + vt * (vb - va))
                    }
                    ValueKind::Vector3 => {
                        let [ax, ay, az] = a.eval(tuple_idx, coord).as_vector3();
                        let [bx, by, bz] = b.eval(tuple_idx, coord).as_vector3();
                        CalcValue::Vector3([
                            ax + vt * (bx - ax),
                            ay + vt * (by - ay),
                            az + vt * (bz - az),
                        ])
                    }
                }
            }
            Self::Mag(inner) => match inner.eval(tuple_idx, coord) {
                CalcValue::Scalar(s) => CalcValue::Scalar(s.abs()),
                CalcValue::Vector3([x, y, z]) => CalcValue::Scalar((x * x + y * y + z * z).sqrt()),
            },
            Self::Mag2(inner) => match inner.eval(tuple_idx, coord) {
                CalcValue::Scalar(s) => CalcValue::Scalar(s * s),
                CalcValue::Vector3([x, y, z]) => CalcValue::Scalar(x * x + y * y + z * z),
            },
            Self::Normalize(inner) => {
                let [x, y, z] = inner.eval(tuple_idx, coord).as_vector3();
                let norm = (x * x + y * y + z * z).sqrt();
                if norm > 0.0 {
                    let inv = 1.0 / norm;
                    CalcValue::Vector3([x * inv, y * inv, z * inv])
                } else {
                    CalcValue::Vector3([0.0, 0.0, 0.0])
                }
            }
            Self::Dot(a, b) => {
                let [ax, ay, az] = a.eval(tuple_idx, coord).as_vector3();
                let [bx, by, bz] = b.eval(tuple_idx, coord).as_vector3();
                CalcValue::Scalar(ax * bx + ay * by + az * bz)
            }
            Self::Cross(a, b) => {
                let [ax, ay, az] = a.eval(tuple_idx, coord).as_vector3();
                let [bx, by, bz] = b.eval(tuple_idx, coord).as_vector3();
                CalcValue::Vector3([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx])
            }
        }
    }
}

fn evaluate_on_table(
    coords: &[[f64; 3]],
    attrs: &Attributes,
    params: &CalculatorParams,
    chunk_size: usize,
) -> Result<DataArray, FilterError> {
    let ast = parse_expression(&params.expression)?;
    let bound = bind_expr(&ast, attrs, params.association, &params.expression)?;

    let num_tuples = coords.len();
    let eff_chunk = chunk_size.max(1);

    match bound.kind() {
        ValueKind::Scalar => {
            let mut out = vec![0.0_f64; num_tuples];
            #[cfg(not(target_arch = "wasm32"))]
            if num_tuples > eff_chunk {
                std::thread::scope(|s| {
                    let bound_ref = &bound;
                    let mut handles = Vec::new();
                    for (chunk_idx, out_slice) in out.chunks_mut(eff_chunk).enumerate() {
                        let start = chunk_idx * eff_chunk;
                        let end = start + out_slice.len();
                        let coord_slice = &coords[start..end];
                        handles.push(s.spawn(move || {
                            for (local_i, (slot, &c)) in
                                out_slice.iter_mut().zip(coord_slice.iter()).enumerate()
                            {
                                *slot = bound_ref.eval(start + local_i, c).as_scalar();
                            }
                        }));
                    }
                    for h in handles {
                        let _ = h.join();
                    }
                });
                return Ok(DataArray::from_f64(ComponentCount::Scalar, out)?);
            }

            for (i, (slot, &c)) in out.iter_mut().zip(coords.iter()).enumerate() {
                *slot = bound.eval(i, c).as_scalar();
            }
            Ok(DataArray::from_f64(ComponentCount::Scalar, out)?)
        }
        ValueKind::Vector3 => {
            let mut out = vec![0.0_f64; num_tuples * 3];
            let elems_per_chunk = eff_chunk * 3;
            #[cfg(not(target_arch = "wasm32"))]
            if num_tuples > eff_chunk {
                std::thread::scope(|s| {
                    let bound_ref = &bound;
                    let mut handles = Vec::new();
                    for (chunk_idx, out_slice) in out.chunks_mut(elems_per_chunk).enumerate() {
                        let start = chunk_idx * eff_chunk;
                        let count = out_slice.len() / 3;
                        let coord_slice = &coords[start..start + count];
                        handles.push(s.spawn(move || {
                            for (local_i, &c) in coord_slice.iter().enumerate() {
                                let [vx, vy, vz] = bound_ref.eval(start + local_i, c).as_vector3();
                                let base = local_i * 3;
                                out_slice[base] = vx;
                                out_slice[base + 1] = vy;
                                out_slice[base + 2] = vz;
                            }
                        }));
                    }
                    for h in handles {
                        let _ = h.join();
                    }
                });
                return Ok(DataArray::from_f64(ComponentCount::Vector3, out)?);
            }

            for (i, &c) in coords.iter().enumerate() {
                let [vx, vy, vz] = bound.eval(i, c).as_vector3();
                let base = i * 3;
                out[base] = vx;
                out[base + 1] = vy;
                out[base + 2] = vz;
            }
            Ok(DataArray::from_f64(ComponentCount::Vector3, out)?)
        }
    }
}

fn invalid_expr(raw_expr: &str, reason: impl Into<String>) -> FilterError {
    FilterError::InvalidCalculatorExpression {
        expression: raw_expr.to_string(),
        reason: reason.into(),
    }
}

fn bind_expr<'a>(
    ast: &AstExpr,
    attrs: &'a Attributes,
    association: CalculatorAssociation,
    raw_expr: &str,
) -> Result<BoundExpr<'a>, FilterError> {
    let invalid = |reason: String| invalid_expr(raw_expr, reason);

    match ast {
        AstExpr::Literal(v) => Ok(BoundExpr::ScalarConst(*v)),
        AstExpr::VectorLiteral([e0, e1, e2]) => {
            let b0 = bind_expr(e0, attrs, association, raw_expr)?;
            let b1 = bind_expr(e1, attrs, association, raw_expr)?;
            let b2 = bind_expr(e2, attrs, association, raw_expr)?;
            for (idx, b) in [&b0, &b1, &b2].iter().enumerate() {
                if b.kind() != ValueKind::Scalar {
                    return Err(invalid_expr(
                        raw_expr,
                        format!("vector literal element {idx} must be a scalar expression"),
                    ));
                }
            }
            Ok(BoundExpr::VectorConstruct([
                Box::new(b0),
                Box::new(b1),
                Box::new(b2),
            ]))
        }
        AstExpr::Var { name, comp } => {
            // Check built-in mathematical constants and coordinate identifiers first
            // if the attribute table does not shadow them.
            if attrs.get(name).is_none() {
                match (name.as_str(), comp) {
                    ("pi" | "PI", None) => return Ok(BoundExpr::ScalarConst(std::f64::consts::PI)),
                    ("e" | "E", None) => return Ok(BoundExpr::ScalarConst(std::f64::consts::E)),
                    ("iHat", None) => return Ok(BoundExpr::VectorConst([1.0, 0.0, 0.0])),
                    ("jHat", None) => return Ok(BoundExpr::VectorConst([0.0, 1.0, 0.0])),
                    ("kHat", None) => return Ok(BoundExpr::VectorConst([0.0, 0.0, 1.0])),
                    ("x", None) => return Ok(BoundExpr::CoordComponent(0)),
                    ("y", None) => return Ok(BoundExpr::CoordComponent(1)),
                    ("z", None) => return Ok(BoundExpr::CoordComponent(2)),
                    ("coords", None) => return Ok(BoundExpr::CoordsVector),
                    ("coords", Some(CompAccess::Index(idx))) if *idx < 3 => {
                        return Ok(BoundExpr::CoordComponent(*idx));
                    }
                    ("coords", Some(CompAccess::Named(c))) => match c.as_str() {
                        "x" => return Ok(BoundExpr::CoordComponent(0)),
                        "y" => return Ok(BoundExpr::CoordComponent(1)),
                        "z" => return Ok(BoundExpr::CoordComponent(2)),
                        _ => {
                            return Err(invalid(format!(
                                "unknown component `{c}` on `coords`; expected x, y, z, 0, 1, 2, or magnitude"
                            )));
                        }
                    },
                    ("coords", Some(CompAccess::Magnitude)) => {
                        return Ok(BoundExpr::CoordMagnitude);
                    }
                    _ => {}
                }
            }

            let arr = attrs.get(name).ok_or_else(|| match association {
                CalculatorAssociation::Point => {
                    FilterError::MissingPointArray { name: name.clone() }
                }
                CalculatorAssociation::Cell => FilterError::MissingCellArray { name: name.clone() },
            })?;

            let num_comp = arr.components().as_usize();
            match comp {
                None => match arr.components() {
                    ComponentCount::Scalar => Ok(BoundExpr::ArrayScalar {
                        array: arr,
                        component: 0,
                    }),
                    ComponentCount::Vector3 => Ok(BoundExpr::ArrayVector3 { array: arr }),
                    ComponentCount::SymmetricTensor6 | ComponentCount::Tensor9 => {
                        Err(invalid(format!(
                            "tensor array `{name}` ({num_comp} components) requires a component selector such as `{name}.xx`, `{name}.0`, or `{name}.magnitude`"
                        )))
                    }
                },
                Some(CompAccess::Magnitude) => Ok(BoundExpr::ArrayMagnitude { array: arr }),
                Some(CompAccess::Index(idx)) => {
                    if *idx >= num_comp {
                        return Err(FilterError::InvalidComponentIndex {
                            array: name.clone(),
                            component: *idx,
                            num_components: num_comp,
                        });
                    }
                    Ok(BoundExpr::ArrayScalar {
                        array: arr,
                        component: *idx,
                    })
                }
                Some(CompAccess::Named(cname)) => {
                    let resolved = resolve_named_component(arr.components(), cname).ok_or_else(
                        || {
                            invalid(format!(
                                "unknown or invalid component `{cname}` for {num_comp}-component array `{name}`"
                            ))
                        },
                    )?;
                    Ok(BoundExpr::ArrayScalar {
                        array: arr,
                        component: resolved,
                    })
                }
            }
        }
        AstExpr::Unary { op, expr } => {
            let inner = bind_expr(expr, attrs, association, raw_expr)?;
            let kind = inner.kind();
            Ok(BoundExpr::Unary {
                op: *op,
                expr: Box::new(inner),
                kind,
            })
        }
        AstExpr::Binary { op, lhs, rhs } => {
            let l = bind_expr(lhs, attrs, association, raw_expr)?;
            let r = bind_expr(rhs, attrs, association, raw_expr)?;
            let kind = match (l.kind(), r.kind()) {
                (ValueKind::Scalar, ValueKind::Scalar) => ValueKind::Scalar,
                (ValueKind::Vector3, ValueKind::Vector3)
                | (ValueKind::Vector3, ValueKind::Scalar)
                | (ValueKind::Scalar, ValueKind::Vector3) => ValueKind::Vector3,
            };
            Ok(BoundExpr::Binary {
                op: *op,
                lhs: Box::new(l),
                rhs: Box::new(r),
                kind,
            })
        }
        AstExpr::Call { func, args } => {
            let fname = func.to_ascii_lowercase();
            // Special case: `mag(tensor_array)` or `norm(tensor_array)` where `tensor_array` is an
            // unindexed tensor identifier in `attrs`.
            if matches!(fname.as_str(), "mag" | "norm" | "length")
                && args.len() == 1
                && let AstExpr::Var { name, comp: None } = &args[0]
                && let Some(arr) = attrs.get(name)
            {
                return Ok(BoundExpr::ArrayMagnitude { array: arr });
            }

            let unary_func = match fname.as_str() {
                "sin" => Some(UnaryFunc::Sin),
                "cos" => Some(UnaryFunc::Cos),
                "tan" => Some(UnaryFunc::Tan),
                "asin" => Some(UnaryFunc::Asin),
                "acos" => Some(UnaryFunc::Acos),
                "atan" => Some(UnaryFunc::Atan),
                "sinh" => Some(UnaryFunc::Sinh),
                "cosh" => Some(UnaryFunc::Cosh),
                "tanh" => Some(UnaryFunc::Tanh),
                "sqrt" => Some(UnaryFunc::Sqrt),
                "cbrt" => Some(UnaryFunc::Cbrt),
                "abs" => Some(UnaryFunc::Abs),
                "exp" => Some(UnaryFunc::Exp),
                "ln" | "log" => Some(UnaryFunc::Ln),
                "log10" => Some(UnaryFunc::Log10),
                "log2" => Some(UnaryFunc::Log2),
                "floor" => Some(UnaryFunc::Floor),
                "ceil" => Some(UnaryFunc::Ceil),
                "round" => Some(UnaryFunc::Round),
                "sign" | "signum" => Some(UnaryFunc::Sign),
                _ => None,
            };
            if let Some(uf) = unary_func {
                if args.len() != 1 {
                    return Err(invalid(format!(
                        "function `{func}` expects 1 argument, got {}",
                        args.len()
                    )));
                }
                let arg = bind_expr(&args[0], attrs, association, raw_expr)?;
                let kind = arg.kind();
                return Ok(BoundExpr::UnaryMath {
                    func: uf,
                    arg: Box::new(arg),
                    kind,
                });
            }

            let binary_func = match fname.as_str() {
                "min" => Some(BinaryScalarFunc::Min),
                "max" => Some(BinaryScalarFunc::Max),
                "pow" => Some(BinaryScalarFunc::Pow),
                "atan2" => Some(BinaryScalarFunc::Atan2),
                "hypot" => Some(BinaryScalarFunc::Hypot),
                _ => None,
            };
            if let Some(bf) = binary_func {
                if args.len() != 2 {
                    return Err(invalid(format!(
                        "function `{func}` expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                let a = bind_expr(&args[0], attrs, association, raw_expr)?;
                let b = bind_expr(&args[1], attrs, association, raw_expr)?;
                if a.kind() != ValueKind::Scalar || b.kind() != ValueKind::Scalar {
                    return Err(invalid(format!(
                        "function `{func}` requires scalar arguments"
                    )));
                }
                return Ok(BoundExpr::BinaryMath {
                    func: bf,
                    a: Box::new(a),
                    b: Box::new(b),
                });
            }

            match fname.as_str() {
                "mag" | "norm" | "length" => {
                    if args.len() != 1 {
                        return Err(invalid(format!(
                            "function `{func}` expects 1 argument, got {}",
                            args.len()
                        )));
                    }
                    let inner = bind_expr(&args[0], attrs, association, raw_expr)?;
                    Ok(BoundExpr::Mag(Box::new(inner)))
                }
                "mag2" | "norm2" => {
                    if args.len() != 1 {
                        return Err(invalid(format!(
                            "function `{func}` expects 1 argument, got {}",
                            args.len()
                        )));
                    }
                    let inner = bind_expr(&args[0], attrs, association, raw_expr)?;
                    Ok(BoundExpr::Mag2(Box::new(inner)))
                }
                "normalize" => {
                    if args.len() != 1 {
                        return Err(invalid(format!(
                            "function `normalize` expects 1 argument, got {}",
                            args.len()
                        )));
                    }
                    let inner = bind_expr(&args[0], attrs, association, raw_expr)?;
                    if inner.kind() != ValueKind::Vector3 {
                        return Err(invalid_expr(
                            raw_expr,
                            "`normalize` requires a 3-vector argument",
                        ));
                    }
                    Ok(BoundExpr::Normalize(Box::new(inner)))
                }
                "dot" => {
                    if args.len() != 2 {
                        return Err(invalid(format!(
                            "function `dot` expects 2 arguments, got {}",
                            args.len()
                        )));
                    }
                    let a = bind_expr(&args[0], attrs, association, raw_expr)?;
                    let b = bind_expr(&args[1], attrs, association, raw_expr)?;
                    if a.kind() != ValueKind::Vector3 || b.kind() != ValueKind::Vector3 {
                        return Err(invalid_expr(
                            raw_expr,
                            "`dot(u, v)` requires two 3-vector arguments",
                        ));
                    }
                    Ok(BoundExpr::Dot(Box::new(a), Box::new(b)))
                }
                "cross" => {
                    if args.len() != 2 {
                        return Err(invalid(format!(
                            "function `cross` expects 2 arguments, got {}",
                            args.len()
                        )));
                    }
                    let a = bind_expr(&args[0], attrs, association, raw_expr)?;
                    let b = bind_expr(&args[1], attrs, association, raw_expr)?;
                    if a.kind() != ValueKind::Vector3 || b.kind() != ValueKind::Vector3 {
                        return Err(invalid_expr(
                            raw_expr,
                            "`cross(u, v)` requires two 3-vector arguments",
                        ));
                    }
                    Ok(BoundExpr::Cross(Box::new(a), Box::new(b)))
                }
                "vec" | "vec3" => {
                    if args.len() != 3 {
                        return Err(invalid(format!(
                            "function `{func}` expects 3 scalar arguments, got {}",
                            args.len()
                        )));
                    }
                    let b0 = bind_expr(&args[0], attrs, association, raw_expr)?;
                    let b1 = bind_expr(&args[1], attrs, association, raw_expr)?;
                    let b2 = bind_expr(&args[2], attrs, association, raw_expr)?;
                    if b0.kind() != ValueKind::Scalar
                        || b1.kind() != ValueKind::Scalar
                        || b2.kind() != ValueKind::Scalar
                    {
                        return Err(invalid(format!(
                            "function `{func}` requires 3 scalar arguments"
                        )));
                    }
                    Ok(BoundExpr::VectorConstruct([
                        Box::new(b0),
                        Box::new(b1),
                        Box::new(b2),
                    ]))
                }
                "clamp" => {
                    if args.len() != 3 {
                        return Err(invalid(format!(
                            "function `clamp` expects 3 scalar arguments, got {}",
                            args.len()
                        )));
                    }
                    let x = bind_expr(&args[0], attrs, association, raw_expr)?;
                    let lo = bind_expr(&args[1], attrs, association, raw_expr)?;
                    let hi = bind_expr(&args[2], attrs, association, raw_expr)?;
                    if x.kind() != ValueKind::Scalar
                        || lo.kind() != ValueKind::Scalar
                        || hi.kind() != ValueKind::Scalar
                    {
                        return Err(invalid_expr(
                            raw_expr,
                            "`clamp(x, lo, hi)` requires 3 scalar arguments",
                        ));
                    }
                    Ok(BoundExpr::Clamp {
                        x: Box::new(x),
                        lo: Box::new(lo),
                        hi: Box::new(hi),
                    })
                }
                "lerp" => {
                    if args.len() != 3 {
                        return Err(invalid(format!(
                            "function `lerp` expects 3 arguments, got {}",
                            args.len()
                        )));
                    }
                    let a = bind_expr(&args[0], attrs, association, raw_expr)?;
                    let b = bind_expr(&args[1], attrs, association, raw_expr)?;
                    let t = bind_expr(&args[2], attrs, association, raw_expr)?;
                    if t.kind() != ValueKind::Scalar || a.kind() != b.kind() {
                        return Err(invalid_expr(
                            raw_expr,
                            "`lerp(a, b, t)` requires `a` and `b` of matching type and scalar `t`",
                        ));
                    }
                    let kind = a.kind();
                    Ok(BoundExpr::Lerp {
                        a: Box::new(a),
                        b: Box::new(b),
                        t: Box::new(t),
                        kind,
                    })
                }
                _ => Err(invalid(format!("unknown function `{func}`"))),
            }
        }
    }
}

fn resolve_named_component(comp_count: ComponentCount, name: &str) -> Option<usize> {
    let lower = name.to_ascii_lowercase();
    match (comp_count, lower.as_str()) {
        (ComponentCount::Scalar, "x" | "0") => Some(0),
        (ComponentCount::Vector3, "x") => Some(0),
        (ComponentCount::Vector3, "y") => Some(1),
        (ComponentCount::Vector3, "z") => Some(2),
        (ComponentCount::SymmetricTensor6, "xx") => Some(0),
        (ComponentCount::SymmetricTensor6, "yy") => Some(1),
        (ComponentCount::SymmetricTensor6, "zz") => Some(2),
        (ComponentCount::SymmetricTensor6, "xy" | "yx") => Some(3),
        (ComponentCount::SymmetricTensor6, "yz" | "zy") => Some(4),
        (ComponentCount::SymmetricTensor6, "xz" | "zx") => Some(5),
        (ComponentCount::Tensor9, "xx") => Some(0),
        (ComponentCount::Tensor9, "xy") => Some(1),
        (ComponentCount::Tensor9, "xz") => Some(2),
        (ComponentCount::Tensor9, "yx") => Some(3),
        (ComponentCount::Tensor9, "yy") => Some(4),
        (ComponentCount::Tensor9, "yz") => Some(5),
        (ComponentCount::Tensor9, "zx") => Some(6),
        (ComponentCount::Tensor9, "zy") => Some(7),
        (ComponentCount::Tensor9, "zz") => Some(8),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Dot,
    Colon,
}

fn tokenize(input: &str) -> Result<Vec<Token>, FilterError> {
    let invalid = |reason: String| invalid_expr(input, reason);

    let bytes = input.as_bytes();
    let mut i = 0;
    let mut tokens = Vec::new();

    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        match b {
            b'+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            b'-' => {
                tokens.push(Token::Minus);
                i += 1;
            }
            b'*' => {
                tokens.push(Token::Star);
                i += 1;
            }
            b'/' => {
                tokens.push(Token::Slash);
                i += 1;
            }
            b'^' => {
                tokens.push(Token::Caret);
                i += 1;
            }
            b'(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            b')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            b'[' => {
                tokens.push(Token::LBracket);
                i += 1;
            }
            b']' => {
                tokens.push(Token::RBracket);
                i += 1;
            }
            b',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            b':' => {
                tokens.push(Token::Colon);
                i += 1;
            }
            b'.' => {
                if i + 1 < bytes.len()
                    && bytes[i + 1].is_ascii_digit()
                    && !matches!(tokens.last(), Some(Token::Ident(_)))
                {
                    let start = i;
                    i += 1;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                        i += 1;
                        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                            i += 1;
                        }
                        while i < bytes.len() && bytes[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                    let s = &input[start..i];
                    let val: f64 = s
                        .parse()
                        .map_err(|_| invalid(format!("invalid numeric literal `{s}`")))?;
                    tokens.push(Token::Number(val));
                } else {
                    tokens.push(Token::Dot);
                    i += 1;
                }
            }
            b'0'..=b'9' => {
                if matches!(tokens.last(), Some(Token::Dot | Token::Colon)) {
                    let start = i;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                    let s = &input[start..i];
                    let val: f64 = s
                        .parse()
                        .map_err(|_| invalid(format!("invalid component index `{s}`")))?;
                    tokens.push(Token::Number(val));
                    continue;
                }
                let start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if i < bytes.len() && bytes[i] == b'.' {
                    i += 1;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                    i += 1;
                    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                        i += 1;
                    }
                    let exp_start = i;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                    if i == exp_start {
                        return Err(invalid(format!(
                            "malformed exponent in numeric literal `{}`",
                            &input[start..i]
                        )));
                    }
                }
                let s = &input[start..i];
                let val: f64 = s
                    .parse()
                    .map_err(|_| invalid(format!("invalid numeric literal `{s}`")))?;
                tokens.push(Token::Number(val));
            }
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                tokens.push(Token::Ident(input[start..i].to_string()));
            }
            other => {
                return Err(invalid(format!(
                    "unexpected character `{}` at byte {i}",
                    other as char
                )));
            }
        }
    }

    if tokens.is_empty() {
        return Err(invalid_expr(input, "expression is empty"));
    }

    Ok(tokens)
}

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    raw: &'a str,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token], raw: &'a str) -> Self {
        Self {
            tokens,
            pos: 0,
            raw,
        }
    }

    fn err(&self, reason: impl Into<String>) -> FilterError {
        FilterError::InvalidCalculatorExpression {
            expression: self.raw.to_string(),
            reason: reason.into(),
        }
    }

    fn peek(&self) -> Option<&'a Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<&'a Token> {
        let tok = self.tokens.get(self.pos)?;
        self.pos += 1;
        Some(tok)
    }

    fn parse_expr(&mut self) -> Result<AstExpr, FilterError> {
        self.parse_additive()
    }

    fn parse_additive(&mut self) -> Result<AstExpr, FilterError> {
        let mut lhs = self.parse_multiplicative()?;
        while let Some(tok) = self.peek() {
            let op = match tok {
                Token::Plus => BinaryOp::Add,
                Token::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_multiplicative()?;
            lhs = AstExpr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_multiplicative(&mut self) -> Result<AstExpr, FilterError> {
        let mut lhs = self.parse_unary()?;
        while let Some(tok) = self.peek() {
            let op = match tok {
                Token::Star => BinaryOp::Mul,
                Token::Slash => BinaryOp::Div,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_unary()?;
            lhs = AstExpr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<AstExpr, FilterError> {
        match self.peek() {
            Some(Token::Plus) => {
                self.pos += 1;
                let inner = self.parse_unary()?;
                Ok(AstExpr::Unary {
                    op: UnaryOp::Plus,
                    expr: Box::new(inner),
                })
            }
            Some(Token::Minus) => {
                self.pos += 1;
                let inner = self.parse_unary()?;
                Ok(AstExpr::Unary {
                    op: UnaryOp::Neg,
                    expr: Box::new(inner),
                })
            }
            _ => self.parse_power(),
        }
    }

    fn parse_power(&mut self) -> Result<AstExpr, FilterError> {
        let base = self.parse_primary()?;
        if matches!(self.peek(), Some(Token::Caret)) {
            self.pos += 1;
            let exp = self.parse_unary()?;
            Ok(AstExpr::Binary {
                op: BinaryOp::Pow,
                lhs: Box::new(base),
                rhs: Box::new(exp),
            })
        } else {
            Ok(base)
        }
    }

    fn parse_primary(&mut self) -> Result<AstExpr, FilterError> {
        match self.next() {
            Some(Token::Number(val)) => Ok(AstExpr::Literal(*val)),
            Some(Token::LParen) => {
                let inner = self.parse_expr()?;
                match self.next() {
                    Some(Token::RParen) => Ok(inner),
                    _ => Err(self.err("expected closing `)`")),
                }
            }
            Some(Token::LBracket) => {
                let e0 = self.parse_expr()?;
                if !matches!(self.next(), Some(Token::Comma)) {
                    return Err(
                        self.err("expected `,` after first element of `[x, y, z]` vector literal")
                    );
                }
                let e1 = self.parse_expr()?;
                if !matches!(self.next(), Some(Token::Comma)) {
                    return Err(
                        self.err("expected `,` after second element of `[x, y, z]` vector literal")
                    );
                }
                let e2 = self.parse_expr()?;
                if !matches!(self.next(), Some(Token::RBracket)) {
                    return Err(self.err("expected closing `]` for `[x, y, z]` vector literal"));
                }
                Ok(AstExpr::VectorLiteral([
                    Box::new(e0),
                    Box::new(e1),
                    Box::new(e2),
                ]))
            }
            Some(Token::Ident(name)) => {
                if matches!(self.peek(), Some(Token::LParen)) {
                    self.pos += 1;
                    let mut args = Vec::new();
                    if matches!(self.peek(), Some(Token::RParen)) {
                        self.pos += 1;
                    } else {
                        loop {
                            args.push(self.parse_expr()?);
                            match self.next() {
                                Some(Token::Comma) => continue,
                                Some(Token::RParen) => break,
                                _ => {
                                    return Err(self
                                        .err(format!("expected `,` or `)` in call to `{name}`")));
                                }
                            }
                        }
                    }
                    Ok(AstExpr::Call {
                        func: name.clone(),
                        args,
                    })
                } else if matches!(self.peek(), Some(Token::Dot | Token::Colon)) {
                    self.pos += 1;
                    let comp = match self.next() {
                        Some(Token::Ident(cname)) => {
                            let lower = cname.to_ascii_lowercase();
                            if lower == "mag" || lower == "magnitude" {
                                CompAccess::Magnitude
                            } else {
                                CompAccess::Named(cname.clone())
                            }
                        }
                        Some(Token::Number(n)) => {
                            if *n < 0.0 || n.fract() != 0.0 {
                                return Err(self.err(format!(
                                    "component index after `.` must be a non-negative integer, got {n}"
                                )));
                            }
                            CompAccess::Index(*n as usize)
                        }
                        _ => {
                            return Err(self
                                .err(format!("expected component name or index after `{name}.`")));
                        }
                    };
                    Ok(AstExpr::Var {
                        name: name.clone(),
                        comp: Some(comp),
                    })
                } else {
                    Ok(AstExpr::Var {
                        name: name.clone(),
                        comp: None,
                    })
                }
            }
            Some(tok) => Err(self.err(format!("unexpected token {tok:?}"))),
            None => Err(self.err("unexpected end of expression")),
        }
    }
}

fn parse_expression(raw: &str) -> Result<AstExpr, FilterError> {
    let tokens = tokenize(raw)?;
    let mut parser = Parser::new(&tokens, raw);
    let ast = parser.parse_expr()?;
    if let Some(extra) = parser.peek() {
        return Err(parser.err(format!("unexpected trailing token {extra:?}")));
    }
    Ok(ast)
}
