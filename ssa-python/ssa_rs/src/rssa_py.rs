//! Python bindings of the Rssa-compatible part of `ssa_rs`: nD / shaped /
//! circular / projection / oblique SSA (`NdSsa`), Complex SSA (`Cssa`),
//! MSSA of series with different lengths (`MssaList`), gap filling, plus
//! the extra methods of the plain 1D / MSSA `Model` (see `lib.rs`).
//!
//! Arrays are NumPy arrays in C order; fields of rank > 1 are converted to
//! the column-major layout of the Rust core (the first axis varies fastest
//! there) and back, so `reconstruct` of an `(n0, n1)` field returns an
//! `(n0, n1)` array.

use faer::Mat;
use numpy::ndarray::{Array1, Array2, ArrayD, ArrayViewD, IxDyn, ShapeBuilder};
use numpy::{Complex64 as NpC64, IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use ssa_rs::rssa::cssa::Cssa as RCssa;
use ssa_rs::rssa::cssa_ext::CssaOssaResult;
use ssa_rs::rssa::gaps::{self, ShapedSsa as RShaped};
use ssa_rs::rssa::misc::{self, Interval};
use ssa_rs::rssa::mssa_list::MssaList as RMssaList;
use ssa_rs::rssa::nd::{self, NdMethod, NdSsa as RNd, NdSsaBuilder, Pairing};
use ssa_rs::rssa::nd_ossa::NdOssaResult;
use ssa_rs::rssa::pssa::Projector;
use ssa_rs::rssa::{Base, Direction, Params, SolveMethod, Subspace};
use ssa_rs::{Complex64, Matrix, SsaError};

fn to_py(e: SsaError) -> PyErr {
    match e {
        SsaError::InvalidArgument(m) => PyValueError::new_err(m),
        SsaError::IndexOutOfRange { .. } => PyIndexError::new_err(e.to_string()),
        SsaError::LinAlg(m) => crate::LinAlgError::new_err(m),
        SsaError::Io(m) => pyo3::exceptions::PyIOError::new_err(m),
    }
}

fn mat_to_py<'py>(py: Python<'py>, m: &Matrix) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)]).into_pyarray(py)
}

fn mat_from2(a: &ArrayViewD<'_, f64>) -> PyResult<Matrix> {
    if a.ndim() != 2 {
        return Err(PyValueError::new_err("expected a 2-D array"));
    }
    Ok(Mat::from_fn(a.shape()[0], a.shape()[1], |i, j| a[[i, j]]))
}

fn vec_py<'py>(py: Python<'py>, v: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    Array1::from_vec(v).into_pyarray(py)
}

fn cvec_py<'py>(py: Python<'py>, v: &[Complex64]) -> Bound<'py, PyArray1<NpC64>> {
    Array1::from_iter(v.iter().map(|z| NpC64::new(z.re, z.im))).into_pyarray(py)
}

fn cmat_py<'py>(py: Python<'py>, m: &Mat<Complex64>) -> Bound<'py, PyArray2<NpC64>> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| NpC64::new(m[(i, j)].re, m[(i, j)].im)).into_pyarray(py)
}

fn cvec_from(a: &PyReadonlyArray1<'_, NpC64>) -> Vec<Complex64> {
    a.as_array().iter().map(|z| Complex64::new(z.re, z.im)).collect()
}

/// C-order NumPy array -> column-major vector.
fn flat_f(a: &ArrayViewD<'_, f64>) -> Vec<f64> {
    a.t().iter().copied().collect()
}

fn flat_f_bool(a: &ArrayViewD<'_, bool>) -> Vec<bool> {
    a.t().iter().copied().collect()
}

/// Column-major vector -> NumPy array of the given shape.
fn unflat<'py>(py: Python<'py>, v: Vec<f64>, dims: &[usize]) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let a = ArrayD::from_shape_vec(IxDyn(dims).f(), v).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(a.as_standard_layout().into_owned().into_pyarray(py))
}

fn params_dict<'py>(py: Python<'py>, p: &Params) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("roots", cvec_py(py, &p.roots))?;
    d.set_item("periods", vec_py(py, p.periods.clone()))?;
    d.set_item("frequencies", vec_py(py, p.frequencies.clone()))?;
    d.set_item("moduli", vec_py(py, p.moduli.clone()))?;
    d.set_item("rates", vec_py(py, p.rates.clone()))?;
    Ok(d)
}

fn p_subspace(s: &str) -> PyResult<Subspace> {
    match s {
        "column" => Ok(Subspace::Column),
        "row" => Ok(Subspace::Row),
        o => Err(PyValueError::new_err(format!("subspace must be 'column' or 'row', got {o:?}"))),
    }
}

fn p_solve(s: &str) -> PyResult<SolveMethod> {
    match s {
        "ls" => Ok(SolveMethod::Ls),
        "tls" => Ok(SolveMethod::Tls),
        o => Err(PyValueError::new_err(format!("method must be 'ls' or 'tls', got {o:?}"))),
    }
}

fn p_dir(s: &str) -> PyResult<Direction> {
    match s {
        "column" => Ok(Direction::Column),
        "row" => Ok(Direction::Row),
        o => Err(PyValueError::new_err(format!("direction must be 'column' or 'row', got {o:?}"))),
    }
}

fn p_base(s: &str) -> PyResult<Base> {
    match s {
        "reconstructed" => Ok(Base::Reconstructed),
        "original" => Ok(Base::Original),
        o => Err(PyValueError::new_err(format!("base must be 'reconstructed' or 'original', got {o:?}"))),
    }
}

/// `svd_method`: "auto" (Rssa default), "full" or "lanczos"; `neig` is the
/// number of triples for "auto" / "lanczos".
fn p_method(svd_method: &str, neig: usize) -> PyResult<NdMethod> {
    match svd_method {
        "auto" => Ok(NdMethod::Auto(neig)),
        "full" => Ok(NdMethod::Full),
        "lanczos" => Ok(NdMethod::Lanczos(neig)),
        o => Err(PyValueError::new_err(format!("svd_method must be 'auto', 'full' or 'lanczos', got {o:?}"))),
    }
}

/// `None`, an integer degree (1 = centering, 2 = linear, ...), or a basis.
fn projector(p: Option<&Bound<'_, PyAny>>) -> PyResult<Projector> {
    let Some(p) = p else { return Ok(Projector::None) };
    if p.is_none() {
        return Ok(Projector::None);
    }
    if let Ok(d) = p.extract::<usize>() {
        return Ok(if d == 0 { Projector::None } else { Projector::Degree(d) });
    }
    let a: PyReadonlyArrayDyn<'_, f64> = p.extract().map_err(|_| PyValueError::new_err("projector must be None, an int degree or a 2-D basis"))?;
    Ok(Projector::Basis(mat_from2(&a.as_array())?))
}

fn groups_arg(g: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    g
}

// --------------------------------------------------------------------- nD

/// Fitted nD / shaped / circular / projection / oblique SSA.
#[pyclass(module = "ssa_rs._core", frozen)]
struct NdSsa {
    inner: RNd,
}

fn wrap_ossa<'py>(py: Python<'py>, r: NdOssaResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("model", Py::new(py, NdSsa { inner: r.model })?)?;
    d.set_item("groups", r.groups)?;
    d.set_item("set", r.set)?;
    d.set_item("iterations", r.iterations)?;
    d.set_item("converged", r.converged)?;
    Ok(d)
}

#[pymethods]
impl NdSsa {
    /// Fits SSA on an array of any rank.
    ///
    /// `window`: window sizes (default: half of every dimension); `wmask`:
    /// boolean array of the window shape (shaped window, e.g. `ball_mask`);
    /// `mask`: boolean array of the field shape (extra missing values);
    /// `circular`: per dimension; `column_projector` / `row_projector`:
    /// None, int degree or basis (Projection SSA); `column_oblique` /
    /// `row_oblique`: positive weights (W-OSSA); `toeplitz`: Toeplitz SSA
    /// (1-D); `svd_method`: "auto" | "full" | "lanczos" with `neig` triples.
    #[staticmethod]
    #[pyo3(signature = (field, window=None, wmask=None, mask=None, circular=None, column_projector=None, row_projector=None, column_oblique=None, row_oblique=None, toeplitz=false, svd_method="auto", neig=50))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        py: Python<'_>,
        field: PyReadonlyArrayDyn<'_, f64>,
        window: Option<Vec<usize>>,
        wmask: Option<PyReadonlyArrayDyn<'_, bool>>,
        mask: Option<PyReadonlyArrayDyn<'_, bool>>,
        circular: Option<Vec<bool>>,
        column_projector: Option<&Bound<'_, PyAny>>,
        row_projector: Option<&Bound<'_, PyAny>>,
        column_oblique: Option<Vec<f64>>,
        row_oblique: Option<Vec<f64>>,
        toeplitz: bool,
        svd_method: &str,
        neig: usize,
    ) -> PyResult<NdSsa> {
        let f = field.as_array();
        let n: Vec<usize> = f.shape().to_vec();
        let mut b = NdSsaBuilder::new(flat_f(&f), n.clone());
        if let Some(w) = wmask {
            let wv = w.as_array();
            b = b.wmask(wv.shape().to_vec(), flat_f_bool(&wv));
        } else if let Some(l) = window {
            b = b.window(l);
        }
        if let Some(m) = mask {
            if m.shape() != f.shape() {
                return Err(PyValueError::new_err("mask must have the shape of the field"));
            }
            b = b.mask(flat_f_bool(&m.as_array()));
        }
        if let Some(c) = circular {
            if c.len() != n.len() {
                return Err(PyValueError::new_err("circular needs one flag per dimension"));
            }
            b = b.circular(c);
        }
        b = b.projectors(projector(column_projector)?, projector(row_projector)?);
        b = b.oblique(column_oblique, row_oblique);
        if toeplitz {
            b = b.toeplitz(Some(neig));
        }
        b = b.method(p_method(svd_method, neig)?);
        let inner = py.detach(|| b.fit()).map_err(to_py)?;
        Ok(NdSsa { inner })
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.n.clone()
    }
    #[getter]
    fn window(&self) -> Vec<usize> {
        self.inner.l.clone()
    }
    #[getter]
    fn d(&self) -> usize {
        self.inner.d
    }
    #[getter]
    fn nspecial(&self) -> usize {
        self.inner.nspecial()
    }
    #[getter]
    fn sigma<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_py(py, self.inner.sigma.clone())
    }
    #[getter]
    fn u<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat_to_py(py, &self.inner.u)
    }
    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat_to_py(py, &self.inner.v)
    }
    /// Hankel weights (multiplicity of every field element), field-shaped.
    fn hankel_weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        unflat(py, self.inner.hweights(), &self.inner.n)
    }

    fn reconstruct<'py>(&self, py: Python<'py>, group: Vec<usize>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let m = &self.inner;
        let r = py.detach(|| m.reconstruct(&group)).map_err(to_py)?;
        unflat(py, r, &m.n)
    }
    fn residuals<'py>(&self, py: Python<'py>, group: Vec<usize>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let r = self.inner.residuals(&group).map_err(to_py)?;
        unflat(py, r, &self.inner.n)
    }
    fn wcor<'py>(&self, py: Python<'py>, groups: Vec<Vec<usize>>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = &self.inner;
        let w = py.detach(|| m.wcor(&groups_arg(groups))).map_err(to_py)?;
        Ok(mat_to_py(py, &w))
    }
    fn wnorm(&self) -> f64 {
        self.inner.wnorm()
    }
    fn contributions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_py(py, self.inner.contributions())
    }
    fn grouping_auto_wcor(&self, components: Vec<usize>, nclust: usize) -> PyResult<Vec<Vec<usize>>> {
        self.inner.grouping_auto_wcor(&components, nclust).map_err(to_py)
    }

    /// ESPRIT per dimension; returns one dict per dimension with `roots`
    /// (complex), `periods`, `frequencies`, `moduli`, `rates`.
    #[pyo3(signature = (group, subspace_kind="column", dimensions=None, method="ls", pairing="diag", normalize=None))]
    fn parestimate<'py>(
        &self,
        py: Python<'py>,
        group: Vec<usize>,
        subspace_kind: &str,
        dimensions: Option<Vec<usize>>,
        method: &str,
        pairing: &str,
        normalize: Option<Vec<bool>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let pr = match pairing {
            "diag" => Pairing::Diag,
            "memp" => Pairing::Memp,
            o => return Err(PyValueError::new_err(format!("pairing must be 'diag' or 'memp', got {o:?}"))),
        };
        let ps = self
            .inner
            .parestimate(&group, p_subspace(subspace_kind)?, dimensions.as_deref(), p_solve(method)?, pr, normalize.as_deref())
            .map_err(to_py)?;
        let out = PyList::empty(py);
        for p in &ps {
            out.append(params_dict(py, p)?)?;
        }
        Ok(out)
    }

    #[pyo3(signature = (rank, correct=true, tol=1e-6, maxiter=0))]
    fn cadzow<'py>(&self, py: Python<'py>, rank: usize, correct: bool, tol: f64, maxiter: usize) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let m = &self.inner;
        let r = py.detach(|| m.cadzow(rank, correct, tol, maxiter)).map_err(to_py)?;
        unflat(py, r, &m.n)
    }

    /// Iterative gap filling of a field with NaN; one filled field per group.
    #[pyo3(signature = (groups, fill=None, tol=1e-6, maxiter=0))]
    fn igapfill<'py>(&self, py: Python<'py>, groups: Vec<Vec<usize>>, fill: Option<f64>, tol: f64, maxiter: usize) -> PyResult<Vec<Bound<'py, PyArrayDyn<f64>>>> {
        let m = &self.inner;
        let r = py.detach(|| m.igapfill(&groups, fill, tol, maxiter)).map_err(to_py)?;
        r.into_iter().map(|v| unflat(py, v, &m.n)).collect()
    }

    // 1-D only
    #[pyo3(signature = (group, reverse=false))]
    fn lrr<'py>(&self, py: Python<'py>, group: Vec<usize>, reverse: bool) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec_py(py, self.inner.lrr(&group, reverse).map_err(to_py)?))
    }
    #[pyo3(signature = (group, steps, reconstructed_base=true, reverse=false))]
    fn rforecast<'py>(&self, py: Python<'py>, group: Vec<usize>, steps: usize, reconstructed_base: bool, reverse: bool) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let m = &self.inner;
        Ok(vec_py(py, py.detach(|| m.rforecast(&group, steps, reconstructed_base, reverse)).map_err(to_py)?))
    }
    fn vforecast<'py>(&self, py: Python<'py>, group: Vec<usize>, steps: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let m = &self.inner;
        Ok(vec_py(py, py.detach(|| m.vforecast(&group, steps)).map_err(to_py)?))
    }

    // oblique decompositions: each returns a dict with `model` (NdSsa),
    // `groups`, `set`, `iterations`, `converged`
    #[pyo3(signature = (nested_groups, filter=None, gamma=f64::INFINITY, normalize=true))]
    fn fossa<'py>(&self, py: Python<'py>, nested_groups: Vec<Vec<usize>>, filter: Option<Vec<f64>>, gamma: f64, normalize: bool) -> PyResult<Bound<'py, PyDict>> {
        let f = filter.unwrap_or_else(|| vec![-1.0, 1.0]);
        wrap_ossa(py, self.inner.fossa(&nested_groups, &f, gamma, normalize).map_err(to_py)?)
    }
    #[pyo3(signature = (nested_groups, tol=1e-5, kappa=Some(2.0), maxiter=100, kappa_balance=0.5))]
    fn iossa<'py>(&self, py: Python<'py>, nested_groups: Vec<Vec<usize>>, tol: f64, kappa: Option<f64>, maxiter: usize, kappa_balance: f64) -> PyResult<Bound<'py, PyDict>> {
        let m = &self.inner;
        let r = py.detach(|| m.iossa(&nested_groups, tol, kappa, maxiter, kappa_balance)).map_err(to_py)?;
        wrap_ossa(py, r)
    }
    #[pyo3(signature = (nested_groups, k, method="ls", dimensions=None))]
    fn eossa<'py>(&self, py: Python<'py>, nested_groups: Vec<Vec<usize>>, k: usize, method: &str, dimensions: Option<Vec<usize>>) -> PyResult<Bound<'py, PyDict>> {
        let r = self.inner.eossa(&nested_groups, k, p_solve(method)?, dimensions.as_deref()).map_err(to_py)?;
        wrap_ossa(py, r)
    }
    fn owcor<'py>(&self, py: Python<'py>, groups: Vec<Vec<usize>>, set: Vec<usize>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(mat_to_py(py, &self.inner.owcor(&groups, &set).map_err(to_py)?))
    }
}

/// Window mask of a ball (`circle` in Rssa for rank 2): returns a boolean
/// array of shape `(2r-1,) * rank`.
#[pyfunction]
fn ball_mask<'py>(py: Python<'py>, r: usize, rank: usize) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    let (dims, m) = nd::ball_mask(r, rank);
    let a = ArrayD::from_shape_vec(IxDyn(&dims).f(), m).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(a.as_standard_layout().into_owned().into_pyarray(py))
}

/// Window mask of a simplex (`triangle` in Rssa).
#[pyfunction]
fn simplex_mask<'py>(py: Python<'py>, side: usize, rank: usize) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    let (dims, m) = nd::simplex_mask(side, rank);
    let a = ArrayD::from_shape_vec(IxDyn(&dims).f(), m).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(a.as_standard_layout().into_owned().into_pyarray(py))
}

// ------------------------------------------------------------ MSSA lists

/// MSSA of series with different lengths (and leading / trailing NaN).
#[pyclass(module = "ssa_rs._core", frozen)]
struct MssaList {
    inner: RMssaList,
}

#[pymethods]
impl MssaList {
    #[staticmethod]
    #[pyo3(signature = (series, window=None, svd_method="auto", neig=50))]
    fn fit(py: Python<'_>, series: Vec<Vec<f64>>, window: Option<usize>, svd_method: &str, neig: usize) -> PyResult<MssaList> {
        let m = p_method(svd_method, neig)?;
        let inner = py.detach(|| RMssaList::fit(&series, window, m)).map_err(to_py)?;
        Ok(MssaList { inner })
    }
    #[getter]
    fn d(&self) -> usize {
        self.inner.nd.d
    }
    #[getter]
    fn sigma<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_py(py, self.inner.nd.sigma.clone())
    }
    #[getter]
    fn lengths(&self) -> Vec<usize> {
        self.inner.n.clone()
    }
    fn reconstruct(&self, group: Vec<usize>) -> PyResult<Vec<Vec<f64>>> {
        self.inner.reconstruct(&group).map_err(to_py)
    }
    fn wcor<'py>(&self, py: Python<'py>, groups: Vec<Vec<usize>>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(mat_to_py(py, &self.inner.wcor(&groups).map_err(to_py)?))
    }
    #[pyo3(signature = (group, steps, base="reconstructed", direction="column"))]
    fn rforecast(&self, group: Vec<usize>, steps: usize, base: &str, direction: &str) -> PyResult<Vec<Vec<f64>>> {
        self.inner.rforecast(&group, steps, p_base(base)?, p_dir(direction)?).map_err(to_py)
    }
    #[pyo3(signature = (group, steps, direction="column"))]
    fn vforecast(&self, group: Vec<usize>, steps: usize, direction: &str) -> PyResult<Vec<Vec<f64>>> {
        self.inner.vforecast(&group, steps, p_dir(direction)?).map_err(to_py)
    }
    #[pyo3(signature = (group, reconstructed_base=false))]
    fn gapfill(&self, group: Vec<usize>, reconstructed_base: bool) -> PyResult<Vec<Vec<f64>>> {
        self.inner.gapfill(&group, reconstructed_base).map_err(to_py)
    }
    #[pyo3(signature = (groups, fill=None, tol=1e-6, maxiter=0))]
    fn igapfill(&self, groups: Vec<Vec<usize>>, fill: Option<f64>, tol: f64, maxiter: usize) -> PyResult<Vec<Vec<Vec<f64>>>> {
        self.inner.igapfill(&groups, fill, tol, maxiter).map_err(to_py)
    }
}

// ------------------------------------------------------------ shaped 1-D

/// 1-D SSA of a series with missing values (NaN).
#[pyclass(module = "ssa_rs._core", frozen)]
struct ShapedSsa {
    inner: RShaped,
}

#[pymethods]
impl ShapedSsa {
    #[staticmethod]
    fn fit(py: Python<'_>, x: Vec<f64>, window: usize) -> PyResult<ShapedSsa> {
        Ok(ShapedSsa { inner: py.detach(|| RShaped::fit(&x, window)).map_err(to_py)? })
    }
    #[getter]
    fn d(&self) -> usize {
        self.inner.d
    }
    #[getter]
    fn sigma<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_py(py, self.inner.sigma.clone())
    }
    fn reconstruct<'py>(&self, py: Python<'py>, group: Vec<usize>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec_py(py, self.inner.reconstruct(&group).map_err(to_py)?))
    }
    fn wcor<'py>(&self, py: Python<'py>, groups: Vec<Vec<usize>>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(mat_to_py(py, &self.inner.wcor(&groups).map_err(to_py)?))
    }
    #[pyo3(signature = (group, reconstructed_base=false, simultaneous=false))]
    fn gapfill<'py>(&self, py: Python<'py>, group: Vec<usize>, reconstructed_base: bool, simultaneous: bool) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let m = &self.inner;
        let r = py
            .detach(|| if simultaneous { m.gapfill_simultaneous(&group, reconstructed_base) } else { m.gapfill(&group, reconstructed_base) })
            .map_err(to_py)?;
        Ok(vec_py(py, r))
    }
}

/// Iterative gap filling of a 1-D series (one filled series per group).
#[pyfunction]
#[pyo3(signature = (x, window, groups, fill=None, tol=1e-6, maxiter=0))]
fn igapfill(py: Python<'_>, x: Vec<f64>, window: usize, groups: Vec<Vec<usize>>, fill: Option<f64>, tol: f64, maxiter: usize) -> PyResult<Vec<Vec<f64>>> {
    py.detach(|| gaps::igapfill(&x, window, &groups, fill, tol, maxiter)).map_err(to_py)
}

/// Gaps of a series: list of `(left, right, position, dense)`.
#[pyfunction]
fn summarize_gaps(x: Vec<f64>, window: usize) -> Vec<(usize, usize, String, bool)> {
    gaps::summarize_gaps(&x, window)
        .into_iter()
        .map(|g| (g.left, g.right, format!("{:?}", g.pos).to_lowercase(), g.dense))
        .collect()
}

/// `(L, percentage of complete lag vectors)` for every window length.
#[pyfunction]
fn complete_lag_fraction(x: Vec<f64>) -> Vec<(usize, f64)> {
    gaps::complete_lag_fraction(&x)
}

// ------------------------------------------------------------ Complex SSA

/// Complex SSA (complex-valued series, also with missing values).
#[pyclass(module = "ssa_rs._core", frozen)]
struct Cssa {
    inner: RCssa,
}

fn wrap_cossa<'py>(py: Python<'py>, r: CssaOssaResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("model", Py::new(py, Cssa { inner: r.model })?)?;
    d.set_item("groups", r.groups)?;
    d.set_item("set", r.set)?;
    d.set_item("iterations", r.iterations)?;
    d.set_item("converged", r.converged)?;
    Ok(d)
}

#[pymethods]
impl Cssa {
    #[staticmethod]
    fn fit(py: Python<'_>, z: PyReadonlyArray1<'_, NpC64>, window: usize) -> PyResult<Cssa> {
        let zv = cvec_from(&z);
        Ok(Cssa { inner: py.detach(|| RCssa::fit(&zv, window)).map_err(to_py)? })
    }
    #[getter]
    fn d(&self) -> usize {
        self.inner.d
    }
    #[getter]
    fn sigma<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_py(py, self.inner.sigma.clone())
    }
    #[getter]
    fn u<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpC64>> {
        cmat_py(py, &self.inner.u)
    }
    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpC64>> {
        cmat_py(py, &self.inner.v)
    }
    fn reconstruct<'py>(&self, py: Python<'py>, group: Vec<usize>) -> PyResult<Bound<'py, PyArray1<NpC64>>> {
        let m = &self.inner;
        let r = py.detach(|| m.reconstruct(&group)).map_err(to_py)?;
        Ok(cvec_py(py, &r))
    }
    fn residuals<'py>(&self, py: Python<'py>, group: Vec<usize>) -> PyResult<Bound<'py, PyArray1<NpC64>>> {
        Ok(cvec_py(py, &self.inner.residuals(&group).map_err(to_py)?))
    }
    /// Moduli of the complex weighted correlations.
    fn wcor<'py>(&self, py: Python<'py>, groups: Vec<Vec<usize>>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(mat_to_py(py, &self.inner.wcor(&groups).map_err(to_py)?))
    }
    fn wnorm(&self) -> f64 {
        self.inner.wnorm()
    }
    fn contributions<'py>(&self, py: Python<'py>, idx: Vec<usize>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec_py(py, self.inner.contributions(&idx).map_err(to_py)?))
    }
    fn lrr<'py>(&self, py: Python<'py>, group: Vec<usize>) -> PyResult<Bound<'py, PyArray1<NpC64>>> {
        Ok(cvec_py(py, &self.inner.lrr(&group).map_err(to_py)?))
    }
    fn rforecast<'py>(&self, py: Python<'py>, group: Vec<usize>, steps: usize) -> PyResult<Bound<'py, PyArray1<NpC64>>> {
        Ok(cvec_py(py, &self.inner.rforecast(&group, steps).map_err(to_py)?))
    }
    fn vforecast<'py>(&self, py: Python<'py>, group: Vec<usize>, steps: usize) -> PyResult<Bound<'py, PyArray1<NpC64>>> {
        let m = &self.inner;
        let r = py.detach(|| m.vforecast(&group, steps)).map_err(to_py)?;
        Ok(cvec_py(py, &r))
    }
    #[pyo3(signature = (group, tls=false))]
    fn parestimate<'py>(&self, py: Python<'py>, group: Vec<usize>, tls: bool) -> PyResult<Bound<'py, PyDict>> {
        params_dict(py, &self.inner.parestimate(&group, tls).map_err(to_py)?)
    }
    #[pyo3(signature = (rank, correct=true, tol=1e-6, maxiter=0))]
    fn cadzow<'py>(&self, py: Python<'py>, rank: usize, correct: bool, tol: f64, maxiter: usize) -> PyResult<Bound<'py, PyArray1<NpC64>>> {
        let m = &self.inner;
        let r = py.detach(|| m.cadzow(rank, correct, tol, maxiter)).map_err(to_py)?;
        Ok(cvec_py(py, &r))
    }
    #[pyo3(signature = (group, reconstructed_base=false, simultaneous=false))]
    fn gapfill<'py>(&self, py: Python<'py>, group: Vec<usize>, reconstructed_base: bool, simultaneous: bool) -> PyResult<Bound<'py, PyArray1<NpC64>>> {
        let m = &self.inner;
        let r = py.detach(|| m.gapfill(&group, reconstructed_base, simultaneous)).map_err(to_py)?;
        Ok(cvec_py(py, &r))
    }
    #[pyo3(signature = (groups, fill=None, tol=1e-6, maxiter=0))]
    fn igapfill<'py>(&self, py: Python<'py>, groups: Vec<Vec<usize>>, fill: Option<(f64, f64)>, tol: f64, maxiter: usize) -> PyResult<Vec<Bound<'py, PyArray1<NpC64>>>> {
        let m = &self.inner;
        let f = fill.map(|(re, im)| Complex64::new(re, im));
        let r = py.detach(|| m.igapfill(&groups, f, tol, maxiter)).map_err(to_py)?;
        Ok(r.iter().map(|v| cvec_py(py, v)).collect())
    }
    #[pyo3(signature = (nested_groups, filter=None, gamma=f64::INFINITY, normalize=true))]
    fn fossa<'py>(&self, py: Python<'py>, nested_groups: Vec<Vec<usize>>, filter: Option<Vec<f64>>, gamma: f64, normalize: bool) -> PyResult<Bound<'py, PyDict>> {
        let f = filter.unwrap_or_else(|| vec![-1.0, 1.0]);
        wrap_cossa(py, self.inner.fossa(&nested_groups, &f, gamma, normalize).map_err(to_py)?)
    }
    #[pyo3(signature = (nested_groups, tol=1e-5, kappa=Some(2.0), maxiter=100, kappa_balance=0.5))]
    fn iossa<'py>(&self, py: Python<'py>, nested_groups: Vec<Vec<usize>>, tol: f64, kappa: Option<f64>, maxiter: usize, kappa_balance: f64) -> PyResult<Bound<'py, PyDict>> {
        let m = &self.inner;
        let r = py.detach(|| m.iossa(&nested_groups, tol, kappa, maxiter, kappa_balance)).map_err(to_py)?;
        wrap_cossa(py, r)
    }
    fn eossa<'py>(&self, py: Python<'py>, nested_groups: Vec<Vec<usize>>, k: usize) -> PyResult<Bound<'py, PyDict>> {
        wrap_cossa(py, self.inner.eossa(&nested_groups, k).map_err(to_py)?)
    }
    fn owcor<'py>(&self, py: Python<'py>, groups: Vec<Vec<usize>>, set: Vec<usize>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(mat_to_py(py, &self.inner.owcor(&groups, &set).map_err(to_py)?))
    }
}

// ------------------------------------------------------------------ misc

/// Heterogeneity matrix (Rssa `hmatr`).
#[pyfunction]
#[pyo3(signature = (x, b=None, t=None, window=None, neig=10))]
fn hmatr<'py>(py: Python<'py>, x: Vec<f64>, b: Option<usize>, t: Option<usize>, window: Option<usize>, neig: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n = x.len();
    let (b, t) = (b.unwrap_or(n / 4), t.unwrap_or(n / 4));
    let l = window.unwrap_or(b / 2);
    let h = py.detach(|| misc::hmatr(&x, b, t, l, neig)).map_err(to_py)?;
    Ok(mat_to_py(py, &h))
}

/// Methods available for each kind of decomposition (Rssa `ssa.capabilities`).
#[pyfunction]
fn capabilities(kind: &str) -> Vec<&'static str> {
    misc::capabilities(kind)
}

pub fn parse_interval(s: &str) -> PyResult<Interval> {
    match s {
        "none" => Ok(Interval::None),
        "confidence" => Ok(Interval::Confidence),
        "prediction" => Ok(Interval::Prediction),
        o => Err(PyValueError::new_err(format!("interval must be 'none', 'confidence' or 'prediction', got {o:?}"))),
    }
}

pub fn parse_base(s: &str) -> PyResult<Base> {
    p_base(s)
}

pub fn parse_direction(s: &str) -> PyResult<Direction> {
    p_dir(s)
}

pub fn parse_subspace(s: &str) -> PyResult<Subspace> {
    p_subspace(s)
}

pub fn parse_solve(s: &str) -> PyResult<SolveMethod> {
    p_solve(s)
}

pub fn params_to_dict<'py>(py: Python<'py>, p: &Params) -> PyResult<Bound<'py, PyDict>> {
    params_dict(py, p)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NdSsa>()?;
    m.add_class::<MssaList>()?;
    m.add_class::<ShapedSsa>()?;
    m.add_class::<Cssa>()?;
    m.add_function(wrap_pyfunction!(ball_mask, m)?)?;
    m.add_function(wrap_pyfunction!(simplex_mask, m)?)?;
    m.add_function(wrap_pyfunction!(igapfill, m)?)?;
    m.add_function(wrap_pyfunction!(summarize_gaps, m)?)?;
    m.add_function(wrap_pyfunction!(complete_lag_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(hmatr, m)?)?;
    m.add_function(wrap_pyfunction!(capabilities, m)?)?;
    Ok(())
}
