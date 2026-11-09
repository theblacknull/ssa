//! Native core of the `ssa_rs` Python package.
//!
//! The pure-Python layer (`ssa-python/ssa_rs/*.py`) reproduces the public API of
//! py-ssa-lib (classes, attribute names, pandas handling, custom strategies)
//! and delegates every numerical routine to this module.

use faer::Mat;
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use ssa_rs::svd::{Decomposition, PowerIterations};
use ssa_rs::{
    EigenSvd, Estimator, FullSvd, Matrix, PrecomputedWeights, RandomizedSvd, SsaError, SsaModel,
    SvdStrategy,
};

mod rssa_py;

pyo3::create_exception!(_core, LinAlgError, PyValueError);

fn to_py(e: SsaError) -> PyErr {
    match e {
        SsaError::InvalidArgument(m) => PyValueError::new_err(m),
        SsaError::IndexOutOfRange { .. } => PyIndexError::new_err(e.to_string()),
        SsaError::LinAlg(m) => LinAlgError::new_err(m),
        SsaError::Io(m) => pyo3::exceptions::PyIOError::new_err(m),
    }
}

fn mat_from(a: &PyReadonlyArray2<'_, f64>) -> Matrix {
    let v = a.as_array();
    Mat::from_fn(v.nrows(), v.ncols(), |i, j| v[(i, j)])
}

fn mat_to_py<'py>(py: Python<'py>, m: &Matrix) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)]).into_pyarray(py)
}

fn vec_to_py<'py>(py: Python<'py>, v: &[f64]) -> Bound<'py, PyArray1<f64>> {
    Array1::from_vec(v.to_vec()).into_pyarray(py)
}

/// Decomposition computed elsewhere (custom Python `SVDStrategy`).
struct Given(Decomposition);

impl SvdStrategy for Given {
    fn decompose(&self, _t: &ssa_rs::Trajectory<'_>) -> ssa_rs::Result<Decomposition> {
        Ok(self.0.clone())
    }
}

/// Fitted model (attributes and methods of a fitted `BaseSSA`).
#[pyclass(module = "ssa_rs._core", frozen)]
struct Model {
    inner: SsaModel,
}

#[pymethods]
impl Model {
    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }
    #[getter]
    fn s(&self) -> usize {
        self.inner.s
    }
    #[getter]
    fn l(&self) -> usize {
        self.inner.l
    }
    #[getter]
    fn k(&self) -> usize {
        self.inner.k
    }
    #[getter]
    fn d(&self) -> usize {
        self.inner.d
    }
    #[getter]
    fn x_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat_to_py(py, &self.inner.x_s)
    }
    #[getter]
    fn x_ss<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat_to_py(py, &self.inner.x_ss())
    }
    #[getter]
    fn u<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat_to_py(py, &self.inner.u)
    }
    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat_to_py(py, &self.inner.v)
    }
    #[getter]
    fn sigma<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_to_py(py, &self.inner.sigma)
    }
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_to_py(py, &self.inner.weights)
    }
    #[getter]
    fn rel_contribution<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_to_py(py, &self.inner.rel_contribution)
    }
    #[getter]
    fn cumsum_contr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_to_py(py, &self.inner.cumsum_contr)
    }

    fn hankel_weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let w: Vec<i64> = self.inner.construct_hankel_weights().into_iter().map(|v| v as i64).collect();
        Array1::from_vec(w).into_pyarray(py)
    }

    fn elementary_matrix<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.inner.elementary_matrix(i).map_err(to_py)?;
        Ok(mat_to_py(py, &m))
    }

    /// `(d, L, K*S)` array of all elementary matrices.
    fn elementary_matrices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let m = &self.inner;
        let mats = py.detach(|| m.elementary_matrices());
        Array3::from_shape_fn((m.d, m.l, m.k * m.s), |(i, r, c)| mats[i][(r, c)]).into_pyarray(py)
    }

    fn reconstruct<'py>(&self, py: Python<'py>, idx: Vec<usize>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = &self.inner;
        let r = py.detach(|| m.reconstruct_components(&idx)).map_err(to_py)?;
        Ok(mat_to_py(py, &r))
    }

    /// `(d, N, S)` array of elementary reconstructed components.
    fn elementary_components<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let m = &self.inner;
        let comps = py.detach(|| m.reconstruct_elementary_components());
        Array3::from_shape_fn((m.d, m.n, m.s), |(i, t, s)| comps[i][(t, s)]).into_pyarray(py)
    }

    fn wcorr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let m = &self.inner;
        let w = py.detach(|| m.compute_weighted_correlation_matrix());
        mat_to_py(py, &w)
    }

    /// Returns `(mu_real, mu_imag, rho, omega)`.
    #[allow(clippy::type_complexity)]
    fn esprit<'py>(
        &self,
        py: Python<'py>,
        idx: Vec<usize>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)>
    {
        let (mu, rho, omega) = self.inner.estimate_esprit(&idx).map_err(to_py)?;
        let re: Vec<f64> = mu.iter().map(|z| z.re).collect();
        let im: Vec<f64> = mu.iter().map(|z| z.im).collect();
        Ok((vec_to_py(py, &re), vec_to_py(py, &im), vec_to_py(py, &rho), vec_to_py(py, &omega)))
    }

    fn lrr<'py>(&self, py: Python<'py>, idx: Vec<usize>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec_to_py(py, &self.inner.estimate_lrr(&idx).map_err(to_py)?))
    }

    fn l_forecast<'py>(
        &self,
        py: Python<'py>,
        steps: usize,
        idx: Vec<usize>,
        return_full: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = &self.inner;
        let r = py.detach(|| m.l_forecast(steps, &idx, return_full)).map_err(to_py)?;
        Ok(mat_to_py(py, &r))
    }

    fn k_forecast<'py>(
        &self,
        py: Python<'py>,
        steps: usize,
        idx: Vec<usize>,
        return_full: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = &self.inner;
        let r = py.detach(|| m.k_forecast(steps, &idx, return_full)).map_err(to_py)?;
        Ok(mat_to_py(py, &r))
    }

    // ---- Rssa-compatible methods of the plain 1-D / MSSA model ----------

    fn nsigma(&self) -> usize {
        self.inner.nsigma()
    }

    /// Signed weighted correlations between reconstructed groups (all
    /// components one by one when `groups` is None).
    #[pyo3(signature = (groups=None))]
    fn wcor_signed<'py>(&self, py: Python<'py>, groups: Option<Vec<Vec<usize>>>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = &self.inner;
        let w = py.detach(|| m.wcor(groups.as_deref())).map_err(to_py)?;
        Ok(mat_to_py(py, &w))
    }

    fn wnorm(&self) -> f64 {
        self.inner.wnorm()
    }

    fn contributions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_to_py(py, &self.inner.contributions())
    }

    fn frobenius_cor<'py>(&self, py: Python<'py>, groups: Vec<Vec<usize>>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(mat_to_py(py, &self.inner.frobenius_cor(&groups).map_err(to_py)?))
    }

    fn residuals<'py>(&self, py: Python<'py>, idx: Vec<usize>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(mat_to_py(py, &self.inner.residuals(&idx).map_err(to_py)?))
    }

    fn grouping_auto_wcor(&self, components: Vec<usize>, nclust: usize) -> PyResult<Vec<Vec<usize>>> {
        self.inner.grouping_auto_wcor(&components, nclust).map_err(to_py)
    }

    /// Periodogram-based auto grouping.  `bins`: an int (equal bands),
    /// a list of upper edges, or a list of `(lower, upper)` pairs.
    #[pyo3(signature = (components, bins=None, threshold=None, base="series", method="constant", drop_empty=true))]
    fn grouping_auto_pgram(
        &self,
        components: Vec<usize>,
        bins: Option<&Bound<'_, PyAny>>,
        threshold: Option<Vec<f64>>,
        base: &str,
        method: &str,
        drop_empty: bool,
    ) -> PyResult<(Vec<Vec<usize>>, Vec<Vec<f64>>, bool)> {
        use ssa_rs::rssa::grouping::{FreqBins, PgramBase, PgramMethod};
        let fb = match bins {
            None => FreqBins::Upper(vec![0.1]),
            Some(b) => {
                if let Ok(n) = b.extract::<usize>() {
                    FreqBins::Equal(n)
                } else if let Ok(r) = b.extract::<Vec<(f64, f64)>>() {
                    FreqBins::Ranges(r)
                } else {
                    FreqBins::Upper(b.extract::<Vec<f64>>()?)
                }
            }
        };
        let pb = match base {
            "series" => PgramBase::Series,
            "eigen" => PgramBase::Eigen,
            "factor" => PgramBase::Factor,
            o => return Err(PyValueError::new_err(format!("unknown base {o:?}"))),
        };
        let pm = match method {
            "constant" => PgramMethod::Constant,
            "linear" => PgramMethod::Linear,
            o => return Err(PyValueError::new_err(format!("unknown method {o:?}"))),
        };
        let r = self
            .inner
            .grouping_auto_pgram(&components, pb, &fb, &threshold.unwrap_or_else(|| vec![0.0]), pm, drop_empty)
            .map_err(to_py)?;
        let c = &r.contributions;
        let cm = (0..c.nrows()).map(|i| (0..c.ncols()).map(|j| c[(i, j)]).collect()).collect();
        Ok((r.groups, cm, r.splitting))
    }

    /// ESPRIT with the Rssa `parestimate` conventions (LS / TLS, column /
    /// row subspace); returns a dict of complex `roots`, `periods`,
    /// `frequencies`, `moduli`, `rates`.
    #[pyo3(signature = (group, subspace="column", method="ls", normalize=false))]
    fn parestimate<'py>(&self, py: Python<'py>, group: Vec<usize>, subspace: &str, method: &str, normalize: bool) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let p = self
            .inner
            .parestimate_esprit(&group, rssa_py::parse_subspace(subspace)?, rssa_py::parse_solve(method)?, normalize)
            .map_err(to_py)?;
        rssa_py::params_to_dict(py, &p)
    }

    #[pyo3(signature = (group, reverse=false))]
    fn lrr_rssa<'py>(&self, py: Python<'py>, group: Vec<usize>, reverse: bool) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec_to_py(py, &self.inner.lrr(&group, reverse).map_err(to_py)?))
    }

    /// Recurrent forecast (Rssa `rforecast`); returns only the new values,
    /// shape `(steps, S)`.
    #[pyo3(signature = (group, steps, base="reconstructed", direction="column", reverse=false))]
    fn rforecast<'py>(&self, py: Python<'py>, group: Vec<usize>, steps: usize, base: &str, direction: &str, reverse: bool) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = &self.inner;
        let (b, d) = (rssa_py::parse_base(base)?, rssa_py::parse_direction(direction)?);
        let r = py.detach(|| m.rforecast(&group, steps, b, d, reverse)).map_err(to_py)?;
        Ok(mat_to_py(py, &r))
    }

    #[pyo3(signature = (group, steps, direction="column"))]
    fn vforecast<'py>(&self, py: Python<'py>, group: Vec<usize>, steps: usize, direction: &str) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = &self.inner;
        let d = rssa_py::parse_direction(direction)?;
        let r = py.detach(|| m.vforecast(&group, steps, d)).map_err(to_py)?;
        Ok(mat_to_py(py, &r))
    }

    /// Bootstrap forecast of one series: `(mean, lower, upper)`.
    #[pyo3(signature = (group, steps, replicates=100, level=0.95, vector=false, prediction=false, seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn bforecast<'py>(
        &self,
        py: Python<'py>,
        group: Vec<usize>,
        steps: usize,
        replicates: usize,
        level: f64,
        vector: bool,
        prediction: bool,
        seed: u64,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let m = &self.inner;
        let (a, b, c) = py.detach(|| m.bforecast(&group, steps, replicates, level, vector, prediction, seed)).map_err(to_py)?;
        Ok((vec_to_py(py, &a), vec_to_py(py, &b), vec_to_py(py, &c)))
    }

    /// Rssa `forecast` / `predict`: dict with `mean`, `lower`, `upper`
    /// (None without intervals), `fitted`, `residuals`, `level`.
    #[pyo3(signature = (group, steps, vector=false, interval="none", level=0.95, replicates=100, seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn forecast<'py>(
        &self,
        py: Python<'py>,
        group: Vec<usize>,
        steps: usize,
        vector: bool,
        interval: &str,
        level: f64,
        replicates: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let m = &self.inner;
        let iv = rssa_py::parse_interval(interval)?;
        let f = py.detach(|| m.forecast(&group, steps, vector, iv, level, replicates, seed)).map_err(to_py)?;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("mean", vec_to_py(py, &f.mean))?;
        d.set_item("lower", f.lower.as_ref().map(|v| vec_to_py(py, v)))?;
        d.set_item("upper", f.upper.as_ref().map(|v| vec_to_py(py, v)))?;
        d.set_item("fitted", vec_to_py(py, &f.fitted))?;
        d.set_item("residuals", vec_to_py(py, &f.residuals))?;
        d.set_item("level", f.level)?;
        Ok(d)
    }

    /// Cadzow iterations: the rank-`rank` Hankel approximation, `(N, S)`.
    #[pyo3(signature = (rank, correct=true, tol=1e-6, maxiter=0))]
    fn cadzow<'py>(&self, py: Python<'py>, rank: usize, correct: bool, tol: f64, maxiter: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = &self.inner;
        let r = py.detach(|| m.cadzow(rank, correct, tol, maxiter)).map_err(to_py)?;
        Ok(mat_to_py(py, &r))
    }

    /// Filter-adjusted O-SSA; returns `(model, groups, set)`.
    #[pyo3(signature = (nested_groups, filter=None, gamma=f64::INFINITY, normalize=true))]
    fn fossa(&self, nested_groups: Vec<Vec<usize>>, filter: Option<Vec<f64>>, gamma: f64, normalize: bool) -> PyResult<(Model, Vec<Vec<usize>>, Vec<usize>)> {
        let f = filter.unwrap_or_else(|| vec![-1.0, 1.0]);
        let r = self.inner.fossa(&nested_groups, &f, gamma, normalize).map_err(to_py)?;
        Ok((Model { inner: r.model }, r.groups, r.set))
    }

    /// Iterative O-SSA; returns `(model, groups, set, iterations, converged)`.
    #[pyo3(signature = (nested_groups, tol=1e-5, kappa=Some(2.0), maxiter=100, kappa_balance=0.5))]
    fn iossa(&self, py: Python<'_>, nested_groups: Vec<Vec<usize>>, tol: f64, kappa: Option<f64>, maxiter: usize, kappa_balance: f64) -> PyResult<(Model, Vec<Vec<usize>>, Vec<usize>, usize, bool)> {
        let m = &self.inner;
        let r = py.detach(|| m.iossa(&nested_groups, tol, kappa, maxiter, kappa_balance)).map_err(to_py)?;
        Ok((Model { inner: r.model }, r.groups, r.set, r.iterations, r.converged))
    }

    /// ESPRIT-based O-SSA; returns `(model, groups, set)`.
    #[pyo3(signature = (nested_groups, k, method="ls"))]
    fn eossa(&self, nested_groups: Vec<Vec<usize>>, k: usize, method: &str) -> PyResult<(Model, Vec<Vec<usize>>, Vec<usize>)> {
        let r = self.inner.eossa(&nested_groups, k, rssa_py::parse_solve(method)?).map_err(to_py)?;
        Ok((Model { inner: r.model }, r.groups, r.set))
    }

    fn owcor<'py>(&self, py: Python<'py>, groups: Vec<Vec<usize>>, set: Vec<usize>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(mat_to_py(py, &self.inner.owcor(&groups, &set).map_err(to_py)?))
    }

    /// Keeps `neig` triples (continues a truncated decomposition).
    fn continue_decomposition(&self, py: Python<'_>, neig: usize) -> PyResult<Model> {
        let m = &self.inner;
        Ok(Model { inner: py.detach(|| m.continue_decomposition(neig)).map_err(to_py)? })
    }
}

/// Fits SSA/MSSA on an `(N, S)` array.
///
/// `method`: "full" | "eigen" | "randomized" | "given" (then `u`, `sigma`,
/// `vt` must be supplied).  `n_iter = None` means sklearn "auto".
#[pyfunction]
#[pyo3(signature = (x, l, weights, method="full", n_components=None, n_oversamples=10, n_iter=None, random_state=None, u=None, sigma=None, vt=None))]
#[allow(clippy::too_many_arguments)]
fn fit(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    l: Option<usize>,
    weights: Vec<f64>,
    method: &str,
    n_components: Option<usize>,
    n_oversamples: usize,
    n_iter: Option<usize>,
    random_state: Option<u64>,
    u: Option<PyReadonlyArray2<'_, f64>>,
    sigma: Option<PyReadonlyArray1<'_, f64>>,
    vt: Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<Model> {
    let x = mat_from(&x);
    let strategy: Box<dyn SvdStrategy> = match method {
        "full" => Box::new(FullSvd),
        "eigen" => Box::new(EigenSvd),
        "toeplitz" => Box::new(ssa_rs::ToeplitzSvd { neig: n_components }),
        "lanczos" => {
            let k = n_components.ok_or_else(|| PyValueError::new_err("n_components is required"))?;
            Box::new(ssa_rs::LanczosSvd::new(k))
        }
        "randomized" => {
            let k = n_components.ok_or_else(|| PyValueError::new_err("n_components is required"))?;
            Box::new(RandomizedSvd {
                n_components: k,
                n_oversamples,
                n_iter: n_iter.map_or(PowerIterations::Auto, PowerIterations::Fixed),
                random_state,
            })
        }
        "given" => {
            let (u, s, vt) = match (u, sigma, vt) {
                (Some(u), Some(s), Some(vt)) => (mat_from(&u), s.as_array().to_vec(), mat_from(&vt)),
                _ => return Err(PyValueError::new_err("u, sigma and vt are required for method='given'")),
            };
            if u.ncols() != s.len() || vt.nrows() != s.len() {
                return Err(PyValueError::new_err("inconsistent decomposition shapes"));
            }
            Box::new(Given(Decomposition { u, sigma: s, v: vt.transpose().to_owned() }))
        }
        other => return Err(PyValueError::new_err(format!("unknown method {other:?}"))),
    };
    let est = Estimator { decomposition_strategy: strategy, weighting_strategy: Box::new(PrecomputedWeights::new(weights)) };
    let model = py.detach(|| est.fit_matrix(x, l)).map_err(to_py)?;
    Ok(Model { inner: model })
}

/// Anti-diagonal averaging of any 2-D array.
#[pyfunction]
fn diagonal_averaging<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    let m = mat_from(&x);
    vec_to_py(py, &ssa_rs::diagonal_averaging(m.as_ref()))
}

/// Trajectory matrix of an `(N, S)` array.
#[pyfunction]
fn trajectory_matrix<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    l: usize,
    weights: Vec<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let m = mat_from(&x);
    if l == 0 || l > m.nrows() || weights.len() != m.ncols() {
        return Err(PyValueError::new_err("invalid L or weights"));
    }
    Ok(mat_to_py(py, &ssa_rs::construct_trajectory_matrix(m.as_ref(), l, &weights)))
}

#[pyfunction]
#[pyo3(signature = (residuals, forecasts, alpha=0.05, n_sims=1000, seed=None))]
fn bootstrap_prediction_intervals<'py>(
    py: Python<'py>,
    residuals: Vec<f64>,
    forecasts: Vec<f64>,
    alpha: f64,
    n_sims: usize,
    seed: Option<u64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (lo, hi) = py
        .detach(|| ssa_rs::bootstrap_prediction_intervals(&residuals, &forecasts, alpha, n_sims, seed))
        .map_err(to_py)?;
    Ok((vec_to_py(py, &lo), vec_to_py(py, &hi)))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(diagonal_averaging, m)?)?;
    m.add_function(wrap_pyfunction!(trajectory_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_prediction_intervals, m)?)?;
    rssa_py::register(m)?;
    m.add("LinAlgError", m.py().get_type::<LinAlgError>())?;
    Ok(())
}
