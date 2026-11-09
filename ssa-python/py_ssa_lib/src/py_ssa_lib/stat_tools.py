
from numpy.typing import NDArray
import numpy as np

def bootstrap_prediction_intervals(
                                    residuals: NDArray[np.float64],
                                    forecasts: NDArray[np.float64],
                                    alpha: float = 0.05,
                                    N_sims: int = 1000,
                                    random_state: int | np.random.Generator | None = None
                                ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Parameters
    ----------
    residuals : ndarray
        Residuals used for bootstrap resampling.
    forecasts : ndarray
        Point forecasts for future observations.
    alpha : float
        Significance level for the prediction interval.
    N_sims : int
        Number of bootstrap simulations.
    random_state : int | Generator | None
        Seed or generator for reproducible intervals.

    Returns
    -------
    lower_pi : ndarray
        Lower prediction bounds.
    upper_pi : ndarray
        Upper prediction bounds.
    This is a bootstrap method for the prediction intervals discucced in the online textbook of Rob J Hyndman and George Athanasopoulos https://otexts.com/fpp3/prediction-intervals.html.
    It computes lower and upper prediction intervals. This method is model free and requires only the independence of residuals and allows to get reasonable prediction intervals for lower computational costs 
    and milder conditions. This means one can use it for any other model besides the MSSA/SSA.
    
    In the case of the SSA one starts with the computation of residuals between original time series and the reconstruted one. It is crucial to check for the abscence of an autocorrelation within the obtained 
    residuals. Using predefined numbers of components one performs either L- or  K-forecast for M future steps. These values are then passed to the corresponding input parameters. Additionally, one can adjust 
    the confidence level and simulation number as well.

    In the case of MSSA, one has to pick the time series of the interest with the same index. The algorithm doesn't change.



    """
    
    residuals = np.asarray(residuals, dtype=float).ravel()
    forecasts = np.asarray(forecasts, dtype=float)

    if forecasts.ndim == 2 and 1 in forecasts.shape:
        forecasts = forecasts.ravel()

    if forecasts.ndim != 1 or forecasts.size == 0:
        raise ValueError("forecasts must be a non-empty one-dimensional array.")

    if residuals.size == 0 or not np.all(np.isfinite(residuals)):
        raise ValueError("residuals must be non-empty and finite.")

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1].")

    if int(N_sims) != N_sims or N_sims < 1:
        raise ValueError("N_sims must be a positive integer.")

    rng = np.random.default_rng(random_state)

    y_pred = np.tile(forecasts, (int(N_sims), 1))
    residuals_bootstrap = rng.choice(residuals, size=y_pred.shape, replace=True)
    sims = y_pred + residuals_bootstrap

    
    lower_pi = np.percentile(sims, (alpha/2) * 100, axis=0)
    upper_pi = np.percentile(sims, (1 - alpha/2) * 100, axis=0)
    
    return lower_pi, upper_pi