"""Estimator-specific noise precision and magnitude-noise checks.

These helpers do not change the primary voxel classifier or its RMS scale.
Reused integration nodes are not a model of spatially shared random noise.
"""

from __future__ import annotations

import numpy as np
from functools import lru_cache
from scipy.interpolate import PchipInterpolator
from scipy.special import i0e, i1e


def median_temperature_variance(
    tau: np.ndarray,
    eta: np.ndarray,
    standard_normal_draws: np.ndarray,
    *,
    temperatures: np.ndarray | None = None,
    batch_size: int = 128,
) -> np.ndarray:
    """Integrate Var(median(T_i + tau_i*Z0 - eta_i*Zi)) per voxel.

    ``temperatures=None`` gives the common-temperature noise model. Supplying
    observed shell estimates gives a plug-in first-order precision calculation.
    Selection is held fixed in both cases. Variance is a population moment of
    the quadrature nodes (ddof=0), not an ordinary Monte Carlo sample estimate.
    """
    tau = np.asarray(tau, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    normals = np.asarray(standard_normal_draws, dtype=np.float64)
    if tau.ndim != 2 or tau.shape[1] != 4 or eta.shape != tau.shape:
        raise ValueError("tau and eta must both have shape (n_voxels, 4)")
    if not np.all(np.isfinite(tau)) or not np.all(np.isfinite(eta)):
        raise ValueError("noise components must be finite")
    if np.any(tau < 0) or np.any(eta < 0):
        raise ValueError("noise standard deviations cannot be negative")
    if normals.ndim != 2 or normals.shape[1] != 5 or len(normals) < 2:
        raise ValueError("draws must have shape (n_draws, 5), n_draws >= 2")
    if not np.all(np.isfinite(normals)) or batch_size < 1:
        raise ValueError("draws must be finite and batch_size must be positive")
    if temperatures is None:
        offsets = np.zeros_like(tau)
    else:
        temperatures = np.asarray(temperatures, dtype=np.float64)
        if temperatures.shape != tau.shape or not np.all(np.isfinite(temperatures)):
            raise ValueError("temperatures must be finite and match tau")
        offsets = temperatures - np.median(temperatures, axis=1)[:, None]

    result = np.empty(len(tau), dtype=np.float64)
    for start in range(0, len(tau), batch_size):
        stop = min(start + batch_size, len(tau))
        error = normals[:, None, :1] * tau[None, start:stop, :]
        error -= normals[:, None, 1:] * eta[None, start:stop, :]
        error += offsets[None, start:stop, :]
        # The even-sample median is the mean of the two middle order statistics.
        error.sort(axis=2)
        medians = 0.5 * (error[:, :, 1] + error[:, :, 2])
        result[start:stop] = np.var(medians, axis=0)
    return result


def regional_noise_se(voxel_variances: np.ndarray, *, spatial_factor: float = 4.84) -> float:
    """Approximate SE of a fixed regional mean from marginal voxel variances.

    The spatial factor is an explicit approximation, not a fitted covariance
    matrix. Do not compute variance of a mean across reused integration nodes:
    that would introduce artificial between-voxel covariance.
    """
    variances = np.asarray(voxel_variances, dtype=np.float64)
    if variances.ndim != 1 or not len(variances):
        raise ValueError("voxel_variances must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(variances)) or np.any(variances < 0):
        raise ValueError("voxel variances must be finite and nonnegative")
    if not np.isfinite(spatial_factor) or spatial_factor <= 0:
        raise ValueError("spatial_factor must be finite and positive")
    return float(np.sqrt(spatial_factor * np.sum(variances)) / len(variances))


def rice_unit_moments(snr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of Rice(A=snr, complex-channel SD=1).

    Scaled Bessel functions avoid exponential overflow. The high-SNR series
    avoids cancellation in the variance. These are magnitude-domain moments,
    not the residual variance after PCA denoising.
    """
    snr = np.asarray(snr, dtype=np.float64)
    if not np.all(np.isfinite(snr)) or np.any(snr < 0):
        raise ValueError("SNR must be finite and nonnegative")
    t = snr**2 / 4
    mean = np.sqrt(np.pi/2) * ((1 + 2*t)*i0e(t) + 2*t*i1e(t))
    variance = 2 + snr**2 - mean**2
    large = snr > 50
    r = snr[large]
    mean[large] = r + 1/(2*r) + 1/(8*r**3) + 3/(16*r**5) + 75/(128*r**7)
    variance[large] = 1 - 1/(2*r**2) - 1/(2*r**4) - 11/(8*r**6)
    return mean, variance


@lru_cache(maxsize=1)
def _rice_inverse_tables() -> tuple[PchipInterpolator, PchipInterpolator]:
    r = np.concatenate([np.array([0.0]), np.arange(0.005, 20, 0.005),
                        np.arange(20, 100, 0.01), np.arange(100, 1000.05, 0.05)])
    mean, variance = rice_unit_moments(r)
    ratio = mean / np.sqrt(variance)
    if not np.all(np.diff(mean) > 0) or not np.all(np.diff(ratio) > 0):
        raise AssertionError("Rice inverse lookup must be strictly monotone")
    return (PchipInterpolator(mean, r, extrapolate=False),
            PchipInterpolator(ratio, r, extrapolate=False))


def invert_rice_mean(mean: np.ndarray, complex_sigma: np.ndarray) -> np.ndarray:
    """Recover amplitude from the magnitude mean and KNOWN complex-channel SD.

    This is an oracle benchmark when the supplied noise map instead estimates
    magnitude SD. It must not silently be used with that map in a new pipeline.
    Values below the Rayleigh mean are assigned zero amplitude.
    """
    mean, sigma = np.broadcast_arrays(np.asarray(mean, dtype=float),
                                      np.asarray(complex_sigma, dtype=float))
    if np.any(~np.isfinite(mean)) or np.any(mean < 0):
        raise ValueError("mean must be finite and nonnegative")
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("complex_sigma must be finite and positive")
    ratio = np.maximum(mean/sigma, np.sqrt(np.pi/2))
    inverse, _ = _rice_inverse_tables()
    if np.any(ratio > inverse.x[-1]):
        raise ValueError("SNR exceeds validated inverse-table range")
    return sigma * inverse(ratio)


def invert_rice_magnitude_moments(
    mean: np.ndarray, magnitude_sd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover amplitude and complex SD from magnitude mean and magnitude SD.

    Numerically inverts the single-coil Rice moment identities (Koay/Basser).
    The magnitude SD is the input-domain noise estimate, not denoised residual
    SD. Ratios below the Rayleigh limit are projected to zero amplitude.
    """
    mean, sd = np.broadcast_arrays(np.asarray(mean, dtype=float),
                                   np.asarray(magnitude_sd, dtype=float))
    if np.any(~np.isfinite(mean)) or np.any(mean < 0):
        raise ValueError("mean must be finite and nonnegative")
    if np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError("magnitude_sd must be finite and positive")
    _, inverse = _rice_inverse_tables()
    rayleigh_ratio = np.sqrt(np.pi/2) / np.sqrt(2-np.pi/2)
    ratio = np.maximum(mean/sd, rayleigh_ratio)
    if np.any(ratio > inverse.x[-1]):
        raise ValueError("SNR exceeds validated inverse-table range")
    snr = inverse(ratio)
    unit_mean, _ = rice_unit_moments(snr)
    complex_sigma = mean/unit_mean
    return snr*complex_sigma, complex_sigma
