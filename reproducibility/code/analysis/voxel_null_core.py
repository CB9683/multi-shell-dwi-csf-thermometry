#!/usr/bin/env python3
"""Numerical helpers for voxel-conditioned four-shell noise nulls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats
from scipy.special import ndtri
from scipy.stats import qmc


@dataclass(frozen=True)
class AdaptiveNullResult:
    """Voxelwise null thresholds and classifications after adaptive refinement."""

    null_median: np.ndarray
    retained: np.ndarray
    refined: np.ndarray


def compute_temperature_error_components(
    *,
    s0: np.ndarray,
    sb: np.ndarray,
    sigma: np.ndarray,
    b_value: float,
    n_b0: int,
    n_shell: int,
    arrhenius_a: float = 2732.0,
    arrhenius_b: float = 19.649,
    minimum_adc: float = 1e-6,
    maximum_sigma_temperature: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return temperature and shared-b0/shell-specific error components.

    The first-order temperature error for one shell is represented as
    ``tau * Z0 - eta * Zi``, where ``Z0`` is common to every shell and ``Zi``
    is shell-specific. ``tau`` and ``eta`` are non-negative standard-deviation
    magnitudes. This propagation is for uncorrected (``4s_raw``) signals.
    """
    if b_value <= 0 or n_b0 < 1 or n_shell < 1:
        raise ValueError("b_value and volume counts must be positive")

    s0_values, sb_values, noise_values = np.broadcast_arrays(
        np.asarray(s0, dtype=np.float64),
        np.asarray(sb, dtype=np.float64),
        np.asarray(sigma, dtype=np.float64),
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        adc = np.log(s0_values / sb_values) / b_value
        log_term = np.log(arrhenius_b / adc)
        temperature = arrhenius_a / log_term - 273.15
        derivative = np.abs(arrhenius_a / (adc * log_term**2))
        tau = derivative * noise_values / (
            b_value * s0_values * np.sqrt(n_b0)
        )
        eta = derivative * noise_values / (
            b_value * sb_values * np.sqrt(n_shell)
        )
        sigma_temperature = np.sqrt(tau**2 + eta**2)

    valid = (
        np.isfinite(s0_values)
        & np.isfinite(sb_values)
        & np.isfinite(noise_values)
        & (s0_values > 0)
        & (sb_values > 0)
        & (noise_values >= 0)
        & np.isfinite(adc)
        & (adc > minimum_adc)
        & np.isfinite(temperature)
        & np.isfinite(tau)
        & np.isfinite(eta)
        & (sigma_temperature > 0)
        & (sigma_temperature < maximum_sigma_temperature)
    )
    return (
        np.where(valid, temperature, np.nan),
        np.where(valid, tau, np.nan),
        np.where(valid, eta, np.nan),
    )


def compute_shell_temperature_and_uncertainty(
    *,
    s0: np.ndarray,
    sb: np.ndarray,
    sigma: np.ndarray,
    b_value: float,
    n_b0: int,
    n_shell: int,
    arrhenius_a: float = 2732.0,
    arrhenius_b: float = 19.649,
    minimum_adc: float = 1e-6,
    maximum_sigma_temperature: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one DWI shell to temperature and propagated uncertainty.

    Noise in the averaged b=0 and diffusion-weighted signals is propagated
    separately because the two means are based on different volume counts.
    Invalid signal combinations are returned as NaN.
    """
    if b_value <= 0 or n_b0 < 1 or n_shell < 1:
        raise ValueError("b_value and volume counts must be positive")

    s0_values, sb_values, noise_values = np.broadcast_arrays(
        np.asarray(s0, dtype=np.float64),
        np.asarray(sb, dtype=np.float64),
        np.asarray(sigma, dtype=np.float64),
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        adc = np.log(s0_values / sb_values) / b_value
        log_term = np.log(arrhenius_b / adc)
        temperature = arrhenius_a / log_term - 273.15
        sigma_adc = (1.0 / b_value) * np.sqrt(
            (noise_values / np.sqrt(n_b0) / s0_values) ** 2
            + (noise_values / np.sqrt(n_shell) / sb_values) ** 2
        )
        derivative = np.abs(arrhenius_a / (adc * log_term**2))
        sigma_temperature = derivative * sigma_adc

    valid = (
        np.isfinite(s0_values)
        & np.isfinite(sb_values)
        & np.isfinite(noise_values)
        & (s0_values > 0)
        & (sb_values > 0)
        & (noise_values >= 0)
        & np.isfinite(adc)
        & (adc > minimum_adc)
        & np.isfinite(temperature)
        & np.isfinite(sigma_temperature)
        & (sigma_temperature > 0)
        & (sigma_temperature < maximum_sigma_temperature)
    )
    return np.where(valid, temperature, np.nan), np.where(
        valid, sigma_temperature, np.nan
    )


def normalise_shell_uncertainties(sigma_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-voxel RMS uncertainty and RMS-normalised shell weights."""
    values = np.asarray(sigma_t, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("sigma_t must have shape (n_voxels, 4)")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("sigma_t must contain finite positive values")

    rms = np.sqrt(np.mean(values**2, axis=1))
    weights = values / rms[:, None]
    return rms, weights


def normalise_temperature_error_components(
    tau: np.ndarray, eta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RMS-normalise shared-b0 and shell-specific temperature errors."""
    shared = np.asarray(tau, dtype=np.float64)
    independent = np.asarray(eta, dtype=np.float64)
    if shared.ndim != 2 or shared.shape[1] != 4 or independent.shape != shared.shape:
        raise ValueError("tau and eta must both have shape (n_voxels, 4)")
    if (
        not np.all(np.isfinite(shared))
        or not np.all(np.isfinite(independent))
        or np.any(shared < 0)
        or np.any(independent <= 0)
    ):
        raise ValueError("tau must be finite/non-negative and eta finite/positive")

    rms = np.sqrt(np.mean(shared**2 + independent**2, axis=1))
    return rms, shared / rms[:, None], independent / rms[:, None]


def temperature_covariance(tau: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Build C = tau tau^T + diag(eta^2) for each four-shell voxel."""
    shared = np.asarray(tau, dtype=np.float64)
    independent = np.asarray(eta, dtype=np.float64)
    if shared.ndim != 2 or shared.shape[1] != 4 or independent.shape != shared.shape:
        raise ValueError("tau and eta must both have shape (n_voxels, 4)")
    if not np.all(np.isfinite(shared)) or not np.all(np.isfinite(independent)):
        raise ValueError("tau and eta must be finite")

    covariance = shared[:, :, None] * shared[:, None, :]
    diagonal = np.arange(4)
    covariance[:, diagonal, diagonal] += independent**2
    return covariance


def mad4(values: np.ndarray) -> np.ndarray:
    """Exact NumPy-style MAD for arrays whose final axis has four values."""
    x = np.asarray(values, dtype=np.float64)
    if x.shape[-1] != 4:
        raise ValueError("values must have four entries on the final axis")

    x0, x1, x2, x3 = (x[..., i] for i in range(4))
    a0, a1 = np.minimum(x0, x1), np.maximum(x0, x1)
    a2, a3 = np.minimum(x2, x3), np.maximum(x2, x3)
    b0, b2 = np.minimum(a0, a2), np.maximum(a0, a2)
    b1, b3 = np.minimum(a1, a3), np.maximum(a1, a3)
    c1, c2 = np.minimum(b1, b2), np.maximum(b1, b2)

    # For sorted a<=b<=c<=d, NumPy's four-value MAD is
    # 0.5 * ((c-b) + min(b-a, d-c)).
    return 0.5 * ((c2 - c1) + np.minimum(c1 - b0, b3 - c2))


def sobol_gaussian_draws(power: int, seed: int, dimension: int = 4) -> np.ndarray:
    """Return deterministic scrambled-Sobol standard-normal draws."""
    if power < 1 or dimension < 1:
        raise ValueError("power and dimension must be positive")
    uniform = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(power)
    np.clip(uniform, 1e-12, 1.0 - 1e-12, out=uniform)
    ndtri(uniform, out=uniform)
    return uniform


def voxel_null_medians(
    weights: np.ndarray,
    standard_normal_draws: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Estimate each voxel's null median z from its relative shell weights."""
    w = np.asarray(weights, dtype=np.float64)
    normals = np.asarray(standard_normal_draws, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != 4:
        raise ValueError("weights must have shape (n_voxels, 4)")
    if normals.ndim != 2 or normals.shape[1] != 4:
        raise ValueError("standard_normal_draws must have shape (n_draws, 4)")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    result = np.empty(len(w), dtype=np.float64)
    for start in range(0, len(w), batch_size):
        stop = min(start + batch_size, len(w))
        simulated = normals[:, None, :] * w[None, start:stop, :]
        result[start:stop] = np.median(mad4(simulated), axis=0)
    return result


def covariance_null_medians(
    tau: np.ndarray,
    eta: np.ndarray,
    standard_normal_draws: np.ndarray,
    batch_size: int = 256,
    draw_chunk_size: Optional[int] = None,
) -> np.ndarray:
    """Estimate null MAD medians for shared-b0 plus shell-specific errors."""
    shared = np.asarray(tau, dtype=np.float64)
    independent = np.asarray(eta, dtype=np.float64)
    normals = np.asarray(standard_normal_draws, dtype=np.float64)
    if shared.ndim != 2 or shared.shape[1] != 4 or independent.shape != shared.shape:
        raise ValueError("tau and eta must both have shape (n_voxels, 4)")
    if normals.ndim != 2 or normals.shape[1] != 5:
        raise ValueError("standard_normal_draws must have shape (n_draws, 5)")
    if batch_size < 1 or (draw_chunk_size is not None and draw_chunk_size < 1):
        raise ValueError("batch_size and draw_chunk_size must be positive")

    result = np.empty(len(shared), dtype=np.float64)
    common_draws = normals[:, 0]
    shell_draws = normals[:, 1:]
    for start in range(0, len(shared), batch_size):
        stop = min(start + batch_size, len(shared))
        # Bound each simulated chunk to 262,144 voxel-draw combinations.
        # The full vector of null MAD values is retained because its exact
        # median is the classifier operating point.
        chunk_size = draw_chunk_size or max(1, 262_144 // (stop - start))
        null_mad = np.empty((len(normals), stop - start), dtype=np.float64)
        for draw_start in range(0, len(normals), chunk_size):
            draw_stop = min(draw_start + chunk_size, len(normals))
            simulated = (
                common_draws[draw_start:draw_stop, None, None]
                * shared[None, start:stop, :]
                + shell_draws[draw_start:draw_stop, None, :]
                * independent[None, start:stop, :]
            )
            null_mad[draw_start:draw_stop] = mad4(simulated)
        result[start:stop] = np.median(null_mad, axis=0)
    return result


def classify_at_null_median(
    z: np.ndarray,
    null_median: np.ndarray,
    temperature: np.ndarray,
    minimum_temperature: float = 25.0,
) -> np.ndarray:
    """Apply P(consistency)>0.5 using the null median operating point."""
    z_values, thresholds, temperatures = np.broadcast_arrays(
        np.asarray(z, dtype=np.float64),
        np.asarray(null_median, dtype=np.float64),
        np.asarray(temperature, dtype=np.float64),
    )
    return (
        np.isfinite(z_values)
        & np.isfinite(thresholds)
        & np.isfinite(temperatures)
        & (temperatures >= minimum_temperature)
        & (z_values < thresholds)
    )


def adaptive_voxel_null_classification(
    *,
    z: np.ndarray,
    weights: np.ndarray,
    temperature: np.ndarray,
    coarse_normals: np.ndarray,
    refined_normals: np.ndarray,
    refinement_margin: float = 0.01,
    batch_size: int = 256,
) -> AdaptiveNullResult:
    """Classify voxels, refining null medians close to the coarse boundary."""
    z_values = np.asarray(z, dtype=np.float64)
    temperatures = np.asarray(temperature, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if z_values.shape != (len(w),) or temperatures.shape != (len(w),):
        raise ValueError("z, temperature, and weights must describe the same voxels")
    if refinement_margin < 0:
        raise ValueError("refinement_margin must be non-negative")

    null_median = voxel_null_medians(w, coarse_normals, batch_size=batch_size)
    refined = np.isfinite(z_values) & (np.abs(z_values - null_median) <= refinement_margin)
    if np.any(refined):
        null_median[refined] = voxel_null_medians(
            w[refined], refined_normals, batch_size=batch_size
        )
    retained = classify_at_null_median(z_values, null_median, temperatures)
    return AdaptiveNullResult(null_median=null_median, retained=retained, refined=refined)


def summarise_region(temperature: np.ndarray, retained: np.ndarray) -> dict[str, float]:
    """Summarise candidate and retained temperatures for one anatomical region."""
    values = np.asarray(temperature, dtype=np.float64)
    keep = np.asarray(retained, dtype=bool)
    if values.shape != keep.shape:
        raise ValueError("temperature and retained must have the same shape")
    selected = values[keep & np.isfinite(values)]
    return {
        "n_candidate": int(len(values)),
        "n_retained": int(len(selected)),
        "temperature_mean": float(np.mean(selected)) if len(selected) else np.nan,
        "temperature_sd": float(np.std(selected, ddof=1)) if len(selected) > 1 else np.nan,
    }


def summarise_paired_regions(lv: np.ndarray, sa: np.ndarray) -> dict[str, float]:
    """Return group inference for paired LV and SA temperatures."""
    lv_values = np.asarray(lv, dtype=np.float64)
    sa_values = np.asarray(sa, dtype=np.float64)
    if lv_values.shape != sa_values.shape:
        raise ValueError("lv and sa must have the same shape")
    finite = np.isfinite(lv_values) & np.isfinite(sa_values)
    lv_values, sa_values = lv_values[finite], sa_values[finite]
    if len(lv_values) < 2:
        raise ValueError("at least two finite subject pairs are required")

    gradient = lv_values - sa_values
    n = len(gradient)
    gradient_mean = float(np.mean(gradient))
    gradient_sd = float(np.std(gradient, ddof=1))
    standard_error = gradient_sd / np.sqrt(n)
    critical = float(stats.t.ppf(0.975, df=n - 1))
    t_result = stats.ttest_1samp(gradient, popmean=0.0)
    n_positive = int(np.sum(gradient > 0))
    sign_result = stats.binomtest(n_positive, n=n, p=0.5, alternative="two-sided")

    return {
        "n_subjects": int(n),
        "lv_mean": float(np.mean(lv_values)),
        "lv_sd": float(np.std(lv_values, ddof=1)),
        "sa_mean": float(np.mean(sa_values)),
        "sa_sd": float(np.std(sa_values, ddof=1)),
        "gradient_mean": gradient_mean,
        "gradient_sd": gradient_sd,
        "gradient_ci_low": gradient_mean - critical * standard_error,
        "gradient_ci_high": gradient_mean + critical * standard_error,
        "paired_t": float(t_result.statistic),
        "paired_t_p": float(t_result.pvalue),
        "cohen_dz": gradient_mean / gradient_sd,
        "n_positive": n_positive,
        "sign_test_p": float(sign_result.pvalue),
    }


def select_top_subjects(
    subjects: np.ndarray, retained_counts: np.ndarray, n: int = 6
) -> list[str]:
    """Select subjects with the largest retained compartments."""
    names = np.asarray(subjects, dtype=str)
    counts = np.asarray(retained_counts, dtype=np.float64)
    if names.shape != counts.shape:
        raise ValueError("subjects and retained_counts must have the same shape")
    if n < 1 or n > len(names):
        raise ValueError("n must be between 1 and the number of subjects")
    order = np.lexsort((names, -counts))
    return names[order[:n]].tolist()


def summarise_subset(
    subjects: np.ndarray,
    sa_temperature: np.ndarray,
    gradients: np.ndarray,
    members: list[str],
) -> dict[str, float]:
    """Summarise SA temperature and gradient for an explicit subject subset."""
    names = np.asarray(subjects, dtype=str)
    sa_values = np.asarray(sa_temperature, dtype=np.float64)
    gradient_values = np.asarray(gradients, dtype=np.float64)
    if not (names.shape == sa_values.shape == gradient_values.shape):
        raise ValueError("subjects, sa_temperature, and gradients must align")
    member_set = set(members)
    selected = np.array([name in member_set for name in names])
    if int(selected.sum()) != len(member_set):
        raise ValueError("every requested member must occur exactly once")

    sa_selected = sa_values[selected]
    gradient_selected = gradient_values[selected]
    return {
        "members": ";".join(members),
        "n_subjects": int(selected.sum()),
        "sa_mean": float(np.mean(sa_selected)),
        "sa_sd": float(np.std(sa_selected, ddof=1)) if len(sa_selected) > 1 else np.nan,
        "gradient_mean": float(np.mean(gradient_selected)),
        "gradient_sd": (
            float(np.std(gradient_selected, ddof=1)) if len(gradient_selected) > 1 else np.nan
        ),
    }


def spearman_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_permutations: int = 999_999,
    seed: int = 42,
    batch_size: int = 100_000,
) -> tuple[float, float]:
    """Two-sided fixed-seed Monte Carlo permutation test for Spearman rho."""
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values, y_values = x_values[finite], y_values[finite]
    if len(x_values) < 3:
        raise ValueError("at least three finite pairs are required")
    if n_permutations < 1 or batch_size < 1:
        raise ValueError("n_permutations and batch_size must be positive")

    x_rank = stats.rankdata(x_values)
    y_rank = stats.rankdata(y_values)
    x_centred = x_rank - x_rank.mean()
    y_centred = y_rank - y_rank.mean()
    denominator = np.sqrt(np.sum(x_centred**2) * np.sum(y_centred**2))
    if denominator == 0:
        raise ValueError("Spearman correlation is undefined for constant input")
    observed = float(np.dot(x_centred, y_centred) / denominator)

    rng = np.random.default_rng(seed)
    exceedances = 0
    completed = 0
    while completed < n_permutations:
        size = min(batch_size, n_permutations - completed)
        permutations = np.argsort(rng.random((size, len(y_centred))), axis=1)
        permuted = y_centred[permutations]
        rho = permuted @ x_centred / denominator
        exceedances += int(np.sum(np.abs(rho) >= abs(observed) - 1e-15))
        completed += size
    p_value = (exceedances + 1.0) / (n_permutations + 1.0)
    return observed, float(p_value)
