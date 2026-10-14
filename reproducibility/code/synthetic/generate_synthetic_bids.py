#!/usr/bin/env python3
"""Generate a deterministic BIDS-compatible DWI thermometry example.

The digital subject is built entirely from analytic geometry. It contains no
participant-derived image, metadata, template, or fitted parameter.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


ARRHENIUS_A_K = 2732.0
ARRHENIUS_B_MM2_S = 19.649
BIDS_VERSION = "1.11.1"
GENERATOR_VERSION = "1.0.0"
SUBJECT = "sub-synthetic01"
ACQUISITION = "thermometry"
SHAPE = (40, 48, 36)
VOXEL_SIZE_MM = 2.0
SHELL_COUNTS = {0: 11, 200: 20, 500: 20, 700: 20, 1100: 19, 1500: 19}
TEMPERATURES_C = {"LV": 38.8, "SA": 36.3, "Other": 43.0}
LABELS = {"LV": 1, "SA": 2, "Other": 3}
AUTHORS = [
    "Christian Behler",
    "Michael Breakspear",
    "Saadallah Ramadan",
    "Thuy Thanh Dao",
    "Thomas G. Close",
    "Sjoerd Vos",
    "Renate Thienel",
    "Joseph Giorgio",
    "Anna Behler",
    "Michael Fay",
]


def temperature_to_adc(temperature_c: float | np.ndarray) -> float | np.ndarray:
    """Invert the ACSF Arrhenius calibration used by the manuscript."""
    temperature_k = np.asarray(temperature_c, dtype=np.float64) + 273.15
    adc = ARRHENIUS_B_MM2_S / np.exp(ARRHENIUS_A_K / temperature_k)
    if np.ndim(temperature_c) == 0:
        return float(adc)
    return adc


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _nifti_image(data: np.ndarray, description: str) -> nib.Nifti1Image:
    affine = np.diag([VOXEL_SIZE_MM, VOXEL_SIZE_MM, VOXEL_SIZE_MM, 1.0])
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    image.header.set_xyzt_units("mm", "sec")
    image.header["descrip"] = description.encode("ascii")[:79]
    if data.ndim == 4:
        image.header["pixdim"][4] = 4.6
    return image


def _write_nifti(path: Path, data: np.ndarray, description: str) -> None:
    nib.save(_nifti_image(data, description), path)


def _digital_anatomy() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return S0, noise sigma, compartment labels, and temperature truth."""
    grid = np.indices(SHAPE, dtype=np.float64)
    x, y, z = grid
    centre = (np.asarray(SHAPE, dtype=np.float64) - 1.0) / 2.0
    xn = (x - centre[0]) / 17.0
    yn = (y - centre[1]) / 20.0
    zn = (z - centre[2]) / 14.0
    radius = np.sqrt(xn**2 + yn**2 + zn**2)

    brain = radius <= 1.0
    sa = (radius >= 0.82) & (radius <= 0.98)

    left_lv = (
        ((x - (centre[0] - 4.2)) / 3.0) ** 2
        + ((y - centre[1]) / 8.0) ** 2
        + ((z - (centre[2] + 0.5)) / 3.6) ** 2
        <= 1.0
    )
    right_lv = (
        ((x - (centre[0] + 4.2)) / 3.0) ** 2
        + ((y - centre[1]) / 8.0) ** 2
        + ((z - (centre[2] + 0.5)) / 3.6) ** 2
        <= 1.0
    )
    lv = (left_lv | right_lv) & brain

    other = (
        ((x - centre[0]) / 7.0) ** 2
        + ((y - (centre[1] + 1.0)) / 8.0) ** 2
        + ((z - (centre[2] - 9.0)) / 3.5) ** 2
        <= 1.0
    ) & brain

    labels = np.zeros(SHAPE, dtype=np.uint8)
    labels[sa] = LABELS["SA"]
    labels[lv] = LABELS["LV"]
    labels[other & ~lv] = LABELS["Other"]

    s0 = np.full(SHAPE, 5.0, dtype=np.float64)
    tissue_modulation = 1.0 + 0.08 * np.cos(np.pi * xn) * np.cos(np.pi * yn)
    s0[brain] = 1350.0 * tissue_modulation[brain]
    s0[labels > 0] = 2200.0 * (1.0 - 0.05 * radius[labels > 0])

    sigma = np.full(SHAPE, 4.0, dtype=np.float64)
    sigma[brain] = 7.0 + 4.0 * np.clip(radius[brain], 0.0, 1.0) ** 2
    sigma[labels == LABELS["LV"]] = 7.0
    sigma[labels == LABELS["SA"]] += 2.0
    sigma[labels == LABELS["Other"]] = 12.0

    temperature = np.full(SHAPE, np.nan, dtype=np.float32)
    for region, label in LABELS.items():
        temperature[labels == label] = TEMPERATURES_C[region]

    return s0, sigma, labels, temperature


def _fibonacci_directions(count: int, phase: float) -> np.ndarray:
    """Create deterministic, approximately uniform unit vectors."""
    index = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    angle = index * golden_angle + phase
    directions = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


def _gradient_table() -> tuple[np.ndarray, np.ndarray]:
    bvals: list[float] = []
    vectors: list[np.ndarray] = []
    b0_before_shell = {200: 3, 500: 2, 700: 2, 1100: 2, 1500: 2}

    for shell_index, shell in enumerate((200, 500, 700, 1100, 1500)):
        for _ in range(b0_before_shell[shell]):
            bvals.append(0.0)
            vectors.append(np.zeros(3, dtype=np.float64))
        directions = _fibonacci_directions(SHELL_COUNTS[shell], phase=0.37 * shell_index)
        for direction in directions:
            bvals.append(float(shell))
            vectors.append(direction)

    bval_array = np.asarray(bvals, dtype=np.float64)
    bvec_array = np.asarray(vectors, dtype=np.float64).T
    if len(bval_array) != sum(SHELL_COUNTS.values()):
        raise RuntimeError("Internal shell schedule does not match SHELL_COUNTS")
    return bval_array, bvec_array


def _temperature_for_shell(labels: np.ndarray, shell: int) -> np.ndarray:
    temperature = np.full(SHAPE, np.nan, dtype=np.float64)
    temperature[labels == LABELS["LV"]] = TEMPERATURES_C["LV"]

    sa_offsets = {200: 0.12, 500: 0.04, 700: -0.04, 1100: -0.12, 1500: -0.20}
    temperature[labels == LABELS["SA"]] = TEMPERATURES_C["SA"] + sa_offsets[shell]

    other_shell_temperatures = {
        200: 34.0,
        500: 45.0,
        700: 52.0,
        1100: 41.0,
        1500: 55.0,
    }
    temperature[labels == LABELS["Other"]] = other_shell_temperatures[shell]
    return temperature


def _simulate_dwi(
    rng: np.random.Generator,
    s0: np.ndarray,
    sigma: np.ndarray,
    labels: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
) -> np.ndarray:
    data = np.empty((*SHAPE, len(bvals)), dtype=np.float32)
    brain = s0 > 100.0
    csf = labels > 0
    tissue = brain & ~csf

    for volume, (bval, direction) in enumerate(zip(bvals, bvecs.T, strict=True)):
        if bval == 0:
            signal = s0.copy()
        else:
            adc = np.full(SHAPE, 0.00075, dtype=np.float64)
            tissue_adc = 0.00065 + 0.00028 * direction[1] ** 2
            adc[tissue] = tissue_adc
            shell_temperature = _temperature_for_shell(labels, int(bval))
            adc[csf] = temperature_to_adc(shell_temperature[csf])
            signal = s0 * np.exp(-bval * adc)

            other = labels == LABELS["Other"]
            flow_phase = 0.10 * np.sin(2.0 * np.pi * volume / 7.0)
            signal[other] *= np.exp(flow_phase * direction[1])

        real_noise = rng.normal(0.0, sigma)
        imaginary_noise = rng.normal(0.0, sigma)
        magnitude = np.sqrt((signal + real_noise) ** 2 + imaginary_noise**2)
        data[..., volume] = magnitude.astype(np.float32)

    return data


def _write_raw_metadata(root: Path, seed: int) -> None:
    _write_json(
        root / "dataset_description.json",
        {
            "Name": "Synthetic multi-shell DWI CSF thermometry example",
            "BIDSVersion": BIDS_VERSION,
            "DatasetType": "raw",
            "License": "CC0-1.0",
            "Authors": AUTHORS,
            "GeneratedBy": [
                {
                    "Name": "ThermoAmyloidSynthetic",
                    "Version": GENERATOR_VERSION,
                    "Description": (
                        "Analytic digital anatomy and simulated magnitude DWI; "
                        "no participant or template data were used."
                    ),
                }
            ],
            "Keywords": ["synthetic", "diffusion MRI", "CSF thermometry"],
        },
    )
    (root / "participants.tsv").write_text("participant_id\nsub-synthetic01\n")
    _write_json(
        root / "participants.json",
        {"participant_id": {"Description": "Synthetic participant identifier"}},
    )
    (root / "README.md").write_text(
        "# Synthetic multi-shell DWI CSF thermometry example\n\n"
        "This dataset contains one analytic digital subject. It contains no human "
        "images, metadata, atlas, or fitted participant parameters. The acquisition "
        "mirrors the study shell counts. The b=1500 shell is included to represent "
        "the acquisition but is excluded from the primary four-shell estimator.\n\n"
        f"Generation seed: `{seed}`.\n"
    )
    (root / "CHANGES").write_text(
        "1.0.0 2026-09-01\n  - Initial deterministic synthetic release.\n"
    )
    (root / "LICENSE").write_text(
        "To the extent possible under law, the authors have dedicated this synthetic "
        "dataset to the public domain under CC0 1.0.\n"
    )


def _write_derivatives(
    root: Path,
    sigma: np.ndarray,
    labels: np.ndarray,
    temperature: np.ndarray,
) -> None:
    derivative_root = root / "derivatives" / "thermometrysynthetic"
    derivative_dwi = derivative_root / SUBJECT / "dwi"
    derivative_dwi.mkdir(parents=True, exist_ok=True)
    source = f"bids:raw:{SUBJECT}/dwi/{SUBJECT}_acq-{ACQUISITION}_dwi.nii.gz"
    derivative_stem = f"{SUBJECT}_acq-{ACQUISITION}_space-orig"

    _write_json(
        derivative_root / "dataset_description.json",
        {
            "Name": "thermometrysynthetic ground-truth derivatives",
            "BIDSVersion": BIDS_VERSION,
            "DatasetType": "derivative",
            "License": "CC0-1.0",
            "Authors": AUTHORS,
            "DatasetLinks": {"raw": "../.."},
            "GeneratedBy": [
                {
                    "Name": "thermometrysynthetic",
                    "Version": GENERATOR_VERSION,
                    "Description": "Known digital-phantom masks, labels, noise, and temperature.",
                }
            ],
        },
    )
    (derivative_root / "README.md").write_text(
        "# Synthetic thermometry ground truth\n\n"
        "This derivative dataset contains analytic compartment labels, a CSF mask, "
        "and known noise and temperature maps for the synthetic raw DWI dataset. "
        "The non-standard quantitative `statmap` files are listed in `.bidsignore`; "
        "BIDS currently standardises the mask and segmentation but not these CSF "
        "thermometry quantities.\n"
    )
    (derivative_root / ".bidsignore").write_text(
        "*_statmap.json\n*_statmap.nii.gz\n"
    )

    csf_path = derivative_dwi / f"{derivative_stem}_label-CSF_mask.nii.gz"
    _write_nifti(csf_path, (labels > 0).astype(np.uint8), "Synthetic CSF mask")
    _write_json(
        csf_path.with_name(csf_path.name.removesuffix(".nii.gz") + ".json"),
        {
            "Description": "Union of all synthetic CSF candidate compartments",
            "Type": "ROI",
            "SpatialReference": "orig",
            "Sources": [source],
        },
    )

    dseg_path = (
        derivative_dwi
        / f"{derivative_stem}_seg-CSF_desc-compartments_dseg.nii.gz"
    )
    _write_nifti(dseg_path, labels.astype(np.uint8), "Synthetic CSF compartments")
    _write_json(
        dseg_path.with_name(dseg_path.name.removesuffix(".nii.gz") + ".json"),
        {
            "Description": "Known synthetic compartment labels",
            "Sources": [source],
            "SpatialReference": "orig",
            "Manual": False,
            "LabelMap": {
                "0": "Background",
                "1": "Lateral ventricles",
                "2": "Subarachnoid space",
                "3": "Other, with shell-dependent artefact",
            },
        },
    )
    dseg_path.with_name(dseg_path.name.removesuffix(".nii.gz") + ".tsv").write_text(
        "index\tname\tabbreviation\tcolor\n"
        "0\tBackground\tBG\t#000000\n"
        "1\tLateral ventricles\tLV\t#D17625\n"
        "2\tSubarachnoid space\tSA\t#367FB7\n"
        "3\tOther artefact-rich CSF\tOther\t#BE78A6\n"
    )

    noise_path = derivative_dwi / f"{derivative_stem}_desc-noise_statmap.nii.gz"
    _write_nifti(noise_path, sigma.astype(np.float32), "Synthetic Gaussian component sigma")
    _write_json(
        noise_path.with_name(noise_path.name.removesuffix(".nii.gz") + ".json"),
        {
            "Description": "Known standard deviation of each complex Gaussian noise component",
            "Units": "arbitrary",
            "StatisticType": "standard deviation",
            "Sources": [source],
        },
    )

    temperature_path = (
        derivative_dwi / f"{derivative_stem}_desc-temperature_statmap.nii.gz"
    )
    _write_nifti(
        temperature_path,
        temperature.astype(np.float32),
        "Synthetic target CSF temperature",
    )
    _write_json(
        temperature_path.with_name(
            temperature_path.name.removesuffix(".nii.gz") + ".json"
        ),
        {
            "Description": "Noise-free compartment target temperature",
            "Units": "degC",
            "Sources": [source],
        },
    )


def generate_dataset(output_root: str | Path, seed: int = 20260901, force: bool = False) -> dict:
    """Generate the complete raw and derivative synthetic BIDS dataset."""
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        if not force:
            raise FileExistsError(f"Output directory is not empty: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    _write_raw_metadata(root, seed)
    s0, sigma, labels, temperature = _digital_anatomy()
    bvals, bvecs = _gradient_table()
    rng = np.random.default_rng(seed)
    data = _simulate_dwi(rng, s0, sigma, labels, bvals, bvecs)

    dwi_dir = root / SUBJECT / "dwi"
    dwi_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{SUBJECT}_acq-{ACQUISITION}_dwi"
    dwi_path = dwi_dir / f"{stem}.nii.gz"
    _write_nifti(dwi_path, data, "Synthetic multi-shell magnitude DWI")
    np.savetxt(dwi_dir / f"{stem}.bval", bvals[None, :], fmt="%.0f")
    np.savetxt(dwi_dir / f"{stem}.bvec", bvecs, fmt="%.8f")
    _write_json(
        dwi_dir / f"{stem}.json",
        {
            "Modality": "MR",
            "MagneticFieldStrength": 3.0,
            "Manufacturer": "Synthetic",
            "ManufacturersModelName": "Analytic digital phantom",
            "PulseSequenceType": "Synthetic single-shot spin-echo EPI",
            "ScanningSequence": "EP",
            "SequenceVariant": "SK",
            "MRAcquisitionType": "2D",
            "RepetitionTime": 4.6,
            "EchoTime": 0.082,
            "FlipAngle": 90.0,
            "PhaseEncodingDirection": "j-",
            "EffectiveEchoSpacing": 0.00058,
            "TotalReadoutTime": 0.02726,
            "ReconMatrixPE": SHAPE[1],
            "ImageType": ["ORIGINAL", "PRIMARY", "DIFFUSION", "NONE"],
            "NoiseDistribution": "Rician magnitude",
            "SyntheticGenerationSeed": seed,
            "SyntheticTemperatureModel": {
                "ArrheniusA_K": ARRHENIUS_A_K,
                "ArrheniusB_mm2_per_s": ARRHENIUS_B_MM2_S,
                "CompartmentTemperatures_degC": TEMPERATURES_C,
            },
        },
    )
    _write_derivatives(root, sigma, labels, temperature)

    return {
        "bids_root": str(root),
        "subject": SUBJECT,
        "dwi": str(dwi_path),
        "n_volumes": int(len(bvals)),
        "shell_counts": SHELL_COUNTS.copy(),
        "seed": int(seed),
        "contains_human_data": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output BIDS root")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = generate_dataset(args.output, seed=args.seed, force=args.force)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
