import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from reproducibility.code.synthetic.generate_synthetic_bids import (
    ARRHENIUS_A_K,
    ARRHENIUS_B_MM2_S,
    SHELL_COUNTS,
    TEMPERATURES_C,
    generate_dataset,
    temperature_to_adc,
)


SUBJECT = "sub-synthetic01"
STEM = f"{SUBJECT}_acq-thermometry"
DERIVATIVE_STEM = f"{STEM}_space-orig"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def synthetic_bids(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-bids") / "dataset"
    manifest = generate_dataset(root, seed=20260901)
    return root, manifest


def test_creates_raw_and_derivative_bids_layout(synthetic_bids):
    root, manifest = synthetic_bids

    description = json.loads((root / "dataset_description.json").read_text())
    assert description["BIDSVersion"] == "1.11.1"
    assert description["DatasetType"] == "raw"
    assert description["GeneratedBy"][0]["Name"] == "ThermoAmyloidSynthetic"
    assert manifest["contains_human_data"] is False

    dwi_dir = root / SUBJECT / "dwi"
    for extension in (".nii.gz", ".json", ".bval", ".bvec"):
        assert (dwi_dir / f"{STEM}_dwi{extension}").is_file()

    derivative_root = root / "derivatives" / "thermometrysynthetic"
    derivative_description = json.loads(
        (derivative_root / "dataset_description.json").read_text()
    )
    assert derivative_description["DatasetType"] == "derivative"
    assert derivative_description["GeneratedBy"][0]["Name"] == "thermometrysynthetic"

    derivative_dwi = derivative_root / SUBJECT / "dwi"
    assert (
        derivative_dwi / f"{DERIVATIVE_STEM}_label-CSF_mask.nii.gz"
    ).is_file()
    mask_metadata = json.loads(
        (
            derivative_dwi / f"{DERIVATIVE_STEM}_label-CSF_mask.json"
        ).read_text()
    )
    assert mask_metadata["Type"] == "ROI"
    assert mask_metadata["SpatialReference"] == "orig"
    assert (
        derivative_dwi
        / f"{DERIVATIVE_STEM}_seg-CSF_desc-compartments_dseg.nii.gz"
    ).is_file()
    assert (
        derivative_dwi / f"{DERIVATIVE_STEM}_desc-noise_statmap.nii.gz"
    ).is_file()
    assert (
        derivative_dwi / f"{DERIVATIVE_STEM}_desc-temperature_statmap.nii.gz"
    ).is_file()
    assert (derivative_root / ".bidsignore").read_text().splitlines() == [
        "*_statmap.json",
        "*_statmap.nii.gz",
    ]


def test_acquisition_matches_study_shell_counts_and_gradients(synthetic_bids):
    root, _ = synthetic_bids
    dwi_dir = root / SUBJECT / "dwi"
    image = nib.load(dwi_dir / f"{STEM}_dwi.nii.gz")
    bvals = np.loadtxt(dwi_dir / f"{STEM}_dwi.bval")
    bvecs = np.loadtxt(dwi_dir / f"{STEM}_dwi.bvec")

    assert image.shape == (40, 48, 36, 109)
    assert image.get_data_dtype() == np.dtype(np.float32)
    assert bvals.shape == (109,)
    assert bvecs.shape == (3, 109)

    observed_counts = {
        shell: int(np.count_nonzero(np.isclose(bvals, shell)))
        for shell in SHELL_COUNTS
    }
    assert observed_counts == SHELL_COUNTS

    norms = np.linalg.norm(bvecs, axis=0)
    np.testing.assert_allclose(norms[bvals == 0], 0.0, atol=1e-8)
    np.testing.assert_allclose(norms[bvals > 0], 1.0, atol=1e-6)


def test_ground_truth_maps_encode_distinct_csf_compartments(synthetic_bids):
    root, _ = synthetic_bids
    derivative_dwi = (
        root / "derivatives" / "thermometrysynthetic" / SUBJECT / "dwi"
    )
    labels = nib.load(
        derivative_dwi
        / f"{DERIVATIVE_STEM}_seg-CSF_desc-compartments_dseg.nii.gz"
    ).get_fdata()
    csf = nib.load(
        derivative_dwi / f"{DERIVATIVE_STEM}_label-CSF_mask.nii.gz"
    ).get_fdata()
    temperature = nib.load(
        derivative_dwi / f"{DERIVATIVE_STEM}_desc-temperature_statmap.nii.gz"
    ).get_fdata()

    assert set(np.unique(labels).astype(int)) == {0, 1, 2, 3}
    np.testing.assert_array_equal(csf > 0, labels > 0)
    for label, region in ((1, "LV"), (2, "SA"), (3, "Other")):
        assert np.count_nonzero(labels == label) >= 500
        np.testing.assert_allclose(
            temperature[labels == label], TEMPERATURES_C[region], atol=1e-6
        )
    assert TEMPERATURES_C["LV"] > TEMPERATURES_C["SA"]


def test_arrhenius_inverse_recovers_configured_temperatures():
    for temperature_c in TEMPERATURES_C.values():
        adc = temperature_to_adc(temperature_c)
        recovered = (
            ARRHENIUS_A_K / np.log(ARRHENIUS_B_MM2_S / adc) - 273.15
        )
        assert recovered == pytest.approx(temperature_c, abs=1e-12)


def test_default_analysis_output_does_not_modify_raw_bids(tmp_path):
    from reproducibility.code.synthetic.run_synthetic_example import default_output_dir

    bids_root = tmp_path / "synthetic_bids"
    assert default_output_dir(bids_root, SUBJECT) == (
        tmp_path / "synthetic_mean_domain_outputs" / "rice_mean_moments" / SUBJECT
    )


def test_analysis_cli_is_runnable_from_repository_root(tmp_path):
    repository_root = Path(__file__).resolve().parents[2]
    script = (
        repository_root
        / "reproducibility"
        / "code"
        / "synthetic"
        / "run_synthetic_example.py"
    )
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--bids-root" in completed.stdout


def test_generation_is_byte_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset(first, seed=17)
    generate_dataset(second, seed=17)

    relative_files = (
        Path(SUBJECT) / "dwi" / f"{STEM}_dwi.nii.gz",
        Path(SUBJECT) / "dwi" / f"{STEM}_dwi.bval",
        Path(SUBJECT) / "dwi" / f"{STEM}_dwi.bvec",
        Path("derivatives")
        / "thermometrysynthetic"
        / SUBJECT
        / "dwi"
        / f"{DERIVATIVE_STEM}_seg-CSF_desc-compartments_dseg.nii.gz",
    )
    assert [_sha256(first / path) for path in relative_files] == [
        _sha256(second / path) for path in relative_files
    ]


def test_primary_analysis_recovers_compartment_order_and_rejects_other(
    synthetic_bids, tmp_path
):
    from reproducibility.code.synthetic.run_synthetic_example import run_example

    root, _ = synthetic_bids
    output = tmp_path / "analysis"
    result = run_example(
        bids_root=root,
        output_dir=output,
        n_sim=2048,
        min_cluster_size=250,
        make_plot=False,
    )

    summary = result["regional_summary"].set_index("region")
    lv = summary.loc["LV"]
    sa = summary.loc["SA"]
    other = summary.loc["Other"]

    assert lv["temperature_mean_retained_c"] == pytest.approx(
        TEMPERATURES_C["LV"], abs=0.8
    )
    assert sa["temperature_mean_retained_c"] == pytest.approx(
        TEMPERATURES_C["SA"], abs=0.8
    )
    assert (
        lv["temperature_mean_retained_c"] - sa["temperature_mean_retained_c"]
        > 1.5
    )
    assert other["mad_median_c"] > 5.0 * max(
        lv["mad_median_c"], sa["mad_median_c"]
    )
    assert other["retained_fraction"] < min(
        lv["retained_fraction"], sa["retained_fraction"]
    )
    assert result["n_clusters"] >= 3
    assert (output / "regional_ground_truth_summary.tsv").is_file()
    assert (output / "cluster_truth_overlap.tsv").is_file()
    assert (output / "temperature_4s_mean_domain.nii.gz").is_file()
