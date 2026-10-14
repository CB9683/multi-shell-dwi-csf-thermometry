"""Scientific contracts for the portable adopted synthetic implementation."""
from importlib.util import find_spec
from types import SimpleNamespace
import numpy as np
import pytest


def _core():
    assert find_spec('reproducibility.code.pipeline.mean_domain_core') is not None, 'The public example must expose the adopted mean-domain/shared-baseline model'
    from reproducibility.code.pipeline import mean_domain_core
    return mean_domain_core


def _signals(temperatures):
    temperatures=np.asarray(temperatures,float)
    adc=19.649*np.exp(-2732/(temperatures+273.15))
    baseline=np.linspace(850,1050,len(temperatures))
    return baseline,{b:baseline*np.exp(-b*adc) for b in (200,500,700,1100)}


def test_magnitude_inversion_recovers_known_rice_mean_and_noise():
    core=_core()
    from reproducibility.code.analysis.thermometry_precision_noise import rice_unit_moments
    amplitude=np.array([20.,60.,120.]);complex_sd=np.array([3.,7.,9.])
    unit_mean,unit_variance=rice_unit_moments(amplitude/complex_sd)
    corrected,_=core.magnitude_mean_correction(complex_sd*unit_mean,complex_sd*np.sqrt(unit_variance))
    np.testing.assert_allclose(corrected,amplitude,rtol=2e-7,atol=1e-6)


def test_unit_scale_matches_independent_shared_reference_error_formula():
    core=_core();temperatures=np.array([36.3,38.8]);s0,sb=_signals(temperatures)
    noise=np.array([7.,12.]);counts={0:11,200:20,500:20,700:20,1100:19}
    state=core.assemble_scaled_covariance_state(s0=s0,shell_signals=sb,
        baseline_mean_sd=core.shell_mean_noise_sd(noise,np.ones(counts[0])),
        shell_mean_sd={b:core.shell_mean_noise_sd(noise,np.ones(counts[b])) for b in sb})
    np.testing.assert_allclose(state['temperature'],temperatures,atol=2e-12)
    adc=19.649*np.exp(-2732/(temperatures+273.15));derivative=2732/(adc*np.log(19.649/adc)**2)
    for i,b in enumerate(sb):
        np.testing.assert_allclose(state['tau_stack'][:,i],derivative*noise/(b*s0*np.sqrt(counts[0])),rtol=1e-13)
        np.testing.assert_allclose(state['eta_stack'][:,i],derivative*noise/(b*sb[b]*np.sqrt(counts[b])),rtol=1e-13)


def test_unequal_scales_use_sum_of_squared_applied_factors():
    core=_core();factors=np.array([.8,1.,1.2]);sigma=np.array([5.,9.])
    expected=sigma*np.sqrt(.8**2+1.**2+1.2**2)/3
    np.testing.assert_allclose(core.shell_mean_noise_sd(sigma,factors),expected,rtol=1e-14)


def test_common_intensity_scaling_preserves_temperature_and_noise_coefficients():
    core=_core();s0,sb=_signals([36.3,38.8])
    def state(scale):
        return core.assemble_scaled_covariance_state(s0=s0*scale,
            shell_signals={b:v*scale for b,v in sb.items()},baseline_mean_sd=np.array([2.,3.])*scale,
            shell_mean_sd={b:np.array([1.5,2.5])*scale for b in sb})
    first,scaled=state(1.),state(1.7)
    for key in ['temperature_stack','tau_stack','eta_stack','sigma_rms','z']:
        np.testing.assert_allclose(first[key],scaled[key],rtol=1e-12,atol=1e-12)


def test_shared_reference_covariance_has_expected_off_diagonal_terms():
    core=_core();s0,sb=_signals([36.3,38.8])
    state=core.assemble_scaled_covariance_state(s0=s0,shell_signals=sb,
        baseline_mean_sd=np.array([2.,3.]),shell_mean_sd={b:np.array([1.5,2.5]) for b in sb})
    from reproducibility.code.analysis.voxel_null_core import temperature_covariance
    actual=temperature_covariance(state['tau_stack'],state['eta_stack'])
    for i in range(2):
        expected=np.outer(state['tau_stack'][i],state['tau_stack'][i])+np.diag(state['eta_stack'][i]**2)
        np.testing.assert_allclose(actual[i],expected,rtol=1e-14)
        assert actual[i,0,1]>0


def test_classifier_applies_own_null_and_primary_eligibility():
    core=_core();s0,sb=_signals([37.,24.,38.,37.])
    state=core.assemble_scaled_covariance_state(s0=s0,shell_signals=sb,
        baseline_mean_sd=np.full(4,2.),shell_mean_sd={b:np.full(4,2.) for b in sb})
    state['region']=np.array(['LV','LV','SA',''])
    state['z']=np.array([0.,0.,10.,0.])
    parameters=SimpleNamespace(coarse_power=8,refined_power=10,ultra_power=12,resolve_power=14,final_power=16)
    normals=core.make_qmc_normals(parameters,5)
    retained,threshold,diagnostic=core.compute_covariance_conditioned(state,normals,.015,.002)
    np.testing.assert_array_equal(retained,[True,False,False,False])
    assert diagnostic['n_eligible']==2
    assert np.all(np.isfinite(threshold[[0,2]]))
    assert np.all(np.isnan(threshold[[1,3]]))
    expected=np.zeros(4,bool);expected[[0,2]]=state['z'][[0,2]]<threshold[[0,2]]
    np.testing.assert_array_equal(retained,expected)


def test_synthetic_runner_names_the_corrected_primary_condition():
    from reproducibility.code.synthetic.run_synthetic_example import PRIMARY_CONDITION
    assert PRIMARY_CONDITION=='4s_mean_domain'


@pytest.mark.parametrize('branch',['rice_mean_moments','quadratic_control'])
def test_volume_correction_includes_baseline_and_uses_same_noise_reference(branch):
    from reproducibility.code.synthetic import run_synthetic_example as runner
    assert hasattr(runner,'correct_denoised_volumes'),'Correct per-volume denoised magnitudes before all shell means'
    core=_core();data=np.array([[[[100.,70.,40.]]]]);noise=np.array([[[5.]]])
    corrected,audit=runner.correct_denoised_volumes(data,noise,branch=branch,bvals=np.array([0,200,500]))
    if branch=='quadratic_control':expected=np.sqrt(data**2-2*noise[...,None]**2)
    else:expected=np.stack([core.magnitude_mean_correction(data[...,i],noise)[0] for i in range(3)],axis=-1)
    np.testing.assert_allclose(corrected,expected,rtol=1e-14)
    assert corrected[...,0].item()<data[...,0].item()
    assert audit['n_volumes']==3 and audit['includes_b0']


def test_synthetic_preprocessing_cache_detects_changed_raw_input(tmp_path):
    from reproducibility.code.synthetic import run_synthetic_example as runner
    assert hasattr(runner,'prepare_synthetic_inputs'),'Raw input hashes must guard shared MP-PCA cache reuse'
    from reproducibility.code.synthetic.generate_synthetic_bids import generate_dataset,SUBJECT
    root=tmp_path/'synthetic_bids';generate_dataset(root,seed=17)
    folder=tmp_path/'preprocessing'
    first=runner.prepare_synthetic_inputs(root,folder)
    assert first['metadata']['estimator']=='Exp2'
    assert first['metadata']['includes_all_volumes']
    dwi=root/SUBJECT/'dwi'
    path=next(dwi.glob('*.bval'));path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='input.*hash|Input.*hash'):
        runner.prepare_synthetic_inputs(root,folder)
