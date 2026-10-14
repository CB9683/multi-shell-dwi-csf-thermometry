#!/usr/bin/env python3
"""Run the adopted mean-domain/shared-baseline method on analytic synthetic DWI."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import nibabel as nib
import numpy as np
import pandas as pd

from reproducibility.code.pipeline import mean_domain_core as core
from reproducibility.code.synthetic.generate_synthetic_bids import LABELS, SUBJECT, TEMPERATURES_C

PRIMARY_CONDITION='4s_mean_domain'
PRIMARY_SHELLS=(200,500,700,1100)
BRANCHES=('rice_mean_moments','quadratic_control')


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_output_dir(bids_root: str | Path, subject: str = SUBJECT) -> Path:
    return Path(bids_root).parent/'synthetic_mean_domain_outputs'/'rice_mean_moments'/subject


def _single_match(directory: Path, pattern: str) -> Path:
    matches=sorted(directory.glob(pattern))
    if len(matches)!=1:
        raise FileNotFoundError(f'Expected one file matching {pattern!r} in {directory}; found {len(matches)}')
    return matches[0]


def load_synthetic_bids(bids_root: str | Path, subject: str = SUBJECT) -> tuple[dict,np.ndarray]:
    """Load raw synthetic magnitude data and truth, without changing either."""
    root=Path(bids_root);dwi_dir=root/subject/'dwi'
    path=_single_match(dwi_dir,f'{subject}_*_dwi.nii.gz');stem=path.name.removesuffix('.nii.gz')
    bvals=np.loadtxt(dwi_dir/f'{stem}.bval')
    image=nib.load(path);data=np.asarray(image.dataobj,dtype=np.float64)
    if data.ndim!=4 or data.shape[-1]!=len(bvals):
        raise ValueError('DWI dimensions do not match b-values')
    signals={};counts={}
    for shell in (0,200,500,700,1100,1500):
        idx=np.flatnonzero(np.isclose(bvals,shell,atol=1.,rtol=0))
        if len(idx):signals[shell]=np.mean(data[...,idx],axis=-1);counts[shell]=len(idx)
    if set((0,*PRIMARY_SHELLS))-set(signals):
        raise ValueError('Missing required shells')
    derivatives=root/'derivatives/thermometrysynthetic'/subject/'dwi'
    prefix=f'{subject}_acq-thermometry_space-orig'
    csf=np.asarray(nib.load(derivatives/f'{prefix}_label-CSF_mask.nii.gz').dataobj)>0
    sigma=np.asarray(nib.load(derivatives/f'{prefix}_desc-noise_statmap.nii.gz').dataobj,dtype=np.float64)
    labels=np.asarray(nib.load(derivatives/f'{prefix}_seg-CSF_desc-compartments_dseg.nii.gz').dataobj,dtype=np.int16)
    if not(csf.shape==sigma.shape==labels.shape==data.shape[:3]):
        raise ValueError('Raw DWI and derivative grids do not match')
    return {'S':signals,'sigma':sigma,'csf_mask':csf,'affine':image.affine,
        'header':image.header.copy(),'N_VOL':counts,'bvals':bvals,'raw_path':path},labels


def prepare_synthetic_inputs(bids_root: str | Path, output_dir: str | Path, subject: str = SUBJECT) -> dict:
    """Run MP-PCA once, or reuse only an exact input/config/output hash match."""
    root=Path(bids_root);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    path=_single_match(root/subject/'dwi',f'{subject}_*_dwi.nii.gz')
    # Truth/geometry hashes prevent mixing a different generated subject with a cache.
    files=sorted(p for p in root.rglob('*') if p.is_file() and not p.name.startswith('._') and p.name!='.DS_Store')
    inputs={p.relative_to(root).as_posix():_sha(p) for p in files}
    config={'estimator':'Exp2','extent':[5,5,5],'nthreads':1,'includes_all_volumes':True,
        'noise_reference':'MP-PCA input magnitude SD','n4_bias_field':'not imposed; identity',
        'volume_drift':'not imposed; all applied factors one','geometric_preprocessing':'not imposed; native grid'}
    manifest=out/'preprocessing_manifest.json'
    if manifest.exists():
        metadata=json.loads(manifest.read_text())
        if metadata['input_sha256']!=inputs:
            raise ValueError('Input hash mismatch in shared synthetic preprocessing cache')
        if metadata['configuration']!=config:
            raise ValueError('Preprocessing configuration mismatch')
        for name,digest in metadata['output_sha256'].items():
            if _sha(out/name)!=digest:raise ValueError('Preprocessing output hash mismatch')
    else:
        command=shutil.which('dwidenoise')
        if not command and os.environ.get('MRTRIX_BIN'):
            candidate=Path(os.environ['MRTRIX_BIN'])/'dwidenoise'
            if candidate.is_file():command=str(candidate)
        if not command:raise FileNotFoundError('MP-PCA requires MRtrix3 dwidenoise (reference version 3.0.8)')
        version=subprocess.run([command,'-version'],capture_output=True,text=True,check=True).stdout.splitlines()[0].strip('= ')
        denoised=out/'dwi_denoised.nii.gz';noise=out/'magnitude_noise_sd.nii.gz'
        subprocess.run([command,str(path),str(denoised),'-noise',str(noise),'-estimator','Exp2',
            '-extent','5,5,5','-nthreads','1','-force','-quiet'],check=True,capture_output=True,text=True)
        metadata={'schema_version':1,'subject':subject,'input_sha256':inputs,'configuration':config,
            'estimator':'Exp2','includes_all_volumes':True,'mrtrix_version':version,
            'output_sha256':{p.name:_sha(p) for p in [denoised,noise]}}
        manifest.write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    image=nib.load(out/'dwi_denoised.nii.gz')
    data=np.asarray(image.dataobj,dtype=np.float64)
    sigma=np.asarray(nib.load(out/'magnitude_noise_sd.nii.gz').dataobj,dtype=np.float64)
    if data.shape[:3]!=sigma.shape or np.any(~np.isfinite(data)) or np.any(~np.isfinite(sigma)) or np.any(sigma<0):
        raise ValueError('Invalid MP-PCA signal/noise values or grids')
    return {'data':data,'sigma':sigma,'metadata':metadata,'manifest':manifest,'affine':image.affine}


def correct_denoised_volumes(data: np.ndarray, noise: np.ndarray, *, branch: str, bvals: np.ndarray) -> tuple[np.ndarray,dict]:
    """Correct every volume, including the baseline, with one shared noise map."""
    data=np.asarray(data,dtype=np.float64);noise=np.asarray(noise,dtype=np.float64)
    if branch not in BRANCHES:raise ValueError(f'Unknown correction branch: {branch}')
    if data.ndim!=4 or data.shape[:3]!=noise.shape or data.shape[-1]!=len(bvals):
        raise ValueError('Correction signal/noise/b-value shapes do not agree')
    if np.any(~np.isfinite(data)) or np.any(~np.isfinite(noise)) or np.any(noise<0):
        raise ValueError('Correction inputs must be finite with nonnegative noise')
    corrected=np.empty_like(data)
    audit={'branch':branch,'n_volumes':data.shape[-1],'includes_b0':bool(np.any(np.isclose(bvals,0))),
        'n_b0_corrected':int(np.sum(np.isclose(bvals,0))),'reference_sd_unchanged_by_signal_correction':True}
    for volume in range(data.shape[-1]):
        if branch=='rice_mean_moments':
            corrected[...,volume],counts=core.magnitude_mean_correction(data[...,volume],noise)
            for key,value in counts.items():audit[key]=audit.get(key,0)+value
        else:corrected[...,volume]=np.sqrt(np.maximum(data[...,volume]**2-2*noise**2,0))
    return corrected,audit


def _clusters(state: dict, coordinates: np.ndarray, min_cluster_size: int) -> tuple[np.ndarray,int]:
    """Geometry/temperature diagnostic; clustering never changes null weights."""
    import hdbscan
    from sklearn.preprocessing import StandardScaler
    valid=state['valid'];labels=np.full(len(valid),-2,dtype=np.int32)
    features=np.column_stack([state['temperature'][valid],state['mad'][valid],coordinates[valid]])
    if len(features):
        labels[valid]=hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=10,
            metric='euclidean',cluster_selection_method='eom').fit_predict(StandardScaler().fit_transform(features))
    n_clusters=len(set(labels[valid])-{-1})
    return labels,n_clusters


def _regional_summary(state: dict, labels: np.ndarray, primary: np.ndarray, challenge: np.ndarray) -> pd.DataFrame:
    rows=[]
    for region,label in LABELS.items():
        mask=labels==label;valid=mask&state['valid']
        selected=mask&(primary if region!='Other' else challenge)
        values=state['temperature'][selected];n=int(len(values));n_valid=int(valid.sum())
        rows.append({'region':region,'label':label,'selection_role':'primary regional estimate' if region!='Other' else 'separate same-rule challenge',
            'eligible_for_regional_statistics':region!='Other','target_temperature_c':TEMPERATURES_C[region],
            'n_total':int(mask.sum()),'n_valid':n_valid,'n_retained':n,
            'retained_fraction':n/n_valid if n_valid else np.nan,
            'temperature_mean_retained_c':float(np.mean(values)) if n else np.nan,
            'temperature_median_retained_c':float(np.median(values)) if n else np.nan,
            'temperature_sd_retained_c':float(np.std(values,ddof=1)) if n>1 else np.nan,
            'mad_median_c':float(np.median(state['mad'][valid])) if n_valid else np.nan,
            'sigma_temperature_rms_median_c':float(np.median(state['sigma_rms'][valid])) if n_valid else np.nan})
    frame=pd.DataFrame(rows)
    frame['temperature_bias_retained_c']=frame.temperature_mean_retained_c-frame.target_temperature_c
    return frame


def _cluster_summary(state: dict, truth: np.ndarray, clusters: np.ndarray, selected: np.ndarray) -> pd.DataFrame:
    rows=[]
    for cluster in sorted(set(clusters)-{-2}):
        mask=clusters==cluster;counts={r:int(np.sum(mask&(truth==label))) for r,label in LABELS.items()}
        total=int(mask.sum());dominant=max(counts,key=counts.get)
        rows.append({'cluster':int(cluster),'cluster_kind':'HDBSCAN noise' if cluster==-1 else 'cluster',
            'n':total,'n_retained':int(np.sum(mask&selected)),'dominant_truth_region':dominant,
            'truth_purity':counts[dominant]/total,**{f'n_{r}':v for r,v in counts.items()}})
    return pd.DataFrame(rows)


def _save_map(path: Path, vector: np.ndarray, mask: np.ndarray, affine: np.ndarray, *, boolean: bool=False) -> None:
    full=np.zeros(mask.shape,dtype=np.uint8) if boolean else np.full(mask.shape,np.nan,dtype=np.float64)
    full[mask]=vector.astype(full.dtype)
    nib.save(nib.Nifti1Image(full,affine),path)


def run_example(bids_root: str | Path, output_dir: str | Path | None=None, subject: str=SUBJECT,
                n_sim: int=4096, min_cluster_size: int=250, make_plot: bool=True,
                *, branch: str='rice_mean_moments', preprocessing_dir: str | Path | None=None) -> dict:
    """Verify the adopted classifier on LV/SA and a separate Other challenge."""
    if n_sim<2 or n_sim&(n_sim-1):raise ValueError('n_sim must be a positive power of two, at least two')
    root=Path(bids_root);out=Path(output_dir) if output_dir is not None else default_output_dir(root,subject)
    if output_dir is None and branch!='rice_mean_moments':out=out.parents[1]/branch/subject
    out.mkdir(parents=True,exist_ok=True)
    prep_dir=Path(preprocessing_dir) if preprocessing_dir is not None else out.parents[1]/'preprocessing'
    inputs,truth=load_synthetic_bids(root,subject)
    prepared=prepare_synthetic_inputs(root,prep_dir,subject)
    np.testing.assert_allclose(prepared['affine'],inputs['affine'],atol=1e-8,rtol=0)
    corrected,audit=correct_denoised_volumes(prepared['data'],prepared['sigma'],branch=branch,bvals=inputs['bvals'])
    csf=inputs['csf_mask'];coordinates=np.argwhere(csf);truth_vector=truth[csf]
    reference=prepared['sigma'][csf]
    if np.any(reference<=0):raise ValueError('Candidate CSF requires positive estimated input noise')
    groups={b:np.flatnonzero(np.isclose(inputs['bvals'],b,atol=1.,rtol=0)) for b in inputs['S']}
    means={b:np.mean(corrected[...,indices],axis=-1) for b,indices in groups.items()}
    baseline_sd=core.shell_mean_noise_sd(reference,np.ones(len(groups[0])))
    shell_sd={b:core.shell_mean_noise_sd(reference,np.ones(len(groups[b]))) for b in PRIMARY_SHELLS}
    state=core.assemble_scaled_covariance_state(s0=means[0][csf],shell_signals={b:means[b][csf] for b in PRIMARY_SHELLS},
        baseline_mean_sd=baseline_sd,shell_mean_sd=shell_sd)
    state['region']=np.where(truth_vector==LABELS['LV'],'LV',np.where(truth_vector==LABELS['SA'],'SA',''))
    powers=SimpleNamespace(coarse_power=int(np.log2(n_sim)),refined_power=15,ultra_power=17,resolve_power=19,final_power=21)
    normals=core.make_qmc_normals(powers,5)
    primary,null,diagnostic=core.compute_covariance_conditioned(state,normals,.015,.002)
    other_state={**state,'region':np.where(truth_vector==LABELS['Other'],'Other','')}
    challenge,other_null,other_diagnostic=core.compute_covariance_conditioned(other_state,normals,.015,.002)
    evaluated_null=np.where(np.isfinite(null),null,other_null)
    primary_eligible=state['valid']&(state['region']!='')&(state['temperature']>=25)
    challenge_eligible=state['valid']&(truth_vector==LABELS['Other'])&(state['temperature']>=25)
    selected=primary|challenge
    clusters,n_clusters=_clusters(state,coordinates,min_cluster_size)
    summary=_regional_summary(state,truth_vector,primary,challenge)
    summary.to_csv(out/'regional_ground_truth_summary.tsv',sep='\t',index=False)
    overlap=_cluster_summary(state,truth_vector,clusters,selected)
    overlap.to_csv(out/'cluster_truth_overlap.tsv',sep='\t',index=False)
    state_saved={k:v for k,v in state.items() if k!='region'}
    np.savez_compressed(out/'candidate_state.npz',**state_saved,coordinates=coordinates,truth_labels=truth_vector,
        primary_eligible=primary_eligible,challenge_eligible=challenge_eligible,primary_retained=primary,
        other_challenge_retained=challenge,null_median=null,other_challenge_null_median=other_null,
        cluster_labels=clusters,reference_sd=reference,baseline_mean_sd=baseline_sd,
        shell_mean_sd=np.column_stack([shell_sd[b] for b in PRIMARY_SHELLS]))
    condition=PRIMARY_CONDITION if branch=='rice_mean_moments' else '4s_quadratic_control'
    maps={'temperature':state['temperature'],'cross_shell_mad':state['mad'],'observed_z':state['z'],
        'null_median':evaluated_null,'sigma_temperature_rms':state['sigma_rms'],
        'primary_retained':primary,'other_challenge_retained':challenge,'classifier_evaluated':primary_eligible|challenge_eligible}
    for name,vector in maps.items():
        _save_map(out/f'{name}_{condition}.nii.gz',vector,csf,inputs['affine'],boolean=vector.dtype==bool)
    for b,mean in means.items():nib.save(nib.Nifti1Image(mean,inputs['affine']),out/f'mean_signal_b{b}.nii.gz')
    indexed=summary.set_index('region')
    gradient=float(indexed.loc['LV','temperature_mean_retained_c']-indexed.loc['SA','temperature_mean_retained_c'])
    source_files=[Path(__file__),Path(core.__file__),Path(core.__file__).with_suffix('.provenance.json')]
    analysis_folder=Path(core.__file__).parents[1]/'analysis'
    source_files+=[analysis_folder/'voxel_null_core.py',analysis_folder/'thermometry_precision_noise.py']
    software={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','nibabel','hdbscan','scikit-learn']}
    software['python']=platform.python_version();software['mrtrix']=prepared['metadata']['mrtrix_version']
    metadata={'schema_version':2,'subject':subject,'condition':condition,'branch':branch,
        'signal_correction':branch,'correction_applied_before_shell_means':True,'baseline_corrected':True,
        'classifier':'voxel-specific shared-b0 median-null','classification_rule':'eligible AND z_observed < own null median',
        'null_factors':5,'null_recalibrated_by_cluster':False,'minimum_median_temperature_c':25.,
        'primary_region_scope':['LV','SA'],'other_is_separate_classifier_challenge':True,
        'shells_used':list(PRIMARY_SHELLS),'shells_present_but_excluded':[1500],
        'n_sim':int(n_sim),'qmc_powers':vars(powers),'qmc_seeds':[1729,2718,31415,92653,161803,141421,173205],
        'refinement_margin':.015,'ultra_margin':.002,'primary_classification_diagnostics':diagnostic,
        'other_challenge_diagnostics':other_diagnostic,'min_cluster_size':min_cluster_size,'n_clusters':n_clusters,
        'target_lv_minus_sa_c':TEMPERATURES_C['LV']-TEMPERATURES_C['SA'],'recovered_lv_minus_sa_c':gradient,
        'noise_reference':'estimated MP-PCA input magnitude SD; unit N4/drift factors; not residual SD',
        'complex_noise_truth_used_for_classification':False,'applied_volume_scales':[1.]*len(inputs['bvals']),
        'preprocessing_scope':prepared['metadata']['configuration'],'preprocessing_manifest_sha256':_sha(prepared['manifest']),
        'raw_input_sha256':prepared['metadata']['input_sha256'],'correction_audit':audit,
        'source_sha256':{p.relative_to(Path(__file__).resolve().parents[3]).as_posix():_sha(p) for p in source_files},
        'software':software,'uses_human_data':False,
        'interpretation':'Known-input implementation demonstration; not physiological or final residual-noise validation'}
    metadata['artifact_sha256']={p.name:_sha(p) for p in sorted(out.iterdir()) if p.is_file() and (p.name.endswith('.nii.gz') or p.suffix in ['.npz','.tsv'])}
    (out/'analysis_metadata.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    if make_plot:
        from reproducibility.code.synthetic.plot_synthetic_validation import load_figure_data,write_figure
        write_figure(load_figure_data(root,out,subject),out/'figure')
    return {**metadata,'output_dir':out,'regional_summary':summary,'cluster_truth_overlap':overlap}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bids-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--preprocessing-dir',type=Path)
    parser.add_argument('--subject',default=SUBJECT)
    parser.add_argument('--branch',choices=BRANCHES,default='rice_mean_moments')
    parser.add_argument('--n-sim',type=int,default=4096)
    parser.add_argument('--min-cluster-size',type=int,default=250)
    parser.add_argument('--no-plot',action='store_true')
    args=parser.parse_args()
    outcome=run_example(args.bids_root,args.output_dir,args.subject,args.n_sim,args.min_cluster_size,
        not args.no_plot,branch=args.branch,preprocessing_dir=args.preprocessing_dir)
    print(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in outcome.items()
        if k not in ['regional_summary','cluster_truth_overlap']},indent=2,sort_keys=True))
    print(outcome['regional_summary'].to_string(index=False))


if __name__=='__main__':main()
